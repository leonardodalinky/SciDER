"""DataSciBench internal plumbing — used only by ``generation``.

* ``Task`` / ``discover_tasks`` — walk a DataSciBench-data dir and yield one
  task per ``<family>_<id>`` subfolder (3 families: csv_excel / dl / human).
* ``stage_inputs`` — copy every file under the subfolder (recursively) except
  ``prompt.json`` into ``<workspace>/inputs/`` so the agent sees them as
  read-only inputs.
* ``run_full_workflow_task`` — invoke ``run_full_workflow`` on a single task
  (data + experiment, no ideation, no paper writing). The sub-workflows save
  ``data_agent_history.json`` and ``experiment_agent_history.json`` into the
  workspace root themselves (crash-safe via ``capture_messages``).
* ``is_workspace_complete`` / ``scan_completed_uids`` / ``write_output_json``
  — skip-existing helpers mirroring ds1000/aiidea.

If a fourth benchmark needs the same primitives, lift the shared bits up to
``data_generation/_common.py`` then.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from loguru import logger

# scider is on sys.path because the entry-point modules
# (data_generation.datascibench.generation) prepend the project root before
# importing this module.
from scider.workflows.full_workflow import FullWorkflow, run_full_workflow

OUTPUT_FILENAME = "output.json"
DATA_HISTORY_FILENAME = "data_agent_history.json"
EXP_HISTORY_FILENAME = "experiment_agent_history.json"
PROMPT_FILENAME = "prompt.md"
INPUTS_SUBDIR = "inputs"

# Files that live IN the subfolder but are metadata, not inputs the agent
# should see. ``prompt.json`` is the upstream prompt definition; we read it
# in Python and wrap it ourselves rather than letting the agent re-read it.
_METADATA_FILENAMES = {"prompt.json"}

# Subfolders are named ``<family>_<id>`` where family ∈ {csv_excel, dl, human}.
_TASK_DIR_RE = re.compile(r"^(csv_excel|dl|human)_(\d+)$")


# --------------------------------------------------------------------------- #
# Task discovery                                                              #
# --------------------------------------------------------------------------- #


TaskFamily = Literal["csv_excel", "dl", "human"]


@dataclass
class Task:
    """One unit of work: one ``<family>_<id>/`` subfolder."""

    uid: str
    family: TaskFamily
    task_id: int
    task_dir: Path
    prompt: str  # raw prompt text from prompt.json
    data_source_type: str  # e.g. "1=no dependency"


def _build_uid(family: str, task_id: int | str) -> str:
    return f"datascibench_{family}_{task_id}"


def discover_tasks(bench_root: Path) -> Iterator[Task]:
    """Walk ``bench_root`` (the dir that contains ``csv_excel_*/``, ``dl_*/``,
    ``human_*/`` subfolders) and yield one ``Task`` per qualifying subfolder.

    Top-level files like ``csv_excel_prompt.csv`` are skipped — every per-task
    prompt is duplicated inside that task's own ``prompt.json``.
    """
    if not bench_root.is_dir():
        raise FileNotFoundError(f"DataSciBench root not a directory: {bench_root}")

    for entry in sorted(bench_root.iterdir()):
        if not entry.is_dir():
            continue
        m = _TASK_DIR_RE.match(entry.name)
        if not m:
            continue
        family = m.group(1)  # type: ignore[assignment]
        task_id = int(m.group(2))

        prompt_json = entry / "prompt.json"
        if not prompt_json.is_file():
            logger.warning("Skipping {} — no prompt.json", entry)
            continue
        try:
            payload = json.loads(prompt_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping {} — malformed prompt.json", entry)
            continue
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            logger.warning("Skipping {} — empty prompt field", entry)
            continue

        yield Task(
            uid=_build_uid(family, task_id),
            family=family,  # type: ignore[arg-type]
            task_id=task_id,
            task_dir=entry,
            prompt=prompt,
            data_source_type=str(payload.get("data_source_type", "")).strip(),
        )


# --------------------------------------------------------------------------- #
# Workspace staging                                                           #
# --------------------------------------------------------------------------- #


def stage_inputs(workspace: Path, task_dir: Path) -> list[str]:
    """Copy everything under ``task_dir`` (except ``prompt.json``) into
    ``<workspace>/inputs/`` recursively. Returns the list of relative paths
    (relative to ``inputs/``) that were staged — handy for the prompt wrapper.

    We copy (not symlink) so the agent's workspace is self-contained; this
    mirrors the discoverybench behaviour for read-only data files. CSV / xlsx
    payloads here are small (< 1 MB typical), so the disk overhead is fine.
    """
    inputs_dir = workspace / INPUTS_SUBDIR
    if inputs_dir.exists():
        # Remove and re-stage so re-runs see a clean inputs/. Don't touch
        # other workspace files (output.json, agent histories, etc.).
        shutil.rmtree(inputs_dir)
    inputs_dir.mkdir(parents=True)

    staged: list[str] = []
    for src in task_dir.iterdir():
        if src.name in _METADATA_FILENAMES:
            continue
        dst = inputs_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
            for inner in dst.rglob("*"):
                if inner.is_file():
                    staged.append(str(inner.relative_to(inputs_dir)))
        else:
            shutil.copy2(src, dst)
            staged.append(src.name)
    return staged


# --------------------------------------------------------------------------- #
# Skip-existing helpers                                                       #
# --------------------------------------------------------------------------- #


def is_workspace_complete(workspace: Path) -> bool:
    """A workspace is "done" iff:
    - ``output.json`` exists, parses, and has ``ok == True``
    - both ``data_agent_history.json`` and ``experiment_agent_history.json``
      exist and are non-empty (the trajectories are what we keep for SFT)
    """
    out = workspace / OUTPUT_FILENAME
    data_hist = workspace / DATA_HISTORY_FILENAME
    exp_hist = workspace / EXP_HISTORY_FILENAME
    try:
        if not out.is_file() or out.stat().st_size == 0:
            return False
        data = json.loads(out.read_text(encoding="utf-8"))
        if not data.get("ok"):
            return False
    except (OSError, json.JSONDecodeError):
        return False
    for hist in (data_hist, exp_hist):
        if not hist.is_file() or hist.stat().st_size == 0:
            return False
    return True


def scan_completed_uids(output_root: Path) -> set[str]:
    """Used by ``--skip-existing``: returns uids whose workspace passes
    ``is_workspace_complete``."""
    if not output_root.is_dir():
        return set()
    completed: set[str] = set()
    for uid_dir in output_root.iterdir():
        if not uid_dir.is_dir():
            continue
        if is_workspace_complete(uid_dir):
            completed.add(uid_dir.name)
    return completed


def write_output_json(workspace: Path, payload: dict[str, Any]) -> None:
    """Write the per-task summary stub atomically (tmp + rename)."""
    out = workspace / OUTPUT_FILENAME
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)


# --------------------------------------------------------------------------- #
# Full-workflow runner                                                        #
# --------------------------------------------------------------------------- #


def run_full_workflow_task(
    *,
    user_query: str,
    workspace_dir: Path,
    data_path: Path | None,
    data_desc: str,
    max_revisions: int = 1,
    data_recursion_limit: int = 80,
    experiment_recursion_limit: int = 100,
) -> FullWorkflow:
    """Run one DataSciBench task through SciDER FullWorkflow (data +
    experiment, no ideation, no paper writing). Returns the executed
    ``FullWorkflow`` object; ``data_agent_history.json`` and
    ``experiment_agent_history.json`` are persisted into ``workspace_dir``
    by the sub-workflows themselves even on crash.

    ``data_path=None`` is valid for csv_excel tasks where the prompt embeds
    the dataset inline and there are no input files to inspect.
    """
    workspace_dir = Path(workspace_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running FullWorkflow in {} (data_path={})", workspace_dir, data_path)
    workflow = run_full_workflow(
        workspace_path=workspace_dir,
        user_query=user_query,
        data_path=data_path,
        data_desc=data_desc,
        max_revisions=max_revisions,
        data_agent_recursion_limit=data_recursion_limit,
        experiment_agent_recursion_limit=experiment_recursion_limit,
        user_approval_enabled=False,
        run_paper_writing=False,
    )
    return workflow
