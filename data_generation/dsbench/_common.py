"""DSBench internal plumbing — used only by ``generation``.

Two task families share this module:

* **analysis** — finance/business spreadsheet QA. Per-task layout under
  ``<bench_root>/data_analysis/data/<id>/``: one ``*.xlsx`` + ``introduction.txt``
  + N ``question{idx}.txt`` files. Truth lives in
  ``data_analysis/data.json`` (we never show it to the agent).

* **modeling** — Kaggle-style ML competitions. Per-task layout under
  ``<bench_root>/data_modeling/data/``:
    - ``task/<comp>.txt``            description (we DO show)
    - ``data_resplit/<comp>/``       train.csv + test.csv + sampleSubmission.csv
    - ``answers/<comp>/test_answer.csv``  held-out gold (we DO NOT show)

Public API:
* ``Task`` / ``discover_analysis_tasks`` / ``discover_modeling_tasks``
* ``stage_analysis_inputs`` / ``stage_modeling_inputs`` (both symlink — datasets
  here can be hundreds of MB and live on the same NFS volume as the workspace)
* ``run_full_workflow_task``
* ``is_workspace_complete`` / ``scan_completed_uids`` / ``write_output_json``
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from loguru import logger

from scider.workflows.full_workflow import FullWorkflow, run_full_workflow

OUTPUT_FILENAME = "output.json"
DATA_HISTORY_FILENAME = "data_agent_history.json"
EXP_HISTORY_FILENAME = "experiment_agent_history.json"
PROMPT_FILENAME = "prompt.md"
INPUTS_SUBDIR = "inputs"

TaskFamily = Literal["analysis", "modeling"]

# Submission filename the modeling agent should write. Kaggle convention.
MODELING_SUBMISSION_FILENAME = "submission.csv"
# Filename the analysis agent should write — JSON of {questionN: answer}.
ANALYSIS_ANSWERS_FILENAME = "pred_answers.json"


# --------------------------------------------------------------------------- #
# Task discovery                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class Task:
    """One DSBench task (analysis OR modeling)."""

    uid: str
    family: TaskFamily
    task_id: str  # "00000001" for analysis; "<comp>" for modeling
    task_dir: Path  # source dir to symlink (analysis: <id>/, modeling: data_resplit/<comp>/)
    # Family-specific extra context — populated by the family's discover fn.
    name: str = ""
    year: int | None = None
    introduction: str = ""  # analysis only — introduction.txt content
    question_files: list[Path] = field(default_factory=list)  # analysis only
    question_ids: list[str] = field(default_factory=list)  # analysis only — ["question6", ...]
    xlsx_filename: str = ""  # analysis only — bare filename, no path
    description: str = ""  # modeling only — task/<comp>.txt content


def _slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return text or "unknown"


def _build_uid(family: TaskFamily, task_id: str) -> str:
    return f"dsbench_{family}_{_slug(task_id)}"


# ---- Analysis ------------------------------------------------------------- #


def discover_analysis_tasks(bench_root: Path) -> Iterator[Task]:
    """Walk ``<bench_root>/data_analysis/data/`` for per-id subdirs and pair
    each with its row in ``data_analysis/data.json``. The data.json's
    ``answers`` field is the held-out gold — we deliberately do NOT carry
    it onto the Task so it cannot leak into the prompt by accident.
    """
    analysis_root = bench_root / "data_analysis"
    data_dir = analysis_root / "data"
    data_json = analysis_root / "data.json"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Analysis data dir missing: {data_dir}")
    if not data_json.is_file():
        raise FileNotFoundError(f"Analysis data.json missing: {data_json}")

    # Index data.json by id. It's jsonl-style despite the .json extension.
    by_id: dict[str, dict] = {}
    for line in data_json.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in row:
            by_id[str(row["id"])] = row

    for task_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        tid = task_dir.name
        meta = by_id.get(tid, {})

        # introduction.txt is the task framing the agent sees.
        intro_path = task_dir / "introduction.txt"
        introduction = (
            intro_path.read_text(encoding="utf-8", errors="replace") if intro_path.is_file() else ""
        )

        # Question files: prefer the order recorded in data.json, fall back to
        # filesystem listing if data.json missing.
        question_ids: list[str] = list(meta.get("questions") or [])
        if not question_ids:
            question_ids = sorted(
                p.stem
                for p in task_dir.iterdir()
                if p.name.startswith("question") and p.suffix == ".txt"
            )
        question_files = [task_dir / f"{q}.txt" for q in question_ids]

        # Identify the spreadsheet (any xlsx in the dir; usually exactly one).
        xlsxs = sorted(p.name for p in task_dir.iterdir() if p.suffix.lower() in (".xlsx", ".xls"))
        xlsx_filename = xlsxs[0] if xlsxs else ""

        if not question_ids or not xlsx_filename:
            logger.warning(
                "[dsbench_analysis_{}] skipping — questions={} xlsx={!r}",
                tid,
                len(question_ids),
                xlsx_filename,
            )
            continue

        yield Task(
            uid=_build_uid("analysis", tid),
            family="analysis",
            task_id=tid,
            task_dir=task_dir,
            name=str(meta.get("name") or ""),
            year=int(meta["year"]) if isinstance(meta.get("year"), int) else None,
            introduction=introduction,
            question_files=question_files,
            question_ids=question_ids,
            xlsx_filename=xlsx_filename,
        )


# ---- Modeling ------------------------------------------------------------- #


def discover_modeling_tasks(bench_root: Path) -> Iterator[Task]:
    """Walk ``<bench_root>/data_modeling/data/task/<comp>.txt`` and yield one
    Task per competition that ALSO has a ``data_resplit/<comp>/`` directory
    (otherwise the task isn't runnable — the data wasn't preprocessed).
    """
    modeling_root = bench_root / "data_modeling" / "data"
    task_dir = modeling_root / "task"
    resplit_dir = modeling_root / "data_resplit"
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Modeling task dir missing: {task_dir}")
    if not resplit_dir.is_dir():
        raise FileNotFoundError(f"Modeling data_resplit dir missing: {resplit_dir}")

    for task_txt in sorted(task_dir.iterdir()):
        if task_txt.suffix != ".txt":
            continue
        comp = task_txt.stem
        comp_data = resplit_dir / comp
        if not comp_data.is_dir():
            logger.debug("[dsbench_modeling_{}] skipping — no data_resplit/{} dir", comp, comp)
            continue

        description = task_txt.read_text(encoding="utf-8", errors="replace")
        yield Task(
            uid=_build_uid("modeling", comp),
            family="modeling",
            task_id=comp,
            task_dir=comp_data,
            name=comp,
            description=description,
        )


# --------------------------------------------------------------------------- #
# Workspace staging                                                           #
# --------------------------------------------------------------------------- #


def _replace_link(link: Path, target: Path) -> None:
    """Idempotent symlink replacement (handles symlink, file, dir cases)."""
    import shutil

    if link.is_symlink():
        link.unlink()
    elif link.exists():
        if link.is_dir():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def stage_analysis_inputs(workspace: Path, task: Task) -> None:
    """Symlink the per-task analysis dir into ``<workspace>/inputs/<id>/``.

    Each analysis task has a unique data folder, so the agent sees:
      ./inputs/00000001/<spreadsheet>.xlsx
      ./inputs/00000001/introduction.txt
      ./inputs/00000001/questionN.txt   (the questions)
    """
    inputs_dir = workspace / INPUTS_SUBDIR
    inputs_dir.mkdir(parents=True, exist_ok=True)
    _replace_link(inputs_dir / task.task_id, task.task_dir.resolve())


def stage_modeling_inputs(workspace: Path, task: Task) -> None:
    """Symlink the resplit data dir into ``<workspace>/inputs/<comp>/``.

    Agent sees:
      ./inputs/<comp>/train.csv
      ./inputs/<comp>/test.csv
      ./inputs/<comp>/sampleSubmission.csv
    The held-out test labels (``answers/<comp>/test_answer.csv``) are NOT
    staged.
    """
    inputs_dir = workspace / INPUTS_SUBDIR
    inputs_dir.mkdir(parents=True, exist_ok=True)
    _replace_link(inputs_dir / task.task_id, task.task_dir.resolve())


# --------------------------------------------------------------------------- #
# Skip-existing helpers                                                       #
# --------------------------------------------------------------------------- #


def is_workspace_complete(workspace: Path) -> bool:
    """A workspace is "done" iff:
    - ``output.json`` exists, parses, and has ``ok == True``
    - both ``data_agent_history.json`` and ``experiment_agent_history.json``
      exist and are non-empty
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
    data_path: Path,
    data_desc: str,
    max_revisions: int = 1,
    data_recursion_limit: int = 80,
    experiment_recursion_limit: int = 128,
) -> FullWorkflow:
    """Run one DSBench task through SciDER FullWorkflow."""
    workspace_dir = Path(workspace_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running FullWorkflow in {} (data_path={})", workspace_dir, data_path)
    return run_full_workflow(
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
