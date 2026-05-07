"""ScienceAgentBench internal plumbing — used only by ``generation``.

* ``Task`` / ``load_tasks`` — fetch the 102 verified rows from HF and resolve
  each row's ``dataset_folder_tree`` first line to a concrete folder under
  the local ``<bench_root>/datasets/`` (matches upstream ``run_infer.py``
  behaviour: ``args.datasets_path + dataset_folder_tree.split("\\n")[0][4:]``).
* ``stage_inputs`` — symlink the resolved dataset folder into
  ``<workspace>/inputs/<folder>`` (one symlink per task, not a recursive
  copy — datasets here can be hundreds of MB and many tasks share folders).
* ``run_full_workflow_task`` — invoke ``run_full_workflow`` on a single task
  (data + experiment, no ideation, no paper writing). Sub-workflows save
  ``data_agent_history.json`` / ``experiment_agent_history.json`` themselves
  (crash-safe via ``capture_messages``).
* ``is_workspace_complete`` / ``scan_completed_uids`` / ``write_output_json``
  — skip-existing helpers mirroring datascibench.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from loguru import logger

from scider.workflows.full_workflow import FullWorkflow, run_full_workflow

OUTPUT_FILENAME = "output.json"
DATA_HISTORY_FILENAME = "data_agent_history.json"
EXP_HISTORY_FILENAME = "experiment_agent_history.json"
PROMPT_FILENAME = "prompt.md"
INPUTS_SUBDIR = "inputs"

HF_DATASET_ID = "osunlp/ScienceAgentBench"
DEFAULT_SPLIT = "verified"


# --------------------------------------------------------------------------- #
# Task discovery                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class Task:
    """One ScienceAgentBench task — one row of the verified split."""

    uid: str
    instance_id: int
    domain: str
    subtask_categories: str
    github_name: str
    task_inst: str
    domain_knowledge: str
    dataset_folder_tree: str
    dataset_preview: str
    src_file_or_path: str
    gold_program_name: str
    output_fname: str  # e.g. "pred_results/clintox_test_pred.csv"
    eval_script_name: str
    dataset_folder: str  # resolved from dataset_folder_tree, e.g. "clintox"
    dataset_path: Path  # absolute path to <bench_root>/datasets/<folder>


def _build_uid(instance_id: int | str) -> str:
    return f"sciagentbench_{instance_id}"


def _resolve_dataset_folder(dataset_folder_tree: str) -> str:
    """Mirror upstream's resolver: first line of the tree is ``|-- <folder>/``,
    so strip the leading 4 chars (``|-- ``) and any trailing slash to get
    the bare folder name.
    """
    first = (dataset_folder_tree or "").strip().splitlines()
    if not first:
        return ""
    folder = first[0][4:].strip().rstrip("/")
    return folder


def load_tasks(bench_root: Path, split: str = DEFAULT_SPLIT) -> list[Task]:
    """Load tasks from HF dataset, resolve each row's dataset path against
    ``<bench_root>/datasets/`` (the layout produced by unzipping
    ``benchmark_verified.zip``).

    Tasks whose dataset folder is missing on disk are still yielded — the
    runner will mark them ``ok=false`` so the failure is visible in
    ``results.json`` rather than silently skipped.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "ScienceAgentBench loading requires `datasets` (HuggingFace). Install with "
            "`uv add datasets` or `pip install datasets`."
        ) from e

    if not bench_root.is_dir():
        raise FileNotFoundError(f"ScienceAgentBench root not a directory: {bench_root}")
    datasets_root = bench_root / "datasets"
    if not datasets_root.is_dir():
        raise FileNotFoundError(
            f"Expected <bench_root>/datasets/ at {datasets_root}. "
            f"Did you unzip benchmark_verified.zip into {bench_root}?"
        )

    ds = load_dataset(HF_DATASET_ID, split=split)
    tasks: list[Task] = []
    for row in ds:
        instance_id = row.get("instance_id")
        folder = _resolve_dataset_folder(row.get("dataset_folder_tree") or "")
        tasks.append(
            Task(
                uid=_build_uid(instance_id),
                instance_id=int(instance_id) if instance_id is not None else len(tasks),
                domain=row.get("domain") or "",
                subtask_categories=row.get("subtask_categories") or "",
                github_name=row.get("github_name") or "",
                task_inst=row.get("task_inst") or "",
                domain_knowledge=row.get("domain_knowledge") or "",
                dataset_folder_tree=row.get("dataset_folder_tree") or "",
                dataset_preview=row.get("dataset_preview") or "",
                src_file_or_path=row.get("src_file_or_path") or "",
                gold_program_name=row.get("gold_program_name") or "",
                output_fname=row.get("output_fname") or "",
                eval_script_name=row.get("eval_script_name") or "",
                dataset_folder=folder,
                dataset_path=(datasets_root / folder).resolve() if folder else datasets_root,
            )
        )
    return tasks


# --------------------------------------------------------------------------- #
# Workspace staging                                                           #
# --------------------------------------------------------------------------- #


def stage_inputs(workspace: Path, task: Task) -> bool:
    """Symlink ``<bench_root>/datasets/<folder>`` into
    ``<workspace>/inputs/<folder>``. Returns True on success, False if the
    source folder is missing (caller should mark the task as failed).

    We symlink (not copy) because: (a) ScienceAgentBench shares dataset
    folders across tasks, (b) some folders are hundreds of MB, (c) the
    bench root and the workspace both live on the same NFS volume on wm,
    so symlinks resolve correctly inside the agent's workspace.
    """
    inputs_dir = workspace / INPUTS_SUBDIR
    inputs_dir.mkdir(parents=True, exist_ok=True)

    if not task.dataset_folder:
        logger.warning("[{}] no dataset folder resolvable from dataset_folder_tree", task.uid)
        return False
    if not task.dataset_path.is_dir():
        logger.warning("[{}] dataset folder missing on disk: {}", task.uid, task.dataset_path)
        return False

    link = inputs_dir / task.dataset_folder
    if link.is_symlink() or link.exists():
        link.unlink() if link.is_symlink() else None
        if link.exists():  # plain dir from a previous run — clear it
            import shutil

            shutil.rmtree(link)
    link.symlink_to(task.dataset_path)
    return True


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
    """Run one ScienceAgentBench task through SciDER FullWorkflow."""
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
