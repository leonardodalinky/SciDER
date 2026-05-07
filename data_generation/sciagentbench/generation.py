"""ScienceAgentBench data-generation workflow.

For each row of the verified split (102 tasks) we hand the upstream task
instruction (+ optional domain knowledge + dataset preview) to SciDER's
FullWorkflow (data + experiment, no ideation, no paper writing). The
agent's deliverable is a Python program that produces a task-specific
output file (each task names its own ``output_fname``).

We don't run the upstream evaluator here — the goal is data, not metric.
Every trajectory is kept; ``output.json.ok`` only requires both agent
histories to exist and be non-empty.

uid format: ``sciagentbench_<instance_id>``  e.g. ``sciagentbench_1``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

# Make project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scider.default.models import register_defaults_from_yaml

from ._common import (
    DATA_HISTORY_FILENAME,
    EXP_HISTORY_FILENAME,
    INPUTS_SUBDIR,
    PROMPT_FILENAME,
    Task,
    load_tasks,
    run_full_workflow_task,
    scan_completed_uids,
    stage_inputs,
    write_output_json,
)

ROLES_YAML_PATH = Path(__file__).parent / "roles.yaml"


def _register_models_from_yaml() -> None:
    if not ROLES_YAML_PATH.exists():
        raise FileNotFoundError(f"Role yaml missing at {ROLES_YAML_PATH}")
    logger.info("Registering roles from {}", ROLES_YAML_PATH)
    registered = register_defaults_from_yaml(ROLES_YAML_PATH)
    logger.info("Registered {} roles: {}", len(registered), sorted(registered))


# --------------------------------------------------------------------------- #
# Prompt building                                                             #
# --------------------------------------------------------------------------- #


def _build_user_query(task: Task, *, use_knowledge: bool) -> str:
    """Compose the experiment-agent prompt.

    Mirrors upstream's ScienceAgent.get_sys_msg structure (task_inst +
    optional domain_knowledge + dataset folder tree + preview) but adds the
    SciDER-specific path remap: dataset is staged under ``./inputs/<folder>/``
    rather than the upstream's central ``benchmark/datasets/`` location.

    The deliverable contract is preserved verbatim: write the program output
    to whatever ``output_fname`` upstream specifies (typically
    ``pred_results/<task>_pred.csv`` at workspace root).
    """
    parts: list[str] = [
        f"# ScienceAgentBench Task `{task.uid}`",
        "",
        f"Domain: **{task.domain}**, subtask categories: {task.subtask_categories}, "
        f"upstream src: `{task.github_name}` (`{task.src_file_or_path}`).",
        "",
        "## Task instruction",
        "",
        task.task_inst.strip(),
        "",
    ]

    if use_knowledge and task.domain_knowledge.strip():
        parts += [
            "## Domain knowledge (provided by the benchmark)",
            "",
            task.domain_knowledge.strip(),
            "",
        ]

    parts += [
        "## Dataset",
        "",
        f"The full dataset folder has been staged under "
        f"`./{INPUTS_SUBDIR}/{task.dataset_folder}/` (read-only). The upstream "
        f"folder tree was authored relative to a central `datasets/` dir; in "
        f"this workspace the same files live under "
        f"`./{INPUTS_SUBDIR}/{task.dataset_folder}/` instead — adjust any "
        f"``open(...)`` / ``pd.read_csv(...)`` / ``Path(...)`` calls "
        f"accordingly.",
        "",
        "Folder tree (upstream-style; replace the leading folder name with "
        f"`./{INPUTS_SUBDIR}/{task.dataset_folder}` when reading):",
        "",
        "```",
        task.dataset_folder_tree.strip(),
        "```",
        "",
        "Preview of dataset file(s):",
        "",
        task.dataset_preview.strip(),
        "",
        "## Output protocol",
        "",
        f"- Write the program output to **`./{task.output_fname}`** at "
        "workspace root (NOT under `./" + INPUTS_SUBDIR + "/`). Create any "
        "intermediate directories the path implies (e.g. `pred_results/`).",
        f"- Use the exact output filename `{task.output_fname}` — the "
        "upstream evaluator (we don't run it here, but the trajectory "
        "should still be faithful) checks that exact path.",
        "- Do NOT install packages with `pip install` / `!pip install`. "
        "Standard scientific tools (numpy / pandas / scikit-learn / "
        "scipy / matplotlib / torch / tensorflow / etc.) are preinstalled.",
        f"- Do NOT modify or write under `./{INPUTS_SUBDIR}/` — it is " "read-only source data.",
        "- Before finishing, run the program end-to-end and confirm the "
        "output file exists at the expected path.",
    ]
    return "\n".join(parts)


def _build_data_summary(task: Task) -> str:
    """Light-EDA brief for the data agent. Same philosophy as datascibench /
    discoverybench: no analysis, no modelling — that's the experiment agent's
    job."""
    return (
        f"ScienceAgentBench task `{task.uid}` (domain: {task.domain}). "
        f"Dataset staged under `./{INPUTS_SUBDIR}/{task.dataset_folder}/`.\n\n"
        f"## Folder tree\n\n"
        f"```\n{task.dataset_folder_tree.strip()}\n```\n\n"
        f"## Scope of THIS phase (light EDA only)\n\n"
        "Do JUST enough to make the experiment phase productive:\n"
        "1. Confirm each referenced input file loads (right delimiter / "
        "encoding / sheet name for xlsx / format for npy / nc / etc.).\n"
        "2. Record shape, dtypes, column-name aliases, missing values.\n"
        "3. Note any unit oddities or class imbalance the experiment agent "
        "should know about.\n\n"
        "Do NOT in this phase:\n"
        "- Implement the task logic (model training, predictions, plots).\n"
        f"- Write the deliverable (`./{task.output_fname}`) — that is the "
        "experiment phase's responsibility.\n"
        f"- Modify files under `./{INPUTS_SUBDIR}/`.\n\n"
        "Standard scientific tools (numpy / pandas / scikit-learn / scipy / "
        "matplotlib / torch / tensorflow / etc.) are preinstalled."
    )


# --------------------------------------------------------------------------- #
# Per-task runner                                                             #
# --------------------------------------------------------------------------- #


def run_one_task(
    *,
    task: Task,
    output_root: Path,
    use_knowledge: bool,
    max_revisions: int,
    data_recursion_limit: int,
    experiment_recursion_limit: int,
) -> dict:
    """Run one ScienceAgentBench task. Returns a record dict suitable for
    appending to ``results.json``. Always returns — exceptions during the
    workflow are caught and surfaced via ``ok=false``."""
    workspace = (output_root / task.uid).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "uid": task.uid,
        "instance_id": task.instance_id,
        "domain": task.domain,
        "subtask_categories": task.subtask_categories,
        "github_name": task.github_name,
        "gold_program_name": task.gold_program_name,
        "output_fname": task.output_fname,
        "workspace": str(workspace),
    }

    try:
        if not stage_inputs(workspace, task):
            raise RuntimeError(
                f"could not stage dataset folder '{task.dataset_folder}' "
                f"(missing at {task.dataset_path})"
            )
        user_query = _build_user_query(task, use_knowledge=use_knowledge)
        data_desc = _build_data_summary(task)
        (workspace / PROMPT_FILENAME).write_text(user_query, encoding="utf-8")

        data_path = workspace / INPUTS_SUBDIR

        logger.info(
            "[{}] running FullWorkflow ({} chars prompt, use_knowledge={})",
            task.uid,
            len(user_query),
            use_knowledge,
        )
        workflow = run_full_workflow_task(
            user_query=user_query,
            workspace_dir=workspace,
            data_path=data_path,
            data_desc=data_desc,
            max_revisions=max_revisions,
            data_recursion_limit=data_recursion_limit,
            experiment_recursion_limit=experiment_recursion_limit,
        )

        data_hist = workspace / DATA_HISTORY_FILENAME
        exp_hist = workspace / EXP_HISTORY_FILENAME
        ok = (
            workflow.final_status == "success"
            and data_hist.is_file()
            and data_hist.stat().st_size > 0
            and exp_hist.is_file()
            and exp_hist.stat().st_size > 0
        )
        record.update(
            ok=ok,
            status=workflow.final_status,
            data_history_path=str(data_hist.relative_to(output_root)),
            experiment_history_path=str(exp_hist.relative_to(output_root)),
        )
        if not ok:
            record["error"] = (
                workflow.error_message
                or f"final_status={workflow.final_status}, "
                f"data_hist_exists={data_hist.is_file()}, "
                f"exp_hist_exists={exp_hist.is_file()}"
            )
        logger.info("[{}] done — ok={} status={}", task.uid, ok, workflow.final_status)
    except Exception as e:
        logger.exception("[{}] failed: {}", task.uid, e)
        record.update(ok=False, error=str(e))
    write_output_json(workspace, record)
    return record


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SFT trajectories from ScienceAgentBench (verified split) "
            "via SciDER's FullWorkflow (data + experiment, no ideation). Each "
            "task produces a workspace under <output_root>/<uid>/ containing "
            "prompt.md, inputs/<folder>/ (symlinked), data_agent_history.json, "
            "experiment_agent_history.json, and output.json."
        ),
        prog="python -m data_generation.sciagentbench.generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bench-root",
        required=True,
        help="Path to the unzipped benchmark dir holding datasets/, "
        "eval_programs/, gold_programs/, scoring_rubrics/. "
        "(e.g. /sciclone/proj-ds/ai4scientist/kelin/SciDER/sciagentbench/benchmark)",
    )
    parser.add_argument(
        "--output-root",
        "-o",
        required=True,
        help="Directory under which per-task workspaces are created.",
    )
    parser.add_argument(
        "--split",
        default="verified",
        help="HF dataset split (default: verified — the only one upstream).",
    )
    parser.add_argument(
        "--use-knowledge",
        action="store_true",
        help="Inject the upstream `domain_knowledge` field into the prompt. "
        "Off by default to match the canonical 'no-hint' setting.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip uids whose output.json marks the run complete (ok=true + "
        "non-empty data + experiment histories).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most the first N tasks (after --skip-existing).",
    )
    parser.add_argument(
        "--uids",
        default=None,
        help="Comma-separated list of uids to run; overrides --limit.",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=1,
        help="Critic/approval retry budget per agent (default 1).",
    )
    parser.add_argument(
        "--data-recursion-limit",
        type=int,
        default=80,
        help="LangGraph recursion limit for the data agent (default 80).",
    )
    parser.add_argument(
        "--experiment-recursion-limit",
        type=int,
        default=128,
        help="LangGraph recursion limit for the experiment agent (default 128).",
    )
    args = parser.parse_args()

    _register_models_from_yaml()

    bench_root = Path(args.bench_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Loading {} from {} (split={})", bench_root, args.bench_root, args.split)
    tasks = load_tasks(bench_root, split=args.split)
    logger.info("Discovered {} tasks", len(tasks))

    if args.skip_existing:
        completed = scan_completed_uids(output_root)
        if completed:
            logger.info(
                "--skip-existing: {} uids already complete; skipping them",
                len(completed),
            )
            tasks = [t for t in tasks if t.uid not in completed]

    if args.uids:
        wanted = {u.strip() for u in args.uids.split(",") if u.strip()}
        selected = [t for t in tasks if t.uid in wanted]
        missing = wanted - {t.uid for t in selected}
        if missing:
            logger.warning("Requested uids not in candidate set: {}", sorted(missing))
    elif args.limit is not None:
        selected = tasks[: args.limit]
    else:
        selected = tasks
    logger.info("Processing {} tasks", len(selected))

    results_path = output_root / "results.json"
    results: list[dict] = []
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
            logger.info("Resuming from {} ({} prior records)", results_path, len(results))
        except json.JSONDecodeError:
            logger.warning("Existing results.json malformed; starting fresh")

    for t in selected:
        record = run_one_task(
            task=t,
            output_root=output_root,
            use_knowledge=args.use_knowledge,
            max_revisions=args.max_revisions,
            data_recursion_limit=args.data_recursion_limit,
            experiment_recursion_limit=args.experiment_recursion_limit,
        )
        results = [r for r in results if r.get("uid") != record["uid"]]
        results.append(record)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.get("ok"))
    logger.info(
        "Done. {}/{} produced both agent histories. Trajectories under {}",
        n_ok,
        len(results),
        output_root,
    )


if __name__ == "__main__":
    _main()
