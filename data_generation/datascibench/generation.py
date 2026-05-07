"""DataSciBench data-generation workflow.

Source: https://arxiv.org/abs/2502.13897
HF dataset: https://huggingface.co/datasets/zd21/DataSciBench

For each ``<family>_<id>/`` subfolder under ``--bench-root`` we hand the
upstream prompt (with input-path note prepended) to SciDER's FullWorkflow
(data + experiment, no ideation, no paper writing). The two agents'
histories are persisted at:

    <workspace>/data_agent_history.json
    <workspace>/experiment_agent_history.json

These become SFT data for OpenSciDER (consumed by ``train/prepare_data.py``).

We don't run the upstream LLM-as-judge evaluator here — the goal is data,
not metric. Every trajectory is kept; ``output.json.ok`` only requires the
workflow to have produced both histories (non-empty).

uid format: ``datascibench_<family>_<id>``  e.g. ``datascibench_human_3``.
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
    discover_tasks,
    run_full_workflow_task,
    scan_completed_uids,
    stage_inputs,
    write_output_json,
)

# ---- Constants ----
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


def _build_user_query(task: Task, staged_paths: list[str]) -> str:
    """Wrap the upstream task prompt with two pieces of context the agent
    needs but the upstream prompt didn't anticipate:

    1. The **input-path remap**: upstream prompts say things like
       "stored in 'data.csv'" assuming the file is at workspace root. We
       moved every input under ``./inputs/`` to keep workspaces tidy and
       to mirror the discoverybench convention, so the agent must remap
       any bare filename it sees into ``./inputs/<filename>``.

    2. The **output protocol**: every task names its own output file
       (``output.csv``, ``most_corr_output.csv``, etc.) inline — we tell
       the agent to save those to workspace root (NOT under ``inputs/``).
    """
    if staged_paths:
        bullets = "\n".join(f"- `./{INPUTS_SUBDIR}/{p}`" for p in staged_paths)
        input_note = (
            f"## Input files\n\n"
            f"All input files are staged under `./{INPUTS_SUBDIR}/` (read-only):\n\n"
            f"{bullets}\n\n"
            f"The original task description (below) was authored assuming inputs "
            f"sit at workspace root. **Whenever the description references a bare "
            f"filename like `data.csv` or `data_1.xlsx`, read from "
            f"`./{INPUTS_SUBDIR}/<that filename>` instead.**"
        )
    else:
        # csv_excel family: prompt embeds the dataset inline; no input files.
        input_note = (
            f"## Input files\n\n"
            f"No external input files for this task — the dataset is embedded "
            f"verbatim inside the task description below. If the description "
            f"asks you to write the data to a file (e.g. `data.csv`), create "
            f"that file in workspace root from the inline data."
        )

    parts = [
        f"# DataSciBench Task `{task.uid}`",
        "",
        f"Family: **{task.family}**, task id: **{task.task_id}**, "
        f"data_source_type: `{task.data_source_type or 'unspecified'}`.",
        "",
        input_note,
        "",
        "## Original task",
        "",
        task.prompt,
        "",
        "## Output protocol",
        "",
        "- Implement the requested logic, run it, and confirm the output "
        "file the task asks for is created at **workspace root** "
        f"(NOT under `./{INPUTS_SUBDIR}/`).",
        "- The exact output filename is whatever the task description "
        "specifies above (typically called out as `output.csv`, "
        "`most_corr_output.csv`, etc. — match it verbatim).",
        "- Standard tabular tools (pandas / numpy / openpyxl / scikit-learn / "
        "statsmodels) are preinstalled. Do not install new packages with "
        "`pip install`.",
        "- Do NOT touch files under `./" + INPUTS_SUBDIR + "/` — they are "
        "the read-only source data.",
        "- Before finishing, verify the output file exists at workspace root "
        "and parses correctly (e.g. `pd.read_csv` round-trip).",
    ]
    return "\n".join(parts)


def _build_data_summary(task: Task, staged_paths: list[str]) -> str:
    """Short context the DataAgent sees under 'Data summary'. Mirrors the
    discoverybench philosophy: light EDA only. No analysis, no hypothesis
    forming — that's the experiment agent's job.
    """
    if staged_paths:
        bullets = "\n".join(f"- `./{INPUTS_SUBDIR}/{p}`" for p in staged_paths)
        sources = f"Input files staged under `./{INPUTS_SUBDIR}/`:\n\n{bullets}\n\n"
    else:
        sources = (
            "No external input files — the dataset is inline in the task "
            "prompt the experiment agent will receive.\n\n"
        )
    return (
        f"DataSciBench task `{task.uid}` (family={task.family}, "
        f"data_source_type={task.data_source_type or 'unspecified'}). "
        f"{sources}"
        "## Scope of THIS phase (light EDA only)\n\n"
        "Do JUST enough to make the experiment phase productive:\n"
        "1. Confirm each input file loads (right delimiter / encoding / "
        "sheet name for xlsx).\n"
        "2. Record shape, dtypes, and any column-name aliases or unit "
        "oddities you notice.\n"
        "3. Spot-check missing values / obvious outliers.\n\n"
        "Do NOT in this phase:\n"
        "- Implement the task logic the experiment agent will be asked to "
        "  run (correlations, model training, dataframe reshaping, etc.).\n"
        "- Write the task's deliverable file — that is produced ONLY by "
        "  the experiment phase.\n"
        "- Generate plots / figures.\n\n"
        "Standard tabular tools (pandas / numpy / openpyxl / scikit-learn / "
        "statsmodels) are preinstalled."
    )


# --------------------------------------------------------------------------- #
# Per-task runner                                                             #
# --------------------------------------------------------------------------- #


def run_one_task(
    *,
    task: Task,
    output_root: Path,
    max_revisions: int,
    data_recursion_limit: int,
    experiment_recursion_limit: int,
) -> dict:
    """Run one DataSciBench task. Returns a record dict suitable for
    appending to ``results.json``. Always returns — exceptions during the
    workflow are caught and surfaced via ``ok=false``."""
    workspace = (output_root / task.uid).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "uid": task.uid,
        "family": task.family,
        "task_id": task.task_id,
        "data_source_type": task.data_source_type,
        "workspace": str(workspace),
    }

    try:
        staged = stage_inputs(workspace, task.task_dir)
        user_query = _build_user_query(task, staged)
        data_desc = _build_data_summary(task, staged)
        (workspace / PROMPT_FILENAME).write_text(user_query, encoding="utf-8")

        # Always pass inputs/ as data_path so FullWorkflow's validator is
        # satisfied (it requires either data_path or feature_desc). For
        # csv_excel tasks the dir is empty — DataAgent will note that in
        # its summary; the experiment agent then materialises the inline
        # CSV from the prompt itself. We do NOT route empty-input tasks
        # through feature_desc because that triggers HypoDataWorkflow
        # (synthetic data generation), which is the wrong contract — the
        # data already exists, just inlined.
        data_path: Path = workspace / INPUTS_SUBDIR

        logger.info(
            "[{}] running FullWorkflow ({} input files, {} chars prompt)",
            task.uid,
            len(staged),
            len(user_query),
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
            n_input_files=len(staged),
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
        logger.info(
            "[{}] done — ok={} status={}",
            task.uid,
            ok,
            workflow.final_status,
        )
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
            "Generate SFT trajectories from DataSciBench via SciDER's "
            "FullWorkflow (data + experiment, no ideation). Each upstream "
            "subfolder produces a workspace under <output_root>/<uid>/ "
            "containing prompt.md, inputs/, data_agent_history.json, "
            "experiment_agent_history.json, and output.json."
        ),
        prog="python -m data_generation.datascibench.generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bench-root",
        required=True,
        help="Path to the directory containing csv_excel_*/, dl_*/, human_*/ "
        "subfolders (e.g. .../DataSciBench/data/DataSciBench-data).",
    )
    parser.add_argument(
        "--output-root",
        "-o",
        required=True,
        help="Directory under which per-task workspaces are created " "(<output_root>/<uid>/).",
    )
    parser.add_argument(
        "--family",
        choices=["csv_excel", "dl", "human"],
        default=None,
        help="Filter to a single task family. Omit to run all three.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip uids whose <output_root>/<uid>/output.json marks the run "
        "complete (ok=true + non-empty data + experiment histories).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most the first N tasks (after --skip-existing / "
        "--family filter). Omit to run all.",
    )
    parser.add_argument(
        "--uids",
        default=None,
        help="Comma-separated list of uids to run; overrides --limit. "
        "Useful for targeted debug runs.",
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
        default=100,
        help="LangGraph recursion limit for the experiment agent (default 100).",
    )
    args = parser.parse_args()

    # 1. Register models first so config errors surface before any file I/O.
    _register_models_from_yaml()

    bench_root = Path(args.bench_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 2. Discover tasks.
    logger.info("Discovering tasks under {}", bench_root)
    tasks = list(discover_tasks(bench_root))
    if args.family:
        tasks = [t for t in tasks if t.family == args.family]
    logger.info("Discovered {} tasks (family filter={})", len(tasks), args.family)

    # 3. --skip-existing.
    if args.skip_existing:
        completed = scan_completed_uids(output_root)
        if completed:
            logger.info(
                "--skip-existing: {} uids already complete; skipping them",
                len(completed),
            )
            tasks = [t for t in tasks if t.uid not in completed]

    # 4. --uids takes precedence over --limit.
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

    # 5. Resume results.json across runs (dedup by uid).
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
