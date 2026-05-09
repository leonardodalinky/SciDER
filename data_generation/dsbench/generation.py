"""DSBench data-generation workflow (analysis + modeling families).

For each task we hand the upstream prompt to SciDER's FullWorkflow
(data + experiment, no ideation, no paper writing). The two agent
histories become SFT trajectories. Upstream gold answers / test labels
are deliberately withheld from the prompt and the workspace.

uid format:
    analysis  → ``dsbench_analysis_<id>``    e.g. ``dsbench_analysis_00000001``
    modeling  → ``dsbench_modeling_<comp>``  e.g. ``dsbench_modeling_titanic``
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
    ANALYSIS_ANSWERS_FILENAME,
    DATA_HISTORY_FILENAME,
    EXP_HISTORY_FILENAME,
    INPUTS_SUBDIR,
    MODELING_SUBMISSION_FILENAME,
    PROMPT_FILENAME,
    Task,
    discover_analysis_tasks,
    discover_modeling_tasks,
    run_full_workflow_task,
    scan_completed_uids,
    stage_analysis_inputs,
    stage_modeling_inputs,
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
# Prompt building — analysis                                                  #
# --------------------------------------------------------------------------- #


def _build_analysis_query(task: Task) -> str:
    """Compose the experiment-agent prompt for one analysis task.

    The agent gets: (1) the introduction text, (2) every question file's
    text, (3) explicit pointers to the spreadsheet path and the question
    files, (4) a JSON-output protocol.
    """
    base = f"./{INPUTS_SUBDIR}/{task.task_id}"
    xlsx_path = f"{base}/{task.xlsx_filename}"

    # Inline each question's text so the agent doesn't have to read N files
    # before it can plan. Keeps the question file paths handy too in case
    # the inline copy gets truncated.
    q_blocks: list[str] = []
    for qid, qpath in zip(task.question_ids, task.question_files):
        try:
            qtext = qpath.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            qtext = "(could not read)"
        q_blocks.append(f"### {qid}\n\n_File: `{base}/{qpath.name}`_\n\n{qtext}")

    parts: list[str] = [
        f"# DSBench Analysis Task `{task.uid}`",
        "",
        f"Source competition: **{task.name or task.task_id}**"
        + (f", year {task.year}" if task.year else "")
        + ".",
        "",
        "You are given a finance/business spreadsheet and a set of questions "
        "to answer about it. Answer EVERY question precisely (a single MCQ "
        "letter, a number, or a short phrase as the question demands). The "
        "spreadsheet is the authoritative data source — read it; do not "
        "speculate.",
        "",
        "## Workspace layout",
        "",
        f"- Spreadsheet: `{xlsx_path}`",
        f"- Introduction: `{base}/introduction.txt`",
        f"- Questions: `{base}/question{{N}}.txt` (one file per Q; full text "
        "also inlined below)",
        "",
        "## Introduction",
        "",
        task.introduction.strip() or "(no introduction provided)",
        "",
        "## Questions",
        "",
        "\n\n".join(q_blocks),
        "",
        "## Output protocol",
        "",
        f"- Save your answers to `./{ANALYSIS_ANSWERS_FILENAME}` at workspace "
        "root, with schema:",
        "",
        "  ```json",
        '  {"questionN": <answer>, "questionM": <answer>, ...}',
        "  ```",
        "",
        f"  Use the EXACT same question keys as listed above " f"(e.g. `{task.question_ids[0]}`).",
        "- Each answer is whatever the question asks for: a single uppercase "
        'MCQ letter (e.g. `"A"`), a number (integer or float, no formatting), '
        "or a short string. Do NOT add explanations into the value field.",
        "- The file MUST parse as valid JSON. Use `json.dump(..., ensure_ascii=False)`.",
        "- Standard tools (pandas / numpy / openpyxl / xlrd) are preinstalled. "
        "Do not install new packages.",
        f"- Do NOT modify or write under `./{INPUTS_SUBDIR}/` — it is read-only.",
        "- Before finishing, re-read your output file and confirm every "
        "question key from the list above is present.",
    ]
    return "\n".join(parts)


def _build_analysis_data_summary(task: Task) -> str:
    base = f"./{INPUTS_SUBDIR}/{task.task_id}"
    return (
        f"DSBench analysis task `{task.uid}` "
        f"(competition: {task.name or task.task_id}). "
        f"Spreadsheet at `{base}/{task.xlsx_filename}`, "
        f"{len(task.question_ids)} questions to answer.\n\n"
        "## Scope of THIS phase (light EDA only)\n\n"
        "Do JUST enough so the experiment phase can answer the questions:\n"
        f"1. Open `{base}/{task.xlsx_filename}` and list every sheet name "
        "+ shape (rows × cols) + dtypes per column.\n"
        "2. Note any merged cells, header rows, or formula cells that may "
        "affect parsing (xlrd vs openpyxl behaviour).\n"
        f"3. Skim `{base}/introduction.txt` so you know which sheets are "
        "load-bearing for the questions.\n\n"
        "Do NOT in this phase:\n"
        "- Compute the question answers — that is the experiment phase.\n"
        f"- Write `./{ANALYSIS_ANSWERS_FILENAME}` — same.\n"
        "- Modify files under `./inputs/`.\n\n"
        "Standard tools (pandas / numpy / openpyxl / xlrd) are preinstalled."
    )


# --------------------------------------------------------------------------- #
# Prompt building — modeling                                                  #
# --------------------------------------------------------------------------- #


def _build_modeling_query(task: Task) -> str:
    base = f"./{INPUTS_SUBDIR}/{task.task_id}"
    parts: list[str] = [
        f"# DSBench Modeling Task `{task.uid}`",
        "",
        f"A Kaggle-style ML competition: **{task.name}**.",
        "",
        "## Goal",
        "",
        f"Train on `{base}/train.csv`, predict labels on `{base}/test.csv`, "
        f"and write a submission file at workspace root: "
        f"`./{MODELING_SUBMISSION_FILENAME}`. The submission must match the "
        f"format (column names + row order) shown in `{base}/sampleSubmission.csv`.",
        "",
        "## Competition description",
        "",
        task.description.strip() or "(no description available)",
        "",
        "## Workspace layout",
        "",
        f"- Train data: `{base}/train.csv`",
        f"- Test data: `{base}/test.csv`",
        f"- Sample submission: `{base}/sampleSubmission.csv`",
        f"- Your submission goes to: `./{MODELING_SUBMISSION_FILENAME}` (workspace root)",
        "",
        "## Output protocol",
        "",
        f"- Write predictions to `./{MODELING_SUBMISSION_FILENAME}` at "
        "workspace root (NOT under `./" + INPUTS_SUBDIR + "/`).",
        f"- The schema MUST match `{base}/sampleSubmission.csv` exactly: same "
        "header, same row count, same id column values in the same order.",
        "- Standard tools (pandas / numpy / scikit-learn / scipy / xgboost / "
        "lightgbm / matplotlib / torch / tensorflow / nltk / etc.) are "
        "preinstalled. Do NOT install packages with `pip install`.",
        f"- Do NOT modify or write under `./{INPUTS_SUBDIR}/` — it is " "read-only source data.",
        f"- Before finishing, run the program end-to-end and confirm "
        f"`./{MODELING_SUBMISSION_FILENAME}` exists, parses as CSV, and has "
        f"the same row count as `{base}/test.csv`.",
    ]
    return "\n".join(parts)


def _build_modeling_data_summary(task: Task) -> str:
    base = f"./{INPUTS_SUBDIR}/{task.task_id}"
    return (
        f"DSBench modeling task `{task.uid}` "
        f"(Kaggle competition: {task.name}).\n\n"
        f"## Files\n\n"
        f"- `{base}/train.csv` — training data with labels\n"
        f"- `{base}/test.csv` — test data, labels withheld\n"
        f"- `{base}/sampleSubmission.csv` — submission format reference\n\n"
        "## Scope of THIS phase (light EDA only)\n\n"
        "Do JUST enough to make the experiment phase productive:\n"
        "1. Confirm train.csv / test.csv / sampleSubmission.csv all load "
        "(right delimiter, encoding, header).\n"
        "2. Record shape, dtypes, target column (the column in train.csv "
        "but NOT in test.csv — usually also matches sampleSubmission.csv's "
        "non-id column).\n"
        "3. Spot-check class imbalance / target distribution / missing "
        "values / obviously useless columns (constants, all-NaN, IDs).\n"
        "4. Note id-column name + dtype so the experiment agent gets the "
        "submission's row order right.\n\n"
        "Do NOT in this phase:\n"
        "- Train any model — that's the experiment phase.\n"
        f"- Write `./{MODELING_SUBMISSION_FILENAME}` — same.\n"
        "- Modify files under `./inputs/`.\n\n"
        "Standard tabular + ML libraries are preinstalled."
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
    workspace = (output_root / task.uid).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "uid": task.uid,
        "family": task.family,
        "task_id": task.task_id,
        "name": task.name,
        "workspace": str(workspace),
    }

    try:
        if task.family == "analysis":
            stage_analysis_inputs(workspace, task)
            user_query = _build_analysis_query(task)
            data_desc = _build_analysis_data_summary(task)
            record["n_questions"] = len(task.question_ids)
        else:  # modeling
            stage_modeling_inputs(workspace, task)
            user_query = _build_modeling_query(task)
            data_desc = _build_modeling_data_summary(task)

        (workspace / PROMPT_FILENAME).write_text(user_query, encoding="utf-8")
        data_path = workspace / INPUTS_SUBDIR

        logger.info(
            "[{}] running FullWorkflow ({} chars prompt, family={})",
            task.uid,
            len(user_query),
            task.family,
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
            "Generate SFT trajectories from DSBench (data_analysis + "
            "data_modeling) via SciDER's FullWorkflow (data + experiment, "
            "no ideation). Each task produces a workspace under "
            "<output_root>/<uid>/ containing prompt.md, inputs/<id-or-comp>/ "
            "(symlinked), data_agent_history.json, experiment_agent_history.json, "
            "and output.json."
        ),
        prog="python -m data_generation.dsbench.generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bench-root",
        required=True,
        help="Path to the unzipped dsbench-data dir (holds data_analysis/ " "and data_modeling/).",
    )
    parser.add_argument(
        "--output-root",
        "-o",
        required=True,
        help="Directory under which per-task workspaces are created.",
    )
    parser.add_argument(
        "--family",
        choices=["analysis", "modeling", "both"],
        default="both",
        help="Which task family to run (default: both).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip uids whose output.json marks the run complete.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most the first N tasks (after --skip-existing / " "--family filter).",
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

    tasks: list[Task] = []
    if args.family in ("analysis", "both"):
        tasks.extend(discover_analysis_tasks(bench_root))
    if args.family in ("modeling", "both"):
        tasks.extend(discover_modeling_tasks(bench_root))
    logger.info("Discovered {} tasks (family={})", len(tasks), args.family)

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
