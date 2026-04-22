"""MLE-Bench Workflow — run SciDER's FullWorkflow on an MLE-Bench competition.

MLE-Bench hands the agent three files per task:

* ``instructions.md`` — specific task instructions (used as ``user_query``).
* ``description.md`` — overall task background (used as ``data_desc``).
* ``/home/data/`` — the competition's data directory.

This wrapper registers model roles from
``bench_workflows/model_configs/mlebench_roles.yaml``, reads those files, and
hands everything to ``run_full_workflow``. The deliverable the MLE-Bench
harness looks for is ``<workspace>/submission.csv``.

Usage
-----
    python -m bench_workflows.mlebench_workflow \\
        --instructions competition/instructions.md \\
        --description  competition/description.md \\
        --data         competition/data \\
        --workspace    workspace
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Make scider + bench modules importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scider.default.models import register_defaults_from_yaml
from scider.workflows.full_workflow import run_full_workflow

# ---- Constants ----
# The per-project model-role assignment for this benchmark. Edit the yaml to
# swap models — don't reintroduce a --models CLI knob; this matches the
# airsbench / astrovisbench refactor.
ROLES_YAML_PATH = Path(__file__).parent / "model_configs" / "mlebench_roles.yaml"


def _register_models_from_yaml() -> None:
    """Load role assignments from the mlebench roles yaml."""
    if not ROLES_YAML_PATH.exists():
        raise FileNotFoundError(
            f"Role yaml missing at {ROLES_YAML_PATH}. This file pins the "
            "experiment / critic / approval / ... roles for MLE-Bench; "
            "don't rename it."
        )
    logger.info("Registering roles from {}", ROLES_YAML_PATH)
    registered = register_defaults_from_yaml(ROLES_YAML_PATH)
    logger.info("Registered {} roles: {}", len(registered), sorted(registered))


def build_mlebench_user_query(
    instructions_path: Path,
    description_path: Path,
) -> tuple[str, str]:
    """Read instructions.md + description.md. Returns ``(user_query, data_desc)``."""
    if not instructions_path.exists():
        raise FileNotFoundError(f"Instructions file not found: {instructions_path}")
    if not description_path.exists():
        raise FileNotFoundError(f"Description file not found: {description_path}")
    return (
        instructions_path.read_text(encoding="utf-8"),
        description_path.read_text(encoding="utf-8"),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MLE-Bench workflow — runs SciDER's FullWorkflow (data + experiment) "
            "on one MLE-Bench competition. The deliverable is "
            "<workspace>/submission.csv, which MLE-Bench's grader picks up."
        ),
        prog="python -m bench_workflows.mlebench_workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instructions",
        "-i",
        required=True,
        help="Path to instructions.md (task-specific instructions, used as user_query).",
    )
    parser.add_argument(
        "--description",
        "-d",
        required=True,
        help="Path to description.md (task background, used as data_desc).",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the competition data directory (mounted at /home/data in the container).",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        required=True,
        help="Workspace directory for the experiment (submission.csv lands at <workspace>/).",
    )
    parser.add_argument(
        "--repo-source",
        default=None,
        help="Optional repository source (local path or git URL) for the experiment agent.",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=3,
        help="Maximum critic/approval revision loops (default 3).",
    )
    parser.add_argument(
        "--data-recursion-limit",
        type=int,
        default=512,
        help="Recursion limit for DataAgent (default 512).",
    )
    parser.add_argument(
        "--experiment-recursion-limit",
        type=int,
        default=512,
        help="Recursion limit for ExperimentAgent (default 512).",
    )
    args = parser.parse_args()

    # 1. Register models first so any config error surfaces before we start work.
    _register_models_from_yaml()

    # 2. Build user query + data description from the MLE-Bench files.
    logger.info("Building user query from MLE-Bench task files...")
    user_query, data_desc = build_mlebench_user_query(
        instructions_path=Path(args.instructions),
        description_path=Path(args.description),
    )
    logger.info("user_query: {} chars; data_desc: {} chars", len(user_query), len(data_desc))

    # 3. Run FullWorkflow.
    result = run_full_workflow(
        data_path=args.data,
        workspace_path=args.workspace,
        user_query=user_query,
        data_desc=data_desc,
        repo_source=args.repo_source,
        max_revisions=args.max_revisions,
        data_agent_recursion_limit=args.data_recursion_limit,
        experiment_agent_recursion_limit=args.experiment_recursion_limit,
    )

    # FullWorkflow persists sub-workflow summaries as it goes; also dump the
    # top-level summary so the grader / debug scripts have it co-located.
    result.save_summary()
    logger.info("Final status: {}", result.final_status)
    print(f"\nStatus: {result.final_status}")


if __name__ == "__main__":
    _main()
