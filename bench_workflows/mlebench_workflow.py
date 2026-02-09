"""
MLE-Bench Workflow

Simple wrapper for running SciEvo FullWorkflow on MLE-Bench competition tasks.

MLE-Bench provides:
- instructions.md: Specific task instructions (used as user_query)
- description.md: Overall task background description

This wrapper register models, reads these files, builds user_query, and invokes FullWorkflow.
"""

import sys
from pathlib import Path

from loguru import logger

# Add parent directory to path to find scievo and bench modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench_workflows.register_models.gemini import (
    register_gemini_low_medium_models,
    register_gemini_medium_high_models,
)
from bench_workflows.register_models.gpt import (
    register_gpt_low_medium_models,
    register_gpt_medium_high_models,
)
from scievo.workflows.full_workflow import run_full_workflow


def build_mlebench_user_query(
    instructions_path: Path,
    description_path: Path,
) -> tuple[str, str]:
    """
    Build user query and data description from MLE-Bench task files.

    Args:
        instructions_path: Path to instructions.md
        description_path: Path to description.md

    Returns:
        Tuple of (user_query, data_desc)
        - user_query: Task instructions for the experiment
        - data_desc: Task description for data analysis context
    """
    # Load instructions
    if not instructions_path.exists():
        raise FileNotFoundError(f"Instructions file not found: {instructions_path}")
    instructions = instructions_path.read_text(encoding="utf-8")

    # Load description
    if not description_path.exists():
        raise FileNotFoundError(f"Description file not found: {description_path}")
    description = description_path.read_text(encoding="utf-8")

    # Use instructions as user_query, description as data_desc
    user_query = instructions
    data_desc = description

    return user_query, data_desc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MLE-Bench Workflow - Run SciEvo on MLE-Bench competition tasks",
        prog="python -m bench.mlebench_workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m bench.mlebench_workflow \\
      -i competition/instructions.md \\
      -d competition/description.md \\
      --data competition/data \\
      -w workspace

  # With custom settings
  python -m bench.mlebench_workflow \\
      -i competition/instructions.md \\
      -d competition/description.md \\
      --data competition/data \\
      -w workspace \\
      --max-revisions 10 \\
      --session-name my_experiment
        """,
    )

    # Required arguments
    parser.add_argument(
        "--instructions",
        "-i",
        required=True,
        help="Path to instructions.md (task instructions)",
    )
    parser.add_argument(
        "--description",
        "-d",
        required=True,
        help="Path to description.md (task background)",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the data directory or file",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        required=True,
        help="Workspace directory for the experiment",
    )

    # Optional arguments
    parser.add_argument(
        "--repo-source",
        default=None,
        help="Optional repository source (local path or git URL)",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=3,
        help="Maximum revision loops (default: 3)",
    )
    parser.add_argument(
        "--data-recursion-limit",
        type=int,
        default=512,
        help="Recursion limit for DataAgent (default: 512)",
    )
    parser.add_argument(
        "--experiment-recursion-limit",
        type=int,
        default=512,
        help="Recursion limit for ExperimentAgent (default: 512)",
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Custom session name (otherwise uses timestamp)",
    )
    parser.add_argument(
        "--models",
        choices=[
            "gpt-low-medium",
            "gpt-medium-high",
            "gemini-low-medium",
            "gemini-medium-high",
        ],
        default="gemini-low-medium",
        help="Model configuration to use (default: gemini-low-medium)",
    )

    args = parser.parse_args()

    # Register models based on choice
    logger.info(f"Registering models: {args.models}")
    match args.models:
        case "gpt-low-medium":
            register_gpt_low_medium_models()
        case "gpt-medium-high":
            register_gpt_medium_high_models()
        case "gemini-low-medium":
            register_gemini_low_medium_models()
        case "gemini-medium-high":
            register_gemini_medium_high_models()

    # Build user query and data description from MLE-Bench files
    logger.info("Building user query from MLE-Bench task files...")
    user_query, data_desc = build_mlebench_user_query(
        instructions_path=Path(args.instructions),
        description_path=Path(args.description),
    )
    logger.info(f"User query built: {len(user_query)} chars")
    logger.info(f"Data description built: {len(data_desc)} chars")

    # Run FullWorkflow
    result = run_full_workflow(
        data_path=args.data,
        workspace_path=args.workspace,
        user_query=user_query,
        data_desc=data_desc,
        repo_source=args.repo_source,
        max_revisions=args.max_revisions,
        data_agent_recursion_limit=args.data_recursion_limit,
        experiment_agent_recursion_limit=args.experiment_recursion_limit,
        session_name=args.session_name,
    )

    # Save summary
    result.save_summary()

    print(f"\nStatus: {result.final_status}")
