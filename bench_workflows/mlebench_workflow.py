"""
MLE-Bench Workflow

Simple wrapper for running SciDER FullWorkflow on MLE-Bench competition tasks.

MLE-Bench provides:
- instructions.md: Specific task instructions (used as user_query)
- description.md: Overall task background description

This wrapper register models, reads these files, builds user_query, and invokes FullWorkflow.
"""

import sys
from pathlib import Path

from loguru import logger

# Add parent directory to path to find scider and bench modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scider.default.models import register_defaults_from_yaml
from scider.workflows.full_workflow import run_full_workflow

# Model presets live under model_settings/presets/ as yaml files. The
# --models CLI flag picks one of these; to add a new preset, drop a yaml
# file in the corresponding directory and add an entry below.
PROJECT_ROOT = Path(__file__).parent.parent
PRESETS_DIR = PROJECT_ROOT / "model_settings" / "presets"
PRESET_MAP: dict[str, Path] = {
    "gemini-low-medium": PRESETS_DIR / "gemini" / "low_medium.yaml",
    "gemini-medium-high": PRESETS_DIR / "gemini" / "medium_high.yaml",
    "gemini3-medium-high": PRESETS_DIR / "gemini" / "gemini3_medium_high.yaml",
    "gpt-low-medium": PRESETS_DIR / "gpt" / "low_medium.yaml",
    "gpt-medium-high": PRESETS_DIR / "gpt" / "medium_high.yaml",
}


def _register_models_from_yaml(preset: str) -> None:
    """Load role assignments from the yaml file for ``preset``."""
    yaml_path = PRESET_MAP[preset]
    if not yaml_path.exists():
        raise FileNotFoundError(f"Preset yaml missing at {yaml_path}")
    logger.info("Registering roles from {}", yaml_path)
    registered = register_defaults_from_yaml(yaml_path)
    logger.info("Registered {} roles: {}", len(registered), sorted(registered))


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
        description="MLE-Bench Workflow - Run SciDER on MLE-Bench competition tasks",
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
            "gemini3-medium-high",
        ],
        default="gemini-low-medium",
        help="Model configuration to use (default: gemini-low-medium)",
    )

    args = parser.parse_args()

    # Register models from yaml preset.
    _register_models_from_yaml(args.models)

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
