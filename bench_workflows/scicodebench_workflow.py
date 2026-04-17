"""
Workflow for running coding tasks using Claude coding subagent.
"""

import sys
import tempfile
from pathlib import Path

from loguru import logger

# Add parent directory to path to find scider and bench modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scider.agents.coding_subagent_claude.build import build
from scider.agents.coding_subagent_claude.state import ClaudeCodingAgentState
from scider.core.code_env import LocalEnv
from scider.default.models import register_defaults_from_yaml

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


def run_coding_workflow(user_query: str, workspace_dir: str | Path | None = None) -> str:
    """
    Run a simple coding workflow using Claude coding subagent.

    Args:
        user_query: The coding task description
        workspace_dir: The working directory for the coding task. If None, a temporary directory is created.

    Returns:
        The output from Claude's coding execution
    """
    logger.info(f"Starting coding workflow with query: {user_query[:100]}...")

    # Create workspace environment
    if workspace_dir is None:
        workspace_dir = tempfile.mkdtemp(prefix="scider_coding_")
        logger.info(f"Using temporary workspace: {workspace_dir}")

    workspace = LocalEnv(working_dir=workspace_dir, create_dir_if_missing=True)

    # Create agent state
    coding_state = ClaudeCodingAgentState(
        user_query=user_query,
        workspace=workspace,
        data_summary="",  # No background data needed for simple coding tasks
        intermediate_full_output=True,  # Store full output in intermediate state
        skip_summary=True,  # Skip final summary to get full output directly
    )

    # Build and compile the graph
    coding_graph = build().compile()

    # Execute the workflow
    logger.info("Executing coding graph...")
    result_state = coding_graph.invoke(coding_state)

    # Extract intermediate states and find the last 'claude' node output
    intermediate_states = result_state.get("intermediate_state", [])

    # Filter for claude nodes and get the last one
    claude_states = [state for state in intermediate_states if state.get("node_name") == "claude"]

    if not claude_states:
        logger.warning("No claude node found in intermediate states")
        return ""

    # Get the output from the last claude node
    last_claude_output = claude_states[-1].get("_raw_claude_result", {}).get("final_result", None)
    assert last_claude_output is not None, "No final_result found in the last claude node output"

    return last_claude_output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SciCodeBench Workflow - Run simple coding tasks using Claude coding subagent",
        prog="python -m bench_workflows.scicodebench_workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with inline query
  python -m bench_workflows.scicodebench_workflow \\
      --query "Create a function to calculate fibonacci numbers"

  # Run with query from file
  python -m bench_workflows.scicodebench_workflow \\
      --query-file task.txt \\
      --workspace ./my_workspace

  # Use specific model configuration
  python -m bench_workflows.scicodebench_workflow \\
      --query "Implement a binary search tree" \\
      --models gpt-medium-high
        """,
    )

    # Query input (mutually exclusive)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query",
        "-q",
        help="Coding task query (inline text)",
    )
    query_group.add_argument(
        "--query-file",
        "-f",
        help="Path to file containing the coding task query",
    )

    # Optional arguments
    parser.add_argument(
        "--workspace",
        "-w",
        default=None,
        help="Workspace directory for the coding task (default: creates temp directory)",
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
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output file to save the result",
    )

    args = parser.parse_args()

    # Register models from yaml preset.
    _register_models_from_yaml(args.models)

    # Get user query
    if args.query:
        user_query = args.query
    else:
        query_path = Path(args.query_file)
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        user_query = query_path.read_text(encoding="utf-8")

    logger.info(f"User query length: {len(user_query)} chars")

    # Run coding workflow
    result = run_coding_workflow(
        user_query=user_query,
        workspace_dir=args.workspace,
    )

    # Print result
    print("\n" + "=" * 80)
    print("CODING WORKFLOW RESULT")
    print("=" * 80)
    print(result)
    print("=" * 80)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result, encoding="utf-8")
        logger.info(f"Result saved to: {output_path}")
        output_path.write_text(result, encoding="utf-8")
        logger.info(f"Result saved to: {output_path}")
