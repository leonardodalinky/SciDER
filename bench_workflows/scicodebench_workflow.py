"""SciCodeBench Workflow — run a single coding task through SciDER's Claude
coding subagent.

The SciCode evaluator (``SciCode/eval/scripts/gencode_scider.py``) calls
``run_coding_workflow`` once per (problem, sub-step) to produce a Python
solution snippet for that step. This wrapper registers role assignments
from ``bench_workflows/model_configs/scicodebench_roles.yaml`` and drives
the subagent; there's no data / experiment / critic pipeline (SciCode
gives the function header + step description and expects code back).

Usage
-----
    python -m bench_workflows.scicodebench_workflow \\
        --query "Create a function to calculate Fibonacci numbers" \\
        --workspace ./my_workspace
"""

import argparse
import sys
import tempfile
from pathlib import Path

from loguru import logger

# Make scider + bench modules importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scider.agents.coding_subagent_claude.build import build
from scider.agents.coding_subagent_claude.state import ClaudeCodingAgentState
from scider.core.code_env import LocalEnv
from scider.default.models import register_defaults_from_yaml

# ---- Constants ----
# The per-project model-role assignment for this benchmark. Edit the yaml
# to swap models — don't reintroduce a --models CLI knob; this matches the
# airsbench / astrovisbench / mlebench refactor.
ROLES_YAML_PATH = Path(__file__).parent / "model_configs" / "scicodebench_roles.yaml"


def _register_models_from_yaml() -> None:
    """Load role assignments from the scicodebench roles yaml."""
    if not ROLES_YAML_PATH.exists():
        raise FileNotFoundError(
            f"Role yaml missing at {ROLES_YAML_PATH}. This file pins the "
            "experiment_coding / ... roles for SciCodeBench; don't rename it."
        )
    logger.info("Registering roles from {}", ROLES_YAML_PATH)
    registered = register_defaults_from_yaml(ROLES_YAML_PATH)
    logger.info("Registered {} roles: {}", len(registered), sorted(registered))


def run_coding_workflow(user_query: str, workspace_dir: str | Path | None = None) -> str:
    """Run a single coding task through the Claude coding subagent.

    Args:
        user_query: The coding task description (the full prompt SciCode
            hands in — problem statement + previous-step context + next
            function header).
        workspace_dir: Working directory for the subagent. If None a
            fresh temp dir is created.

    Returns:
        The raw final output from Claude's last invocation (the generated
        Python snippet plus any surrounding explanation — SciCode's
        ``extract_python_script`` pulls the actual code out).
    """
    logger.info("Starting coding workflow with query: {}...", user_query[:100])

    if workspace_dir is None:
        workspace_dir = tempfile.mkdtemp(prefix="scider_coding_")
        logger.info("Using temporary workspace: {}", workspace_dir)

    workspace = LocalEnv(working_dir=workspace_dir, create_dir_if_missing=True)

    coding_state = ClaudeCodingAgentState(
        user_query=user_query,
        workspace=workspace,
        data_summary="",  # SciCode tasks are self-contained — no separate data blob
        intermediate_full_output=True,  # keep full output so we can find the final result
        skip_summary=True,  # SciCode wants raw code, not a narrative summary
    )

    coding_graph = build().compile()

    logger.info("Executing coding graph...")
    result_state = coding_graph.invoke(coding_state)

    # The coding subagent pipeline accumulates per-node dicts under
    # ``intermediate_state``. We want the LAST ``claude`` node's
    # ``final_result``. Missing that indicates the graph terminated early
    # and the caller should see an empty string / handle the failure.
    intermediate_states = result_state.get("intermediate_state", [])
    claude_states = [s for s in intermediate_states if s.get("node_name") == "claude"]
    if not claude_states:
        logger.warning("No claude node found in intermediate states")
        return ""

    last_claude_output = claude_states[-1].get("_raw_claude_result", {}).get("final_result")
    assert last_claude_output is not None, "No final_result found in the last claude node output"
    return last_claude_output


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SciCodeBench workflow — runs one coding task through SciDER's "
            "Claude coding subagent. Primarily used as a library function "
            "by SciCode/eval/scripts/gencode_scider.py; this CLI is for "
            "smoke-testing a single query."
        ),
        prog="python -m bench_workflows.scicodebench_workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query",
        "-q",
        help="Coding task query (inline text).",
    )
    query_group.add_argument(
        "--query-file",
        "-f",
        help="Path to file containing the coding task query.",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        default=None,
        help="Workspace directory for the coding task (default: creates a temp directory).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output file to save the result.",
    )
    args = parser.parse_args()

    # 1. Register models from yaml (errors here surface before we do real work).
    _register_models_from_yaml()

    # 2. Resolve the query.
    if args.query:
        user_query = args.query
    else:
        query_path = Path(args.query_file)
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        user_query = query_path.read_text(encoding="utf-8")
    logger.info("User query length: {} chars", len(user_query))

    # 3. Run.
    result = run_coding_workflow(
        user_query=user_query,
        workspace_dir=args.workspace,
    )

    print("\n" + "=" * 80)
    print("CODING WORKFLOW RESULT")
    print("=" * 80)
    print(result)
    print("=" * 80)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result, encoding="utf-8")
        logger.info("Result saved to: {}", output_path)


if __name__ == "__main__":
    _main()
