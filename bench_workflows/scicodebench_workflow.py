"""SciCodeBench Workflow — run a single coding task through SciDER's Claude
coding subagent.

The SciCode evaluator (``SciCode/eval/scripts/gencode_scider.py``) calls
``run_coding_workflow`` once per (problem, sub-step) to produce a Python
solution snippet for that step.

Design (redesigned from the original markdown-parsing version):

* The agent is asked to write its final code to ``./code.py`` in a
  per-(problem, step) workspace under ``--workspace-root``. We then
  return the literal contents of that file. No more ``extract_python_
  script(response)`` regex parsing — the file IS the deliverable.
* Optional ``--scicode-venv`` is a shared Python environment (numpy,
  scipy, sympy, matplotlib, ...) prepended to PATH inside the agent's
  shell so it can ``python -c "from code import f; print(f(...))"``
  to self-test before declaring done.

Usage
-----
    python -m bench_workflows.scicodebench_workflow \\
        --query "Write a Python function add(a,b) returning a+b." \\
        --workspace-root ./workspace \\
        --problem-id smoke-test
"""

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

# Make scider + bench modules importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scider.core.code_env import LocalEnv, WorkspaceInitConfig
from scider.default.models import register_defaults_from_yaml

# CODING_AGENT_VERSION aliases — same mapping the experiment_agent uses
# (see scider/agents/experiment_agent/coding_subagent.py). The "v3" alias
# is the legacy default value.
_CODING_BACKEND_ALIASES = {
    "v3": "claude_sdk",
    "claude_sdk": "claude_sdk",
    "native": "native",
}


def _resolve_coding_backend() -> str:
    """Read ``CODING_AGENT_VERSION`` at call time (not import time).

    Reading at call time means a script that loads ``.env`` after this
    module is imported still gets the right backend.
    """
    raw = os.getenv("CODING_AGENT_VERSION", "native")
    return _CODING_BACKEND_ALIASES.get(raw, raw)


def _load_coding_subagent():
    """Return ``(build, StateCls, backend_name)`` for the selected backend.

    Importing here (lazy) keeps the import graph tight: a workflow that
    only ever uses Claude SDK doesn't pay to import the native subagent
    module, and vice-versa.
    """
    backend = _resolve_coding_backend()
    if backend == "native":
        from scider.agents.coding_subagent_native.build import build as build_fn
        from scider.agents.coding_subagent_native.state import NativeCodingAgentState

        return build_fn, NativeCodingAgentState, backend
    if backend == "claude_sdk":
        from scider.agents.coding_subagent_claude.build import build as build_fn
        from scider.agents.coding_subagent_claude.state import ClaudeCodingAgentState

        return build_fn, ClaudeCodingAgentState, backend
    raise ValueError(
        f"Unsupported CODING_AGENT_VERSION={os.getenv('CODING_AGENT_VERSION')!r}. "
        "Set it to 'claude_sdk' (default) or 'native'."
    )


# ---- Constants ----
ROLES_YAML_PATH = Path(__file__).parent / "model_configs" / "scicodebench_roles.yaml"
# Name of the file the agent writes its final solution to. Kept as a
# constant so gencode_scider.py and any future debug script can agree.
DEFAULT_CODE_FILENAME = "code.py"


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


def _wrap_user_query(user_query: str, code_filename: str) -> str:
    """Append delivery instructions to the SciCode prompt.

    SciCode's prompt template ends with the next-step function header and
    expects an implementation. We append a short "output protocol" block
    that pins the deliverable to a file (so we don't depend on the agent
    formatting code as markdown). The original SciCode prompt is left
    untouched above this block.
    """
    return (
        user_query.rstrip()
        + "\n\n"
        + "## Output protocol\n\n"
        + f"- Write your final solution to `./{code_filename}` in this workspace.\n"
        + "- The harness reads that file directly — do NOT rely on chat output.\n"
        + "- Include all imports the new function needs at the top of "
        + f"`./{code_filename}`.\n"
        + "- You may freely write scratch files (tests, prints) elsewhere; "
        + f"only `./{code_filename}` is consumed.\n"
        + f"- Before finishing, sanity-check the solution by running e.g. "
        + f'`python -c "from code import <func_name>; print(<func_name>(...))"` '
        + "in the workspace shell.\n"
    )


def run_coding_workflow(
    user_query: str,
    workspace_dir: str | Path,
    scicode_venv: Path | None = None,
    code_filename: str = DEFAULT_CODE_FILENAME,
) -> str:
    """Run a single coding task and return the contents of ``code.py``.

    Args:
        user_query: The coding task prompt (SciCode hands in the
            problem-spec + previous-step context + next function header).
        workspace_dir: Per-task working directory. The agent has full
            access here; the deliverable lands at
            ``<workspace_dir>/<code_filename>``.
        scicode_venv: Optional path to a shared Python venv (with numpy,
            scipy, etc. preinstalled). If set, ``<scicode_venv>/bin`` is
            prepended to PATH so the agent's shell calls resolve to that
            interpreter.
        code_filename: Override the deliverable filename. Default is
            ``code.py`` — change only if you know the harness reads
            something else.

    Returns:
        The literal text of ``<workspace_dir>/<code_filename>``. Returns
        ``""`` if the agent failed to create that file (caller should
        treat empty string as an error case for bookkeeping).
    """
    workspace_dir = Path(workspace_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Wipe any stale code.py from a previous run on this workspace, so we
    # never accidentally return last attempt's output if the agent crashes
    # before writing this attempt.
    code_path = workspace_dir / code_filename
    if code_path.exists():
        code_path.unlink()

    # Build the LocalEnv with the optional shared venv. ``env_manager=
    # "python"`` + ``init_uv=False`` mirrors astrovisbench: we don't want
    # the workspace to be uv-init'd into its own project; we want it to
    # see the shared venv's binaries on PATH.
    init_config = WorkspaceInitConfig(
        env_manager="python",
        init_uv=False,
        venv_path=scicode_venv.resolve() if scicode_venv else None,
    )
    workspace = LocalEnv(
        working_dir=workspace_dir,
        create_dir_if_missing=True,
        init_config=init_config,
    )

    build_fn, StateCls, backend = _load_coding_subagent()
    logger.info("Coding subagent backend: {}", backend)

    state_kwargs: dict = {
        "user_query": _wrap_user_query(user_query, code_filename),
        "workspace": workspace,
        "data_summary": "",
    }
    # Claude-SDK-only fields. Native state has no such knobs — it always
    # runs the standard query() loop and emits a summary; we don't care
    # about either, since ``code.py`` on disk is the deliverable.
    if backend == "claude_sdk":
        state_kwargs.update(intermediate_full_output=True, skip_summary=True)

    coding_state = StateCls(**state_kwargs)
    coding_graph = build_fn().compile()
    logger.info("Executing coding graph in {}...", workspace_dir)
    coding_graph.invoke(coding_state)

    if not code_path.is_file():
        logger.warning(
            "Agent did not produce {} in {}; returning empty string",
            code_filename,
            workspace_dir,
        )
        return ""

    code_text = code_path.read_text(encoding="utf-8")
    if not code_text.strip():
        logger.warning("{} exists but is empty in {}", code_filename, workspace_dir)
        return ""
    return code_text


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SciCodeBench workflow — runs one coding task through SciDER's "
            "Claude coding subagent. Primarily a library function used by "
            "SciCode/eval/scripts/gencode_scider.py; this CLI is for "
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
        "--workspace-root",
        required=True,
        help="Root dir for per-task workspaces. The actual workspace will "
        "be <workspace_root>/<problem_id>/, holding the code.py "
        "deliverable + the agent's full conversation trace.",
    )
    parser.add_argument(
        "--problem-id",
        default="smoke-test",
        help="Subdir name under workspace-root (e.g. '42_3' for problem 42 "
        "step 3). Default 'smoke-test'.",
    )
    parser.add_argument(
        "--scicode-venv",
        default=None,
        help="Optional path to a shared Python venv with the SciCode "
        "scientific stack (numpy, scipy, sympy, matplotlib, ...). When "
        "set, its bin/ is prepended to the agent's shell PATH so it can "
        "self-test the generated code.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output file to save the resulting code.py contents.",
    )
    args = parser.parse_args()

    _register_models_from_yaml()

    if args.query:
        user_query = args.query
    else:
        query_path = Path(args.query_file)
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        user_query = query_path.read_text(encoding="utf-8")
    logger.info("User query length: {} chars", len(user_query))

    workspace_dir = Path(args.workspace_root).resolve() / args.problem_id
    scicode_venv = Path(args.scicode_venv).resolve() if args.scicode_venv else None
    if scicode_venv is not None and not (scicode_venv / "bin" / "python").exists():
        logger.warning(
            "scicode_venv={} has no bin/python — PATH injection will still "
            "happen but the agent may not find the expected interpreter.",
            scicode_venv,
        )

    code = run_coding_workflow(
        user_query=user_query,
        workspace_dir=workspace_dir,
        scicode_venv=scicode_venv,
    )

    print("\n" + "=" * 80)
    print(f"CODE WRITTEN ({len(code)} chars) — from {workspace_dir / DEFAULT_CODE_FILENAME}")
    print("=" * 80)
    print(code)
    print("=" * 80)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(code, encoding="utf-8")
        logger.info("Result saved to: {}", output_path)


if __name__ == "__main__":
    _main()
