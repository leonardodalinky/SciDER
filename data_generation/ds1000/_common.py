"""DS-1000 internal plumbing — shared between ``generation`` and ``eval``.

* ``run_coding_task`` — invoke the native coding subagent on a single task,
  capture its full message history to ``<workspace>/coding_agent_history.json``,
  and return whatever solution file the agent wrote.
* ``is_workspace_complete`` / ``scan_completed_uids`` — skip-existing helpers
  that mirror the bench_workflows convention (look for a valid ``output.json``).
* ``write_output_json`` — atomic write of the per-task summary stub used by the
  skip-existing scan.

If a second benchmark needs the same primitives later, lift them up to
``data_generation/_common.py`` then — premature sharing makes both consumers
harder to evolve.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

# scider is on sys.path because the entry-point modules
# (data_generation.ds1000.generation / .eval) prepend the project root
# before importing this module.
from scider.core.code_env import LocalEnv, WorkspaceInitConfig
from scider.workflows.history_export import capture_messages, save_conversation_history

OUTPUT_FILENAME = "output.json"
HISTORY_FILENAME = "coding_agent_history.json"
PROMPT_FILENAME = "prompt.md"
DEFAULT_CODE_FILENAME = "code.py"


# --------------------------------------------------------------------------- #
# Coding subagent backend resolution (mirrors bench_workflows pattern)        #
# --------------------------------------------------------------------------- #

_CODING_BACKEND_ALIASES = {
    "v3": "claude_sdk",
    "claude_sdk": "claude_sdk",
    "native": "native",
}


def _resolve_coding_backend() -> str:
    raw = os.getenv("CODING_AGENT_VERSION", "native")
    return _CODING_BACKEND_ALIASES.get(raw, raw)


def _load_coding_subagent():
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
        "Set it to 'native' (default for openscider) or 'claude_sdk'."
    )


# --------------------------------------------------------------------------- #
# Skip-existing helpers                                                       #
# --------------------------------------------------------------------------- #


def is_workspace_complete(workspace: Path, *, code_filename: str = DEFAULT_CODE_FILENAME) -> bool:
    """A workspace is "done" iff:
    - ``output.json`` exists, parses, and has ``ok == True``
    - ``<code_filename>`` exists and is non-empty
    - ``coding_agent_history.json`` exists and is non-empty (the trajectory
      is what we actually care about for SFT data — no point keeping a
      run that lost its history)
    """
    out = workspace / OUTPUT_FILENAME
    code = workspace / code_filename
    hist = workspace / HISTORY_FILENAME
    try:
        if not out.is_file() or out.stat().st_size == 0:
            return False
        data = json.loads(out.read_text(encoding="utf-8"))
        if not data.get("ok"):
            return False
    except (OSError, json.JSONDecodeError):
        return False
    if not code.is_file() or code.stat().st_size == 0:
        return False
    if not hist.is_file() or hist.stat().st_size == 0:
        return False
    return True


def scan_completed_uids(
    output_root: Path, *, code_filename: str = DEFAULT_CODE_FILENAME
) -> set[str]:
    """Used by ``--skip-existing``: returns uids whose workspace passes
    ``is_workspace_complete``."""
    if not output_root.is_dir():
        return set()
    completed: set[str] = set()
    for uid_dir in output_root.iterdir():
        if not uid_dir.is_dir():
            continue
        if is_workspace_complete(uid_dir, code_filename=code_filename):
            completed.add(uid_dir.name)
    return completed


def write_output_json(workspace: Path, payload: dict[str, Any]) -> None:
    """Write the per-task summary stub atomically (tmp + rename)."""
    out = workspace / OUTPUT_FILENAME
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)


# --------------------------------------------------------------------------- #
# Coding-task runner                                                          #
# --------------------------------------------------------------------------- #


def run_coding_task(
    *,
    user_query: str,
    workspace_dir: Path,
    code_filename: str = DEFAULT_CODE_FILENAME,
    venv_path: Path | None = None,
) -> tuple[str, list]:
    """Run one coding task in ``workspace_dir`` and return (code_text, history).

    Captures the agent's full message history via ``capture_messages`` and
    persists it to ``<workspace>/coding_agent_history.json`` (the filename is
    the one ``train/prepare_data.py`` looks for).

    Returns ``(code_text, history_messages)``. ``code_text`` is "" if the
    agent failed to produce ``<code_filename>``.
    """
    workspace_dir = Path(workspace_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    code_path = workspace_dir / code_filename
    if code_path.exists():
        # Wipe stale solution from a previous attempt so we never silently
        # return last run's output if this run crashes early.
        code_path.unlink()

    init_config = WorkspaceInitConfig(
        env_manager="python",
        init_uv=False,
        venv_path=venv_path.resolve() if venv_path else None,
    )
    workspace = LocalEnv(
        working_dir=workspace_dir,
        create_dir_if_missing=True,
        init_config=init_config,
    )

    build_fn, StateCls, backend = _load_coding_subagent()
    logger.info("Coding subagent backend: {}", backend)

    state_kwargs: dict = {
        "user_query": user_query,
        "workspace": workspace,
        "data_summary": "",
    }
    if backend == "claude_sdk":
        state_kwargs.update(intermediate_full_output=True, skip_summary=True)

    coding_state = StateCls(**state_kwargs)
    coding_graph = build_fn().compile()
    logger.info("Executing coding graph in {}...", workspace_dir)

    with capture_messages() as captured:
        try:
            coding_graph.invoke(coding_state)
        except Exception as e:
            # Persist whatever we have so the trajectory isn't lost on crash.
            logger.warning("Coding graph crashed: {} — saving partial history", e)

    save_conversation_history(
        captured,
        workspace_dir / HISTORY_FILENAME,
        agent_name="coding",
    )

    code_text = ""
    if code_path.is_file():
        code_text = code_path.read_text(encoding="utf-8")
    return code_text, captured


# --------------------------------------------------------------------------- #
# Output-protocol prompt wrapper                                              #
# --------------------------------------------------------------------------- #


def output_protocol_block(code_filename: str = DEFAULT_CODE_FILENAME) -> str:
    """Standard footer appended to a benchmark's raw prompt so the agent
    knows to write its solution to a deterministic file.

    Kept separate so each generation workflow can decide whether to use it
    (some prompts already specify a different output contract)."""
    return (
        "## Output protocol\n\n"
        f"- Write your final solution to `./{code_filename}` in this workspace.\n"
        "- The harness reads that file directly — do NOT rely on chat output.\n"
        f"- Include all imports the solution needs at the top of `./{code_filename}`.\n"
        "- You may freely write scratch files (tests, prints) elsewhere; "
        f"only `./{code_filename}` is consumed.\n"
        f"- Before finishing, sanity-check the solution by running it (e.g. "
        f'`python ./{code_filename}` or `python -c "from code import ...; ..."`).\n'
    )
