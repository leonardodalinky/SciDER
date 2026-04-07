"""Register the coding subagent in AgentRegistry.

Wraps the existing Claude Agent SDK / OpenHands / Native coding subagent
so it can be invoked via `Agent(prompt="...", subagent_type="coding")`.

Supported CODING_AGENT_VERSION values:
  - "claude_sdk" or "v3" (default): Claude Agent SDK
  - "openhands" or "v2": OpenHands (requires SCIDER_ENABLE_OPENHANDS=1)
  - "native": SciDER built-in coding subagent
"""

from __future__ import annotations

import os

from loguru import logger

from scider.tools.agent_tool import AgentRegistry, AgentType

# Normalize version names: accept both old (v2/v3) and new names
_VERSION_ALIASES = {
    "v3": "claude_sdk",
    "v2": "openhands",
    "claude_sdk": "claude_sdk",
    "openhands": "openhands",
    "native": "native",
}


def _resolve_version() -> str:
    """Read CODING_AGENT_VERSION from env at call time (NOT import time).

    Reading at call time means re-registration picks up updated env vars,
    which matters when .env is loaded after this module is first imported.
    """
    raw = os.getenv("CODING_AGENT_VERSION", "v3")
    return _VERSION_ALIASES.get(raw, raw)


def register_coding_subagent() -> str | None:
    """Build and register the coding subagent graph.

    Reads CODING_AGENT_VERSION from environment EVERY time it's called,
    so changing the env var and re-calling will swap the registered backend.
    Returns the registered backend name, or None if registration failed.
    """
    coding_agent_version = _resolve_version()
    _OPENHANDS_ENABLED = os.getenv("SCIDER_ENABLE_OPENHANDS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    match coding_agent_version:
        case "native":
            from scider.agents.coding_subagent_native.build import build as coding_build_fn
            from scider.agents.coding_subagent_native.state import (
                NativeCodingAgentState as CodingAgentState,
            )

            compiled_graph = coding_build_fn().compile()
        case "openhands":
            if not _OPENHANDS_ENABLED:
                logger.warning(
                    "CODING_AGENT_VERSION=openhands requires SCIDER_ENABLE_OPENHANDS=1. "
                    "Coding subagent not registered."
                )
                return None
            from scider.agents.coding_subagent_openhands import build as coding_build
            from scider.agents.coding_subagent_openhands.state import CodingAgentState

            compiled_graph = coding_build().compile()
        case "claude_sdk":
            from scider.agents.coding_subagent_claude.build import build as coding_build_fn
            from scider.agents.coding_subagent_claude.state import (
                ClaudeCodingAgentState as CodingAgentState,
            )

            compiled_graph = coding_build_fn().compile()
        case _:
            logger.warning(
                "Unsupported CODING_AGENT_VERSION: {}. Coding subagent not registered.",
                coding_agent_version,
            )
            return None

    def _build_coding_state(prompt: str, parent_state) -> dict:
        """Build CodingAgentState kwargs from AgentTool prompt + parent state."""
        data_summary = ""
        if parent_state is not None:
            if hasattr(parent_state, "data_summary"):
                data_summary = getattr(parent_state, "data_summary", "") or ""
            elif hasattr(parent_state, "data_desc"):
                data_summary = getattr(parent_state, "data_desc", "") or ""

        workspace = None
        if parent_state is not None and hasattr(parent_state, "workspace"):
            workspace = parent_state.workspace

        kwargs = {
            "user_query": prompt,
            "data_summary": data_summary,
        }
        # Claude/OpenHands subagents skip summary (experiment agent generates its own);
        # native subagent always generates summary via LLM.
        if coding_agent_version != "native":
            kwargs["skip_summary"] = True
        if workspace is not None:
            kwargs["workspace"] = workspace
        return kwargs

    def _extract_coding_result(result: dict) -> dict:
        """Extract useful results from coding subagent output."""
        output_summary = result.get("output_summary", "")
        # If skip_summary was True, extract from last assistant message in history
        if not output_summary and "history" in result:
            for msg in reversed(result["history"]):
                if hasattr(msg, "role") and msg.role == "assistant" and msg.content:
                    output_summary = msg.content[:5000]
                    break
        return {
            "summary": output_summary,
            "status": "completed",
        }

    AgentRegistry.register(
        AgentType(
            name="coding",
            description=(
                f"Delegate complex coding tasks to a specialized coding agent ({coding_agent_version}). "
                "The agent can read, write, edit files, run commands, and manage a full coding workflow. "
                "Use for tasks requiring multi-file changes, environment setup, or significant implementation work."
            ),
            compiled_graph=compiled_graph,
            state_cls=CodingAgentState,
            state_builder=_build_coding_state,
            result_extractor=_extract_coding_result,
        )
    )
    logger.info("Registered coding subagent (backend={})", coding_agent_version)
    return coding_agent_version


# Backwards-compat alias for the old private name
_register_coding_subagent = register_coding_subagent


# Initial registration on import. Note: callers should re-invoke
# register_coding_subagent() after .env loading to ensure the env var is fresh.
try:
    CODING_AGENT_VERSION = register_coding_subagent() or _resolve_version()
except Exception as e:
    logger.warning("Failed to register coding subagent: {}", e)
    CODING_AGENT_VERSION = _resolve_version()
