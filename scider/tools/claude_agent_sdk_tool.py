"""
Claude Agent SDK toolset.

Implements a SciDER tool wrapper around Anthropic's Claude Agent SDK so the
coding subagent can delegate repo edits to Claude Code runtime programmatically.

Reference: https://platform.claude.com/docs/en/agent-sdk/overview
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from ..core import constant
from .registry import register_tool, register_toolset_desc

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")


register_toolset_desc(
    "claude_agent_sdk",
    "Claude Agent SDK toolset. Runs Claude Code runtime programmatically in a target folder. "
    "Requires `claude-agent-sdk` to be installed and `ANTHROPIC_API_KEY` to be set.",
)


def _resolve_cwd(cwd: str | None, agent_state) -> Path:
    if cwd:
        return Path(os.path.expandvars(cwd)).expanduser().resolve()
    if agent_state is not None:
        if hasattr(agent_state, "local_env") and hasattr(agent_state.local_env, "working_dir"):
            try:
                return Path(agent_state.local_env.working_dir).resolve()
            except Exception:
                pass
        if hasattr(agent_state, "repo_dir") and agent_state.repo_dir:
            try:
                return Path(agent_state.repo_dir).resolve()
            except Exception:
                pass
    return Path.cwd().resolve()


@register_tool(
    "claude_agent_sdk",
    {
        "type": "function",
        "function": {
            "name": "run_claude_agent_sdk",
            "description": (
                "Run Claude Agent SDK (Claude Code runtime as a library) to apply edits in a target directory. "
                "This streams messages from the SDK and returns a compact summary. "
                "Requires `pip install claude-agent-sdk` and `ANTHROPIC_API_KEY`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task prompt for the Claude agent (e.g. 'Find and fix the bug in auth.py').",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the agent (defaults to repo_dir/local_env.working_dir).",
                        "default": None,
                    },
                    "allowed_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Allowed tools for the Claude agent SDK. "
                            "Common: Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch."
                        ),
                        "default": ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
                    },
                    "permission_mode": {
                        "type": "string",
                        "description": "Claude Agent SDK permission mode (e.g. 'acceptEdits').",
                        "default": "acceptEdits",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
)
def run_claude_agent_sdk(
    prompt: str,
    cwd: str | None = None,
    allowed_tools: list[str] | None = None,
    permission_mode: str = "acceptEdits",
    **kwargs,
) -> str:
    if not prompt or not prompt.strip():
        return json.dumps({"error": "prompt must be a non-empty string"})

    # Import lazily so SciDER can run without the SDK installed.
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore
    except Exception as e:
        return json.dumps(
            {
                "error": "Claude Agent SDK not available",
                "detail": str(e),
                "hint": "Install with `pip install claude-agent-sdk` and set ANTHROPIC_API_KEY.",
            }
        )

    agent_state = kwargs.get(constant.__AGENT_STATE_NAME__)
    working_dir = _resolve_cwd(cwd, agent_state)
    if not working_dir.exists() or not working_dir.is_dir():
        return json.dumps({"error": f"Invalid working directory: {str(working_dir)}"})

    tools = allowed_tools or ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]

    async def _run() -> dict:
        # Run inside working dir so SDK built-in tools operate on the target project.
        old = Path.cwd()
        os.chdir(str(working_dir))
        try:
            msgs: list[dict] = []
            options = ClaudeAgentOptions(
                model=CLAUDE_MODEL,
                allowed_tools=tools,
                permission_mode=permission_mode,
                max_buffer_size=10 * 1024 * 1024,  # 10 MB
            )

            final_result = None

            async for message in query(prompt=prompt, options=options):
                # message is a pydantic-ish object; try best-effort serialization
                try:
                    data = asdict(message)
                except Exception:
                    try:
                        data = dict(message)  # type: ignore[arg-type]
                    except Exception:
                        data = {"raw": str(message)}
                msgs.append(data)
                from claude_agent_sdk import ResultMessage

                if isinstance(message, ResultMessage) and message.result is not None:
                    final_result = message.result

            return {
                "cwd": str(working_dir),
                "allowed_tools": tools,
                "permission_mode": permission_mode,
                "messages": msgs[-3:],  # keep tail only
                "message_count": len(msgs),
                "final_result": final_result,
            }
        finally:
            os.chdir(old)

    try:
        # NOTE: Jupyter/Notebook typically runs an event loop already, so calling
        # asyncio.run() directly will raise:
        #   RuntimeError: asyncio.run() cannot be called from a running event loop
        #
        # To support both script and notebook environments, we:
        # - Run asyncio.run(...) normally when no loop is running
        # - Otherwise run asyncio.run(...) inside a dedicated thread
        def _run_sync() -> dict:
            # Create the coroutine inside this function to avoid "coroutine was never awaited"
            # when asyncio.run fails fast due to an existing running loop.
            return asyncio.run(_run())

        try:
            asyncio.get_running_loop()
            in_running_loop = True
        except RuntimeError:
            in_running_loop = False

        if not in_running_loop:
            result = _run_sync()
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as ex:
                result = ex.submit(_run_sync).result()

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": "Claude Agent SDK execution failed", "detail": str(e)})
