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

from scider.core import constant

CLAUDE_SDK_MODEL = os.getenv("CLAUDE_SDK_MODEL", "claude-haiku-4-5")


def _resolve_cwd(cwd: str | None, agent_state) -> Path:
    if cwd:
        return Path(os.path.expandvars(cwd)).expanduser().resolve()
    if agent_state is not None:
        # Experiment/Data/Writing agents store the LocalEnv as `workspace`.
        # (Historical name `local_env` kept as a fallback for any legacy state.)
        for attr in ("workspace", "local_env"):
            env = getattr(agent_state, attr, None)
            wd = getattr(env, "working_dir", None) if env is not None else None
            if wd is not None:
                try:
                    return Path(wd).resolve()
                except Exception:
                    pass
        if hasattr(agent_state, "repo_dir") and agent_state.repo_dir:
            try:
                return Path(agent_state.repo_dir).resolve()
            except Exception:
                pass
    return Path.cwd().resolve()


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
                model=CLAUDE_SDK_MODEL,
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
