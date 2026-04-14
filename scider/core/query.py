"""
Core query function that unifies LLM chat and tool calling into a single agent loop.

Inspired by Claude Code's query() (src/query.ts), adapted for SciDER's
synchronous, non-streaming architecture.

The loop: call LLM → check for tool calls → execute tools → repeat until done.
"""

from __future__ import annotations

import contextlib
import enum
import inspect
import json
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import litellm
from loguru import logger

from ..tools import Tool, ToolRegistry
from .compact import CompactState, apply_autocompact, run_compression_pipeline
from .constant import __AGENT_STATE_NAME__, __CTX_NAME__
from .llms import ModelRegistry
from .plan_mode import PlanModeState, QueryMode
from .types import HistoryState, Message


class StopReason(str, enum.Enum):
    """Known stop reasons for the query loop."""

    END_TURN = "end_turn"
    MAX_TURNS = "max_turns"
    STOP_HOOK = "stop_hook"
    ERROR = "error"
    CONTEXT_TOO_LONG = "context_too_long"


class StopQuery(Exception):
    """Raised by on_tool_result callback to break out of the query loop early."""

    def __init__(self, reason: str = StopReason.STOP_HOOK):
        super().__init__(reason)
        self.reason = reason


@dataclass
class QueryResult:
    """Result of a query() call."""

    messages: list[Message]
    stop_reason: StopReason | str  # Known enum or custom string for unexpected cases
    turn_count: int
    error: Exception | None = None

    @property
    def last_message(self) -> Message | None:
        return self.messages[-1] if self.messages else None


def gather_tools(tool_names: list[str]) -> dict[str, Tool]:
    """Gather tools by name from ToolRegistry."""
    return ToolRegistry.get_tools_by_names(tool_names)


def _execute_tool_calls(
    *,
    tool_calls: list,
    function_map: dict[str, Callable],
    agent_state: HistoryState,
    agent_name: str,
    tool_execution_context: AbstractContextManager | None,
    ctx_dict: dict | None,
) -> tuple[list[Message], bool]:
    """Execute a list of tool calls, returning (tool response Messages, all_errors flag)."""
    results: list[Message] = []
    error_count = 0
    ctx_mgr = tool_execution_context or contextlib.nullcontext()

    for tool_call in tool_calls:
        tool_name = tool_call.function.name

        # Tool not found
        if tool_name not in function_map:
            error_msg = f"Tool {tool_name} not found"
            results.append(
                Message(
                    role="tool",
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    content=error_msg,
                ).with_log()
            )
            error_count += 1
            continue

        # Parse arguments
        try:
            args = json.loads(tool_call.function.arguments)
            assert isinstance(args, dict)
        except (json.JSONDecodeError, AssertionError) as e:
            error_msg = f"Invalid tool arguments: {e}"
            results.append(
                Message(
                    role="tool",
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    content=error_msg,
                ).with_log()
            )
            error_count += 1
            continue

        # Inject context for tool execution
        func = function_map[tool_name]
        if getattr(func, "__is_new_style_tool__", False):
            # New-style tool: inject ToolContext
            from ..tools.base import ToolContext

            args["__tool_context__"] = ToolContext(
                agent_name=agent_name,
                agent_state=agent_state,
                extra=ctx_dict or {"current_agent": agent_name},
            )
        else:
            # Legacy tool: inspect-based injection
            sig = inspect.signature(func)
            if __AGENT_STATE_NAME__ in sig.parameters:
                args[__AGENT_STATE_NAME__] = agent_state
            if __CTX_NAME__ in sig.parameters:
                args[__CTX_NAME__] = ctx_dict or {"current_agent": agent_name}

        # Permission check (before execution)
        from .permissions import check_tool_permission

        # Build input string for rule matching (e.g., Bash command)
        perm_input = args.get("command") or args.get("file_path") or args.get("path")
        perm_result = check_tool_permission(tool_name, perm_input)

        # Also check tool-level permissions for new-style tools
        if perm_result.allowed and getattr(func, "__is_new_style_tool__", False):
            base_tools: dict = getattr(ToolRegistry.instance(), "_base_tools", {})
            if tool_name in base_tools:
                tool_ctx = args.get("__tool_context__")
                if tool_ctx:
                    perm_result = base_tools[tool_name].check_permissions(args, tool_ctx)

        if not perm_result.allowed:
            results.append(
                Message(
                    role="tool",
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    content=f"Permission denied: {perm_result.reason}",
                ).with_log()
            )
            error_count += 1
            continue

        # Execute the tool
        try:
            with ctx_mgr:
                result = func(**args)

            # ToolResult carries both text and image attachments; plain str
            # results still work unchanged.
            from ..tools.base import ToolResult as _ToolResult

            if isinstance(result, _ToolResult):
                text = result.text
                images = [
                    {"media_type": img.media_type, "data": img.data} for img in result.images
                ] or None
            else:
                text = str(result)
                images = None

            results.append(
                Message(
                    role="tool",
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    content=text,
                    tool_result_images=images,
                ).with_log()
            )
        except Exception as e:
            error_msg = f"Tool {tool_name} execution failed: {e}"
            results.append(
                Message(
                    role="tool",
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    content=error_msg,
                ).with_log()
            )
            error_count += 1

    all_errors = len(results) > 0 and error_count == len(results)
    return results, all_errors


def _try_compression_recovery(
    agent_state: HistoryState,
) -> bool:
    """Attempt emergency autocompact to recover from context window overflow.

    Returns True if compression was performed, False otherwise.
    """
    try:
        success = apply_autocompact(agent_state, CompactState())
        if success:
            logger.info("Emergency autocompact recovery succeeded")
        return success
    except Exception as recovery_error:
        logger.error("Compression recovery failed: {}", recovery_error)
        return False


def _call_llm(
    *,
    model_name: str,
    agent_state: HistoryState,
    system_prompt: str,
    tool_names: list[str],
    agent_name: str,
    tool_choice: str | None,
) -> Message:
    """Call ModelRegistry.completion with proper history handling."""
    history = agent_state.messages
    if len(history) == 0 or all(msg.role == "system" for msg in history):
        logger.warning(
            "message history is empty or only contains system messages, adding dummy user message"
        )
        history = [
            Message(
                role="user",
                content="Please continue with the task.",
                agent_sender=agent_name,
                is_meta=True,
            )
        ]

    return ModelRegistry.completion(
        model_name,
        history,
        system_prompt=(Message(role="system", content=system_prompt).with_log(cond=False).content),
        agent_sender=agent_name,
        tools=tool_names if tool_names else None,
        tool_choice=tool_choice,
    ).with_log()


_PLAN_MODE_SYSTEM_REMINDER = """

<system-reminder>
Plan mode is active. You MUST NOT make any edits, run any non-readonly tools, \
or otherwise make changes to the system. Only read-only tools are available.

You should:
1. Explore the codebase and data to understand the task
2. Design your approach
3. Write a complete plan
4. Call ExitPlanMode with your plan to submit it for approval

Only after the plan is approved can you execute it.
</system-reminder>
"""


def _is_read_only_tool(name: str) -> bool:
    """Check if a tool is read-only (safe for plan mode).

    Calls is_read_only() with no kwargs (static check).
    Tools like BashTool return False here (command-dependent),
    while Read/Glob/Grep return True (always read-only).
    """
    base_tools: dict = getattr(ToolRegistry.instance(), "_base_tools", {})
    if name in base_tools:
        return base_tools[name].is_read_only()
    return False


def _wait_for_background_tasks(
    agent_state: HistoryState,
    agent_name: str,
    all_messages: list[Message],
    poll_interval: float = 5.0,
    max_wait: float = 43200.0,  # 12 hours
) -> bool:
    """Wait for running background tasks before allowing END_TURN.

    If background tasks are running, blocks until at least one completes,
    then drains notifications and injects them into the conversation.
    Returns True if notifications were injected (caller should continue the loop).
    Returns False if no tasks are running (caller can END_TURN).
    """
    import time

    from .task import TaskManager, TaskStatus

    running = [t for t in TaskManager.list_tasks() if t.status == TaskStatus.RUNNING]
    if not running:
        return False

    task_descs = ", ".join(f"{t.id} ({t.description[:40]})" for t in running)
    logger.info("Waiting for {} background task(s): {}", len(running), task_descs)

    # Inject a meta message so the LLM knows we're waiting
    wait_msg = Message(
        role="user",
        content=(
            f"<system-reminder>\n"
            f"You have {len(running)} background task(s) still running: {task_descs}.\n"
            f"Waiting for completion before proceeding...\n"
            f"</system-reminder>"
        ),
        agent_sender=agent_name,
        is_meta=True,
    )
    agent_state.add_message(wait_msg)
    all_messages.append(wait_msg)

    # Poll until at least one task completes
    start = time.time()
    while time.time() - start < max_wait:
        notifications = TaskManager.drain_notifications()
        if notifications:
            for notification in notifications:
                notify_msg = Message(
                    role="user",
                    content=f"<system-reminder>\n{notification}\n</system-reminder>",
                    agent_sender=agent_name,
                    is_meta=True,
                )
                agent_state.add_message(notify_msg)
                all_messages.append(notify_msg)
            return True

        # Check if any tasks are still running
        still_running = [t for t in TaskManager.list_tasks() if t.status == TaskStatus.RUNNING]
        if not still_running:
            # All tasks finished but notifications were already drained earlier
            return False

        time.sleep(poll_interval)

    # Timed out waiting
    logger.warning("Timed out waiting for background tasks after {}s", max_wait)
    timeout_msg = Message(
        role="user",
        content=(
            "<system-reminder>\n"
            "Background task wait timed out. Some tasks may still be running.\n"
            "</system-reminder>"
        ),
        agent_sender=agent_name,
        is_meta=True,
    )
    agent_state.add_message(timeout_msg)
    all_messages.append(timeout_msg)
    return True


def query(
    *,
    model_name: str,
    agent_state: HistoryState,
    system_prompt: str,
    tools: dict[str, Tool],
    agent_name: str,
    max_turns: int = 64,
    tool_execution_context: AbstractContextManager | None = None,
    on_tool_result: Callable[[str, str, Message], None] | None = None,
    ctx_dict: dict | None = None,
    tool_choice: str | None = None,
) -> QueryResult:
    """Core agent loop: call LLM → execute tools → repeat until done.

    Args:
        model_name: Registered model name in ModelRegistry (e.g. "data").
        agent_state: Any state extending HistoryState. Mutated in-place via add_message().
        system_prompt: Pre-rendered system prompt string.
        tools: Tool name -> Tool mapping, typically from gather_tools().
        agent_name: Agent identifier for Message.agent_sender.
        max_turns: Maximum number of LLM calls before stopping. Default 64.
        tool_execution_context: Optional context manager for tool execution
            (e.g. agent_state.workspace for data agent, agent_state.local_env for critic).
        on_tool_result: Optional callback (tool_name, tool_call_id, msg) called after
            each tool execution. May raise StopQuery to break the loop early.
        ctx_dict: Dict injected into tools that accept a ctx parameter.
        tool_choice: Passed to ModelRegistry.completion().

    Returns:
        QueryResult with all produced messages, stop reason, and turn count.
    """
    all_messages: list[Message] = []
    function_map = {tool.name: tool.func for tool in tools.values()}
    consecutive_error_turns = 0
    prev_turn_had_errors = False  # track if last tool execution had any errors
    gave_up_nudges = 0  # how many times we nudged the model to retry after giving up
    empty_response_nudges = (
        0  # how many times we nudged after empty (no content + no tools) response
    )
    compact_state = CompactState()

    # --- Set tool results dir to workspace if available ---
    from .compact import set_tool_results_dir

    if hasattr(agent_state, "workspace") and hasattr(agent_state.workspace, "working_dir"):
        workspace_dir = str(agent_state.workspace.working_dir)
        set_tool_results_dir(str(Path(workspace_dir) / ".tool-results"))
    else:
        set_tool_results_dir(None)  # use default

    # --- Inject tool-specific prompts into system prompt ---
    from ..tools.registry import get_tool_prompts

    tool_prompts = get_tool_prompts([tool.name for tool in tools.values()])
    if tool_prompts:
        system_prompt = system_prompt + "\n\n# Tool-specific guidance\n" + "\n".join(tool_prompts)

    # --- Inject skill prompts into system prompt ---
    # Preloaded skills get full content; on-demand skills get listing only (use Skill tool)
    from .skills import SkillRegistry

    SkillRegistry.instance().load_default_directories()
    skill_section = SkillRegistry.instance().build_system_prompt_section(agent_name)
    if skill_section:
        system_prompt = system_prompt + "\n\n# Skills\n" + skill_section

    # --- Inject memory index into system prompt ---
    from .memory import build_memory_prompt_section

    memory_section = build_memory_prompt_section()
    if memory_section:
        system_prompt = system_prompt + "\n\n# Memory\n" + memory_section

    # --- Inject SCIDER.md project context ---
    from .scider_context import load_scider_md

    _ws = getattr(getattr(agent_state, "workspace", None), "working_dir", None)
    scider_md = load_scider_md(_ws)
    if scider_md:
        system_prompt = system_prompt + "\n\n# Project Instructions (SCIDER.md)\n" + scider_md

    # Plan mode state — shared with tools via ToolContext.extra
    plan_mode_state = PlanModeState()
    if ctx_dict is None:
        ctx_dict = {"current_agent": agent_name}
    ctx_dict["plan_mode_state"] = plan_mode_state

    for turn in range(max_turns):
        # --- Step 0: Compression pipeline (before each LLM call) ---
        run_compression_pipeline(agent_state, compact_state)

        # --- Step 0.1: Inject task notifications (from background tasks) ---
        from .task import TaskManager

        for notification in TaskManager.drain_notifications():
            notify_msg = Message(
                role="user",
                content=f"<system-reminder>\n{notification}\n</system-reminder>",
                agent_sender=agent_name,
                is_meta=True,
            )
            agent_state.add_message(notify_msg)
            all_messages.append(notify_msg)

        # --- Step 0.5: Filter tools by plan mode ---
        if plan_mode_state.is_plan_mode:
            active_tools = {name: tool for name, tool in tools.items() if _is_read_only_tool(name)}
        else:
            active_tools = tools
        active_tool_names = [tool.name for tool in active_tools.values()]

        # --- Step 0.6: Inject plan mode / plan / todo reminders into system prompt ---
        effective_system_prompt = system_prompt
        if plan_mode_state.is_plan_mode:
            effective_system_prompt = system_prompt + _PLAN_MODE_SYSTEM_REMINDER

        # Remind model of approved plan (if any)
        if plan_mode_state.plan_content and plan_mode_state.plan_approved:
            effective_system_prompt += (
                "\n\n<system-reminder>\n"
                "You have an approved plan. Follow it step by step:\n\n"
                f"{plan_mode_state.plan_content}\n"
                "</system-reminder>"
            )

        # Remind model of current todo list (if any)
        current_todos = ctx_dict.get("todos") if ctx_dict else None
        if current_todos:
            status_icons = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}
            todo_lines = [
                f"{status_icons.get(t['status'], '[ ]')} {t['content']}" for t in current_todos
            ]
            effective_system_prompt += (
                "\n\n<system-reminder>\n"
                "Current task list:\n"
                + "\n".join(todo_lines)
                + "\n\nContinue with the current in-progress task."
                "\n</system-reminder>"
            )

        # --- Step 1: LLM call (with error recovery) ---
        try:
            assistant_msg = _call_llm(
                model_name=model_name,
                agent_state=agent_state,
                system_prompt=effective_system_prompt,
                tool_names=active_tool_names,
                agent_name=agent_name,
                tool_choice=tool_choice,
            )
        except litellm.ContextWindowExceededError as e:
            # PTL recovery: attempt history compression and retry
            logger.warning("Context window exceeded, attempting compression recovery")
            if _try_compression_recovery(agent_state):
                try:
                    assistant_msg = _call_llm(
                        model_name=model_name,
                        agent_state=agent_state,
                        system_prompt=effective_system_prompt,
                        tool_names=active_tool_names,
                        agent_name=agent_name,
                        tool_choice=tool_choice,
                    )
                except Exception as retry_error:
                    logger.error("LLM call failed after compression recovery: {}", retry_error)
                    return QueryResult(
                        messages=all_messages,
                        stop_reason=StopReason.CONTEXT_TOO_LONG,
                        turn_count=turn,
                        error=e,
                    )
            else:
                return QueryResult(
                    messages=all_messages,
                    stop_reason=StopReason.CONTEXT_TOO_LONG,
                    turn_count=turn,
                    error=e,
                )
        except Exception as e:
            # Unrecoverable LLM error (after ModelRegistry's own 3x retry)
            logger.error("Unrecoverable LLM error: {}", e)
            return QueryResult(
                messages=all_messages,
                stop_reason=StopReason.ERROR,
                turn_count=turn,
                error=e,
            )

        agent_state.add_message(assistant_msg)
        all_messages.append(assistant_msg)

        # --- Step 2: Check for tool calls ---
        if not assistant_msg.tool_calls or len(assistant_msg.tool_calls) == 0:
            # Check for output truncation (finish_reason == "length")
            # If truncated, inject a continue prompt so the model can finish
            if getattr(assistant_msg, "finish_reason", None) == "length":
                logger.info("Output truncated (finish_reason=length), injecting continue prompt")
                continue_msg = Message(
                    role="user",
                    content=(
                        "<system-reminder>\n"
                        "Your previous response was truncated due to output length limits. "
                        "Continue exactly where you left off.\n"
                        "</system-reminder>"
                    ),
                    agent_sender=agent_name,
                    is_meta=True,
                )
                agent_state.add_message(continue_msg)
                all_messages.append(continue_msg)
                continue

            # Detect "give up after error" pattern: previous turn had tool
            # errors but model responded with plain text instead of retrying.
            # Nudge it to self-correct (up to 2 times).
            if prev_turn_had_errors and gave_up_nudges < 2:
                gave_up_nudges += 1
                logger.info(
                    "Model gave up after tool error (nudge {}/2), injecting retry prompt",
                    gave_up_nudges,
                )
                nudge_msg = Message(
                    role="user",
                    content=(
                        "<system-reminder>\n"
                        "You described a tool error in plain text instead of fixing it. "
                        "Do NOT restate errors — fix them by calling the tool again with "
                        "corrected parameters. For example, if a file was not found, try "
                        "listing the directory first. If a file was too large, use offset "
                        "and limit parameters. Act now.\n"
                        "</system-reminder>"
                    ),
                    agent_sender=agent_name,
                    is_meta=True,
                )
                agent_state.add_message(nudge_msg)
                all_messages.append(nudge_msg)
                prev_turn_had_errors = False
                continue

            # Empty response guard: if the model returned no content AND no
            # tool calls (not truncation), it may be a transient API glitch
            # rather than an intentional end-turn. Nudge it to continue.
            if (
                not assistant_msg.content or not assistant_msg.content.strip()
            ) and empty_response_nudges < 2:
                empty_response_nudges += 1
                logger.info(
                    "Empty response (no content, no tool calls) — nudging agent to continue ({}/2)",
                    empty_response_nudges,
                )
                nudge_msg = Message(
                    role="user",
                    content=(
                        "<system-reminder>\n"
                        "Your previous response was empty (no text and no tool calls). "
                        "This was likely a transient error. Continue working on the task "
                        "from where you left off. Use tools to make progress.\n"
                        "</system-reminder>"
                    ),
                    agent_sender=agent_name,
                    is_meta=True,
                )
                agent_state.add_message(nudge_msg)
                all_messages.append(nudge_msg)
                continue

            # Before ending: check if background tasks are still running.
            # If so, wait for them and inject notifications so the agent can
            # see the results and continue working.
            if _wait_for_background_tasks(agent_state, agent_name, all_messages):
                # Notifications injected — continue the loop so the LLM sees them
                continue

            return QueryResult(
                messages=all_messages,
                stop_reason=StopReason.END_TURN,
                turn_count=turn + 1,
            )

        # --- Step 3: Execute tools ---
        try:
            tool_msgs, all_errors = _execute_tool_calls(
                tool_calls=assistant_msg.tool_calls,
                function_map=function_map,
                agent_state=agent_state,
                agent_name=agent_name,
                tool_execution_context=tool_execution_context,
                ctx_dict=ctx_dict,
            )

            for msg in tool_msgs:
                agent_state.add_message(msg)
                all_messages.append(msg)

                # Notify callback (may raise StopQuery)
                if on_tool_result is not None:
                    on_tool_result(msg.tool_name or "", msg.tool_call_id or "", msg)

        except StopQuery as e:
            return QueryResult(
                messages=all_messages,
                stop_reason=e.reason,
                turn_count=turn + 1,
            )

        # --- Step 3.5: Track if this turn had any tool errors ---
        any_errors = any(
            msg.content and ("Error" in msg.content or "failed" in msg.content.lower())
            for msg in tool_msgs
            if msg.role == "tool"
        )
        prev_turn_had_errors = any_errors
        if not any_errors:
            gave_up_nudges = 0  # reset on clean turn
            empty_response_nudges = 0

        # --- Step 4: Consecutive error circuit breaker ---
        if all_errors:
            consecutive_error_turns += 1
            if consecutive_error_turns >= 3:
                logger.warning("Circuit breaker: 3 consecutive turns with all tool errors")
                return QueryResult(
                    messages=all_messages,
                    stop_reason=StopReason.ERROR,
                    turn_count=turn + 1,
                    error=RuntimeError("All tool calls failed for 3 consecutive turns"),
                )
            # Nudge the model to retry with correct parameters
            nudge_msg = Message(
                role="user",
                content=(
                    "<system-reminder>\n"
                    f"All tool calls failed this turn ({consecutive_error_turns}/3 consecutive failures). "
                    "Review the error messages above carefully and retry with corrected parameters. "
                    "Do NOT give up — fix the tool call and try again.\n"
                    "</system-reminder>"
                ),
                agent_sender=agent_name,
                is_meta=True,
            )
            agent_state.add_message(nudge_msg)
            all_messages.append(nudge_msg)
        else:
            consecutive_error_turns = 0

    # max_turns exhausted
    logger.warning("query() reached max_turns={} for agent {}", max_turns, agent_name)
    return QueryResult(
        messages=all_messages,
        stop_reason=StopReason.MAX_TURNS,
        turn_count=max_turns,
    )
