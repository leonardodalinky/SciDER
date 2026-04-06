"""TodoWrite tool — structured task list management.

Modeled after Claude Code's TodoWriteTool. Manages a structured todo list
within the agent's session. Each task has content, activeForm, and status.

The todo list is stored in ToolContext.extra["todos"] so it persists across
tool calls within a single query() session, and is also written to
agent_state.intermediate_state for UI rendering.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext


class TodoItem(BaseModel):
    content: str = Field(
        description="Imperative form: what needs to be done (e.g., 'Run tests')",
        min_length=1,
    )
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="Task status: pending, in_progress, or completed",
    )
    activeForm: str = Field(
        description="Present continuous form shown during execution (e.g., 'Running tests')",
        min_length=1,
    )


class TodoWriteInput(BaseModel):
    todos: list[TodoItem] = Field(
        description="The complete updated todo list (replaces previous list)",
    )


class TodoTool(BaseTool):
    name = "TodoWrite"
    description = (
        "Create and manage a structured task list for the current session. "
        "Use this to track progress on complex, multi-step tasks. "
        "Always provide both content (imperative) and activeForm (present continuous) for each task."
    )
    input_schema = TodoWriteInput
    _always_read_only = True  # doesn't modify files

    prompt = (
        "# TodoWrite tool usage\n"
        "- Use proactively for tasks with 3+ steps. Skip for trivial single-step tasks.\n"
        "- Each task needs `content` (imperative: 'Fix bug'), `activeForm` (continuous: 'Fixing bug'), "
        "and `status` (pending/in_progress/completed).\n"
        "- Keep exactly ONE task as in_progress at a time.\n"
        "- Mark tasks completed IMMEDIATELY after finishing — don't batch completions.\n"
        "- Only mark completed when FULLY done. If blocked, keep as in_progress and add a new task.\n"
        "- The full todo list is replaced each call — always include all tasks, not just changes.\n"
    )

    def call(self, context: ToolContext, *, todos: list[dict]) -> str:
        # Validate items
        validated_todos = [TodoItem.model_validate(t) for t in todos]

        # Store in context for persistence across tool calls
        todo_dicts = [t.model_dump() for t in validated_todos]
        context.extra["todos"] = todo_dicts

        # Also write to intermediate_state for UI rendering
        if context.agent_state is not None and hasattr(context.agent_state, "intermediate_state"):
            # Find and update existing todo entry, or append new one
            intermediate = context.agent_state.intermediate_state
            todo_entry = {
                "node_name": "todo",
                "output": self._format_todo_display(validated_todos),
            }
            # Replace last todo entry if exists, otherwise append
            for i in range(len(intermediate) - 1, -1, -1):
                if intermediate[i].get("node_name") == "todo":
                    intermediate[i] = todo_entry
                    break
            else:
                intermediate.append(todo_entry)

        # Build response with full list so the model can reference it later
        in_progress = [t for t in validated_todos if t.status == "in_progress"]
        completed = [t for t in validated_todos if t.status == "completed"]
        pending = [t for t in validated_todos if t.status == "pending"]

        summary = (
            f"Todos updated: {len(completed)} completed, "
            f"{len(in_progress)} in progress, {len(pending)} pending."
        )
        task_list = self._format_todo_display(validated_todos)
        return f"{summary}\n\n{task_list}\n\nProceed with the current in-progress task."

    @staticmethod
    def _format_todo_display(todos: list[TodoItem]) -> str:
        """Format todo list for display in intermediate state."""
        if not todos:
            return "No tasks."
        lines = []
        status_icons = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}
        for t in todos:
            icon = status_icons.get(t.status, "[ ]")
            lines.append(f"{icon} {t.content}")
        return "\n".join(lines)
