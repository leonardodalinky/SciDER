"""TaskOutput tool — retrieve output from background tasks.

Modeled after Claude Code's TaskOutputTool.
Supports both blocking (wait for completion) and non-blocking reads.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..base import BaseTool, ToolContext


class TaskOutputInput(BaseModel):
    task_id: str = Field(description="The task ID returned by Bash(run_in_background=true)")
    block: bool = Field(
        default=False,
        description="If true, wait for the task to complete before returning output",
    )
    timeout: int = Field(
        default=30,
        description="Max seconds to wait when block=true (default 30, max 43200 i.e. 12 hours)",
    )
    tail: int | None = Field(
        default=None,
        description="If set, return only the last N lines of output",
    )


class TaskOutputTool(BaseTool):
    name = "TaskOutput"
    description = (
        "Retrieve status and output from a background task. "
        "Use after running Bash(run_in_background=true) to check results. "
        "Set block=true to wait for completion."
    )
    input_schema = TaskOutputInput
    _always_read_only = True

    def call(
        self,
        context: ToolContext,
        *,
        task_id: str,
        block: bool = False,
        timeout: int = 30,
        tail: int | None = None,
    ) -> str:
        from scider.core.task import TaskManager, TaskStatus

        task = TaskManager.get(task_id)
        if not task:
            return f"Error: Task '{task_id}' not found"

        # Blocking wait
        if block and task.status == TaskStatus.RUNNING:
            task = TaskManager.wait_for_task(task_id, timeout=float(timeout))
            if not task:
                return f"Error: Task '{task_id}' not found after wait"
            if task.status == TaskStatus.RUNNING:
                # Still running after timeout — return partial output
                output = TaskManager.read_output(task_id, tail=tail)
                return (
                    f"Task {task_id} is still running (waited {timeout}s).\n"
                    f"Status: {task.status.value}\n"
                    f"Partial output:\n{output}"
                )

        # Read output
        output = TaskManager.read_output(task_id, tail=tail)

        elapsed = ""
        if task.end_time and task.start_time:
            elapsed = f", elapsed: {task.end_time - task.start_time:.1f}s"

        header = (
            f"Task: {task_id}\n"
            f"Status: {task.status.value}"
            f"{f', exit code: {task.exit_code}' if task.exit_code is not None else ''}"
            f"{elapsed}\n"
            f"Command: {task.command}\n"
            f"---\n"
        )
        return header + output
