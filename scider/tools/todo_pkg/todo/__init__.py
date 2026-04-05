"""TodoTool — record a todo item."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext


class TodoInput(BaseModel):
    todo: str = Field(description="The todo item to record.")


class TodoTool(BaseTool):
    name = "Todo"
    description = "Record a todo item and echo it back."
    input_schema = TodoInput

    def call(self, context: ToolContext, *, todo: str) -> str:
        return todo
