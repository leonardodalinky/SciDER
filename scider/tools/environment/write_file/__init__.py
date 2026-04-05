"""WriteFileTool — write content to a file."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext


class WriteFileInput(BaseModel):
    path: str = Field(description="File path to write to")
    content: str = Field(description="Content to write to the file")


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file path"
    input_schema = WriteFileInput

    def call(self, context: ToolContext, *, path: str, content: str) -> str:
        Path(path).write_text(content)
        return f"File written to {path}"
