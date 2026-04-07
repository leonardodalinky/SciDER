"""SaveFileTool — save content to a file."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext


class SaveFileInput(BaseModel):
    path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")


class SaveFileTool(BaseTool):
    name = "FileWrite"
    description = "Save the given content to a file path (overwrites existing file)."
    input_schema = SaveFileInput
    prompt = (
        "# FileWrite tool usage\n"
        "- This tool overwrites the existing file. If editing an existing file, "
        "use FileEdit instead — it only sends the diff.\n"
        "- Use FileWrite only to create new files or for complete rewrites.\n"
        "- You MUST read an existing file with Read before overwriting it.\n"
    )

    def call(self, context: ToolContext, *, path: str, content: str) -> str:
        path = os.path.expandvars(os.path.expanduser(path))
        dir_path = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(dir_path):
            return f"Error: Directory '{dir_path}' does not exist. Create it first with Bash(command=\"mkdir -p {dir_path}\")."
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Saved {len(content)} characters to '{path}'"
        except Exception as e:
            return f"Error saving file '{path}': {e}"
