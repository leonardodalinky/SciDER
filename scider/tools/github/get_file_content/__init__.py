"""GetFileContentTool — retrieve file content from a repository."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext


class GetFileContentInput(BaseModel):
    repo_dir: str = Field(description="Local path to the repository.")
    relative_path: str = Field(description="Relative path of the file inside the repository.")


class GetFileContentTool(BaseTool):
    name = "get_file_content"
    description = "Retrieve the content of a file inside a cloned GitHub repository."
    input_schema = GetFileContentInput
    _always_read_only = True

    def call(self, context: ToolContext, *, repo_dir: str, relative_path: str) -> str:
        try:
            repo_path = Path(os.path.expandvars(repo_dir)).expanduser()
            file_path = repo_path / relative_path

            if not file_path.exists():
                return f"Error: File '{relative_path}' does not exist in repository"

            return file_path.read_text(errors="ignore")
        except Exception as e:
            return f"Error reading file: {e}"
