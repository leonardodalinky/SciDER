"""ListRepoFilesTool — recursively list files in a repository."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext


class ListRepoFilesInput(BaseModel):
    repo_dir: str = Field(description="Path to the local repository folder.")


class ListRepoFilesTool(BaseTool):
    name = "list_repo_files"
    description = "Recursively list all files inside a cloned GitHub repository."
    input_schema = ListRepoFilesInput
    _always_read_only = True

    def call(self, context: ToolContext, *, repo_dir: str) -> str:
        try:
            repo_path = Path(os.path.expandvars(repo_dir)).expanduser()
            if not repo_path.exists():
                return f"Error: Repository directory '{repo_dir}' does not exist"

            files = [str(p) for p in repo_path.rglob("*") if p.is_file()]
            if not files:
                return "No files found inside repository."

            return "\n".join(files)
        except Exception as e:
            return f"Error listing repository files: {e}"
