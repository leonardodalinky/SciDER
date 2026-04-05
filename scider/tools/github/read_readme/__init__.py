"""ReadReadmeTool — read README.md from a repository."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext


class ReadReadmeInput(BaseModel):
    repo_dir: str = Field(description="Local directory where repository was cloned.")


class ReadReadmeTool(BaseTool):
    name = "read_readme"
    description = "Read README.md from a repository directory."
    input_schema = ReadReadmeInput
    _always_read_only = True

    def call(self, context: ToolContext, *, repo_dir: str) -> str:
        try:
            repo_path = Path(os.path.expandvars(repo_dir)).expanduser()
            if not repo_path.exists():
                return f"Error: Repository directory '{repo_dir}' does not exist"

            candidates = ["README.md", "readme.md", "Readme.md", "README.MD"]
            for filename in candidates:
                file_path = repo_path / filename
                if file_path.exists():
                    return file_path.read_text(errors="ignore")

            return "No README.md file found in the repository."
        except Exception as e:
            return f"Error reading README: {e}"
