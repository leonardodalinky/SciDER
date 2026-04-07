"""CloneRepoTool — clone a GitHub repository."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext


class CloneRepoInput(BaseModel):
    repo_url: str = Field(description="HTTP(S) URL of the GitHub repository to clone.")
    dest_dir: str = Field(description="Local directory path where the repository will be cloned.")


class CloneRepoTool(BaseTool):
    name = "clone_repo"
    description = "Clone a GitHub repository to a target local directory."
    input_schema = CloneRepoInput

    def call(self, context: ToolContext, *, repo_url: str, dest_dir: str) -> str:
        try:
            dest_path = Path(os.path.expandvars(dest_dir)).expanduser()
            dest_path.mkdir(parents=True, exist_ok=True)

            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
            destination = dest_path / repo_name

            if destination.exists():
                shutil.rmtree(destination)

            result = os.system(f"git clone {repo_url} {destination}")

            if result != 0:
                return f"Error: Failed to clone repository from {repo_url}"

            return f"Repository cloned to: {destination}"
        except Exception as e:
            return f"Error cloning repository: {e}"
