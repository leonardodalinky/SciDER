"""CreateVirtualenvTool — create a Python virtual environment."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext
from .._shell_helper import run_shell_cmd


class CreateVirtualenvInput(BaseModel):
    path: str = Field(description="Directory of the venv")


class CreateVirtualenvTool(BaseTool):
    name = "create_virtualenv"
    description = "Create a Python virtual environment using python3 -m venv"
    input_schema = CreateVirtualenvInput

    def call(self, context: ToolContext, *, path: str) -> str:
        return run_shell_cmd(f"python3 -m venv {path}")
