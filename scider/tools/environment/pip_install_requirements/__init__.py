"""PipInstallRequirementsTool — install from requirements.txt."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext
from .._shell_helper import run_shell_cmd


class PipInstallRequirementsInput(BaseModel):
    requirements_path: str = Field(description="Path to requirements.txt file")
    venv: str = Field(default="", description="Path to virtual environment (optional)")


class PipInstallRequirementsTool(BaseTool):
    name = "pip_install_requirements"
    description = "Install dependencies from a requirements.txt file"
    input_schema = PipInstallRequirementsInput

    def call(self, context: ToolContext, *, requirements_path: str, venv: str = "") -> str:
        pip = f"{venv}/bin/pip" if venv else "pip"
        return run_shell_cmd(f"{pip} install -r {requirements_path}")
