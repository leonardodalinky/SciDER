"""PipInstallTool — install a Python package via pip."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext
from .._shell_helper import run_shell_cmd


class PipInstallInput(BaseModel):
    package: str = Field(description="Package name")
    version: str = Field(default="", description="Package version (optional)")
    venv: str = Field(default="", description="Path to virtual environment (optional)")


class PipInstallTool(BaseTool):
    name = "pip_install"
    description = "Install a Python package via pip inside the environment"
    input_schema = PipInstallInput

    def call(self, context: ToolContext, *, package: str, version: str = "", venv: str = "") -> str:
        pkg = f"{package}=={version}" if version else package
        pip = f"{venv}/bin/pip" if venv else "pip"
        return run_shell_cmd(f"{pip} install {pkg}")
