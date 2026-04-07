"""CheckPythonImportTool — check whether a Python module can be imported."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext
from .._shell_helper import run_shell_cmd


class CheckPythonImportInput(BaseModel):
    module: str = Field(description="Python module name to check")
    python: str = Field(default="python3", description="Python interpreter to use")


class CheckPythonImportTool(BaseTool):
    name = "check_python_import"
    description = "Check whether a Python module can be imported"
    input_schema = CheckPythonImportInput
    _always_read_only = True

    def call(self, context: ToolContext, *, module: str, python: str = "python3") -> str:
        return run_shell_cmd(f"{python} -c 'import {module}; print(\"OK\")'")
