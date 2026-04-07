"""Environment management toolset — new-style tools."""

from ..registry import register_new_tool
from .check_python_import import CheckPythonImportTool
from .create_virtualenv import CreateVirtualenvTool
from .pip_install import PipInstallTool
from .pip_install_requirements import PipInstallRequirementsTool
from .write_file import WriteFileTool

register_new_tool(CreateVirtualenvTool())
register_new_tool(PipInstallTool())
register_new_tool(PipInstallRequirementsTool())
register_new_tool(CheckPythonImportTool())
register_new_tool(WriteFileTool())
