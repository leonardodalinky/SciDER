"""File system tools."""

from ..registry import register_new_tool
from .edit_file import FileEditTool
from .glob import GlobTool
from .grep import GrepTool
from .read_file import ReadFileTool
from .save_file import SaveFileTool

register_new_tool(ReadFileTool())
register_new_tool(FileEditTool())
register_new_tool(SaveFileTool())
register_new_tool(GlobTool())
register_new_tool(GrepTool())
