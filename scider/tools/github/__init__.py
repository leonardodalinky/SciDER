"""GitHub toolset — new-style tools."""

from ..registry import register_new_tool
from .clone_repo import CloneRepoTool
from .get_file_content import GetFileContentTool
from .list_repo_files import ListRepoFilesTool
from .read_readme import ReadReadmeTool

register_new_tool(CloneRepoTool())
register_new_tool(ReadReadmeTool())
register_new_tool(ListRepoFilesTool())
register_new_tool(GetFileContentTool())
