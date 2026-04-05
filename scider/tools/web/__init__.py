"""Web toolset."""

from ..registry import register_new_tool
from .get_url_content import WebFetchTool
from .web_search import WebSearchTool

register_new_tool(WebSearchTool())
register_new_tool(WebFetchTool())
