from typing import Callable

from pydantic import BaseModel


class ToolRegistry:
    _instance = None

    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self._toolsets_desc: dict[str, str] = {}

    @staticmethod
    def get_instance() -> "ToolRegistry":
        if ToolRegistry._instance is None:
            ToolRegistry._instance = ToolRegistry()
        return ToolRegistry._instance

    @staticmethod
    def get_toolset(toolset: str) -> dict[str, "Tool"]:
        return {
            tool_name: tool
            for tool_name, tool in ToolRegistry.get_instance().tools.items()
            if tool.toolset == toolset
        }

    @staticmethod
    def get_toolsets_desc(toolsets: list[str]) -> dict[str, str]:
        ret = {}
        for toolset in toolsets:
            if toolset not in ToolRegistry.get_instance()._toolsets_desc:
                raise ValueError(f"Toolset {toolset} is not registered")
            ret[toolset] = ToolRegistry.get_instance()._toolsets_desc[toolset]
        return ret


class Tool(BaseModel):
    # The name of the toolset this tool belongs to
    toolset: str
    # The name of the tool
    json_schema: dict
    # function name
    name: str
    # The function to be called when the tool is executed
    func: Callable


def register_tool(toolset: str, json_schema: dict):
    """
    A decorator to register a tool to the tool registry
    """

    def decorator(func):
        if (func_name := json_schema["function"]["name"]) in ToolRegistry.get_instance().tools:
            raise ValueError(f"Tool {func_name} is already registered")
        else:
            ToolRegistry.get_instance().tools[func_name] = Tool(
                toolset=toolset,
                json_schema=json_schema,
                name=func_name,
                func=func,
            )
            return func

    return decorator


def register_toolset_desc(toolset: str, desc: str):
    ToolRegistry.get_instance()._toolsets_desc[toolset] = desc


register_toolset_desc("dummy", "Dummy toolset.")
register_toolset_desc("noop", "No-op toolset.")
register_toolset_desc("fs", "File system toolset.")
register_toolset_desc("state", "State management toolset.")
