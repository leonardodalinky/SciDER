from pydantic import BaseModel


class ToolRegistry:
    _instance = None

    def __init__(self):
        self.tools: dict[str, Tool] = {}

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


class Tool(BaseModel):
    # The name of the toolset this tool belongs to
    toolset: str
    # The name of the tool
    json_schema: dict


def register_tool(toolset: str, json_schema: dict):
    """
    A decorator to register a tool to the tool registry
    """

    def decorator(func):
        if (func_name := func.__name__) in ToolRegistry.get_instance().tools:
            raise ValueError(f"Tool {func_name} is already registered")
        else:
            ToolRegistry.get_instance().tools[func_name] = Tool(
                toolset=toolset, json_schema=json_schema
            )
            return func

    return decorator
