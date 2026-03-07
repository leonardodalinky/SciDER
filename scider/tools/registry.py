from __future__ import annotations

from threading import RLock
from typing import Callable

from pydantic import BaseModel


class ToolRegistry:
    _instance: ToolRegistry | None = None
    _lock: RLock = RLock()

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.tools: dict[str, Tool] = {}
        self._toolsets_desc: dict[str, str] = {}

    @classmethod
    def instance(cls) -> ToolRegistry:
        return cls()

    @classmethod
    def get_toolset(cls, toolset: str) -> dict[str, "Tool"]:
        return {
            tool_name: tool
            for tool_name, tool in cls.instance().tools.items()
            if tool.toolset == toolset
        }

    @classmethod
    def get_toolsets_desc(cls, toolsets: list[str]) -> dict[str, str]:
        ret = {}
        for toolset in toolsets:
            if toolset not in cls.instance()._toolsets_desc:
                raise ValueError(f"Toolset {toolset} is not registered")
            ret[toolset] = cls.instance()._toolsets_desc[toolset]
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
        if (func_name := json_schema["function"]["name"]) in ToolRegistry.instance().tools:
            raise ValueError(f"Tool {func_name} is already registered")
        else:
            ToolRegistry.instance().tools[func_name] = Tool(
                toolset=toolset,
                json_schema=json_schema,
                name=func_name,
                func=func,
            )
            return func

    return decorator


def register_toolset_desc(toolset: str, desc: str):
    ToolRegistry.instance()._toolsets_desc[toolset] = desc
