from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING, Callable

from loguru import logger
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from .base import BaseTool


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

    @classmethod
    def instance(cls) -> ToolRegistry:
        return cls()

    @classmethod
    def get_tools_by_names(cls, names: list[str]) -> dict[str, "Tool"]:
        """Get tools by their names. Unknown names are silently skipped."""
        registry = cls.instance()
        return {name: registry.tools[name] for name in names if name in registry.tools}

    @classmethod
    def get_all_tools(cls) -> dict[str, "Tool"]:
        """Get all registered tools."""
        return dict(cls.instance().tools)


class Tool(BaseModel):
    json_schema: dict
    name: str
    func: Callable
    # If set, json_schema is regenerated from this source on each access.
    # Used by AgentTool whose schema depends on dynamically registered agent types.
    _dynamic_schema_source: "BaseTool | None" = None

    def get_json_schema(self) -> dict:
        """Get the tool's JSON schema, regenerating if dynamic."""
        if self._dynamic_schema_source is not None:
            return self._dynamic_schema_source.to_json_schema()
        return self.json_schema


def register_new_tool(tool_instance: "BaseTool") -> None:
    """Register a new-style BaseTool by wrapping it into a Tool object.

    The wrapper handles ToolContext injection and Pydantic input validation,
    so downstream code (query.py, llms.py) sees only Tool objects.
    """
    from .base import ToolContext

    name = tool_instance.name
    registry = ToolRegistry.instance()

    if name in registry.tools:
        raise ValueError(f"Tool {name} is already registered")

    json_schema = tool_instance.to_json_schema()

    def _wrapper(**kwargs) -> str:
        context = kwargs.pop("__tool_context__", ToolContext(agent_name="unknown"))

        try:
            validated = tool_instance.validate_input(kwargs)
        except ValidationError as e:
            return f"Input validation error for tool {name}: {e}"

        return tool_instance.call(context, **validated)

    _wrapper.__is_new_style_tool__ = True  # type: ignore[attr-defined]

    # If tool has dynamic schema (e.g. AgentTool), store reference for lazy regeneration
    has_dynamic_schema = hasattr(tool_instance, "_dynamic_description")

    tool_obj = Tool(
        json_schema=json_schema,
        name=name,
        func=_wrapper,
    )
    if has_dynamic_schema:
        tool_obj._dynamic_schema_source = tool_instance
    registry.tools[name] = tool_obj

    # Store BaseTool instance for prompt retrieval etc.
    if not hasattr(registry, "_base_tools"):
        registry._base_tools: dict[str, BaseTool] = {}
    registry._base_tools[name] = tool_instance

    logger.debug("Registered new-style tool: {}", name)


def get_tool_prompts(tool_names: list[str]) -> list[str]:
    """Get tool-specific prompts for the given tool names."""
    base_tools: dict = getattr(ToolRegistry.instance(), "_base_tools", {})
    return [
        base_tools[name].prompt
        for name in tool_names
        if name in base_tools and base_tools[name].prompt is not None
    ]
