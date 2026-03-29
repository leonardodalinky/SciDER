"""Wrapper that exposes a compiled LangGraph subagent as a tool callable by the LLM."""

from __future__ import annotations

import inspect
import json
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger
from pydantic import BaseModel

from scider.core import constant

from .registry import Tool, ToolRegistry, register_toolset_desc

register_toolset_desc("subagents", "Invoke specialized sub-agents for complex tasks.")


@dataclass
class ToolParam:
    """Describes one parameter the LLM can pass to the subagent-tool."""

    json_type: str
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None
    # Maps to this subagent state field. Defaults to the param name in the dict key.
    state_field: str | None = None


class SubagentToolWrapper:
    """Wraps a compiled LangGraph subagent so it can be used as a tool.

    Usage::

        wrapper = SubagentToolWrapper(
            name="run_paper_search_agent",
            toolset="subagents",
            description="Search for academic papers ...",
            compiled_graph=paper_graph_compiled,
            state_cls=PaperSearchAgentState,
            tool_params={"user_query": ToolParam(json_type="string", ...)},
            parent_state_injector=lambda ps, kw: {"data_summary": ps.data_desc},
            result_extractor=lambda r: {"papers": r.get("papers", [])},
        )
        wrapper.register()
    """

    def __init__(
        self,
        name: str,
        toolset: str,
        description: str,
        compiled_graph: Any,
        state_cls: type[BaseModel],
        tool_params: dict[str, ToolParam],
        state_defaults: dict[str, Any] | None = None,
        parent_state_injector: Callable[[Any, dict], dict] | None = None,
        result_extractor: Callable[[dict], dict] | None = None,
        parent_state_updater: Callable[[Any, dict], None] | None = None,
    ):
        self.name = name
        self.toolset = toolset
        self.description = description
        self.compiled_graph = compiled_graph
        self.state_cls = state_cls
        self.tool_params = tool_params
        self.state_defaults = state_defaults or {}
        self.parent_state_injector = parent_state_injector
        self.result_extractor = result_extractor or (lambda r: r)
        self.parent_state_updater = parent_state_updater

        self._json_schema = self._build_json_schema()
        self._tool_func = self._make_tool_func()

    def _build_json_schema(self) -> dict:
        properties: dict[str, dict] = {}
        required: list[str] = []

        for param_name, param in self.tool_params.items():
            prop: dict[str, Any] = {
                "type": param.json_type,
                "description": param.description,
            }
            if param.enum is not None:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param_name] = prop
            if param.required:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _make_tool_func(self) -> Callable:
        """Create a callable whose signature includes ``agent_state`` so
        ``tool_calling_node`` auto-injects the parent state."""

        wrapper = self  # capture for closure

        # We need the function to have `agent_state` as a real parameter name
        # so that `inspect.signature` in tool_calling_node detects it.
        # Using a thin wrapper class with __call__ and custom __signature__.
        class _SubagentToolCallable:
            def __call__(self, agent_state: Any = None, **kwargs: Any) -> str:
                return wrapper._invoke(agent_state, kwargs)

        fn = _SubagentToolCallable()
        # Build a proper signature so inspect.signature(fn) shows agent_state + tool_params
        params = [
            inspect.Parameter(
                constant.__AGENT_STATE_NAME__,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
            ),
        ]
        for param_name, tp in wrapper.tool_params.items():
            params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=tp.default if tp.default is not None else inspect.Parameter.empty,
                )
            )
        fn.__signature__ = inspect.Signature(params)
        fn.__name__ = self.name
        fn.__qualname__ = self.name
        return fn

    def _invoke(self, parent_state: Any, kwargs: dict) -> str:
        """Run the subagent graph and return JSON results."""
        logger.info(f"Subagent tool '{self.name}' invoked with params: {list(kwargs.keys())}")

        try:
            # Build state kwargs from LLM args, mapping via state_field if set
            state_kwargs: dict[str, Any] = dict(self.state_defaults)
            for param_name, value in kwargs.items():
                tp = self.tool_params.get(param_name)
                field_name = (tp.state_field or param_name) if tp else param_name
                state_kwargs[field_name] = value

            # Inject fields from parent state (workspace, data_summary, etc.)
            if self.parent_state_injector and parent_state is not None:
                injected = self.parent_state_injector(parent_state, kwargs)
                state_kwargs.update(injected)

            # Construct and invoke subagent
            subagent_state = self.state_cls(**state_kwargs)
            result_dict = self.compiled_graph.invoke(subagent_state)

            # Extract results
            extracted = self.result_extractor(result_dict)

            # Optionally update parent state fields
            if self.parent_state_updater and parent_state is not None:
                try:
                    self.parent_state_updater(parent_state, extracted)
                except Exception as e:
                    logger.warning(f"parent_state_updater failed for '{self.name}': {e}")

            logger.info(f"Subagent tool '{self.name}' completed successfully")
            return json.dumps(extracted, ensure_ascii=False, default=str)

        except Exception as e:
            logger.exception(f"Subagent tool '{self.name}' failed")
            return json.dumps({"error": f"Subagent '{self.name}' failed: {e}"})

    def register(self) -> None:
        """Register this subagent as a tool in the global ToolRegistry."""
        registry = ToolRegistry.instance()
        if self.name in registry.tools:
            logger.warning(f"Tool '{self.name}' already registered, skipping")
            return
        registry.tools[self.name] = Tool(
            toolset=self.toolset,
            json_schema=self._json_schema,
            name=self.name,
            func=self._tool_func,
        )
        logger.debug(f"Registered subagent tool: {self.name} (toolset={self.toolset})")
