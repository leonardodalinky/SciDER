"""AgentTool — generic subagent dispatch tool.

Modeled after Claude Code's AgentTool. Instead of creating a separate tool
for each subagent, there is ONE AgentTool that dispatches to registered
agent types based on the `subagent_type` parameter.

Agent types are registered in the AgentRegistry with their compiled graph,
state class, and configuration callbacks.

Usage:
    # Register an agent type
    AgentRegistry.register(
        AgentType(
            name="paper_search",
            description="Search for academic papers and datasets",
            compiled_graph=compiled_graph,
            state_cls=PaperSearchAgentState,
            state_builder=lambda prompt, parent_state: {...},
            result_extractor=lambda result: {...},
        )
    )

    # The LLM calls the unified AgentTool:
    agent_tool(prompt="Find papers about transformers", subagent_type="paper_search")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

from loguru import logger
from pydantic import BaseModel, Field

from .base import BaseTool, ToolContext
from .registry import register_new_tool


@dataclass
class AgentType:
    """Definition of a registered agent type."""

    name: str
    description: str
    compiled_graph: Any
    state_cls: type[BaseModel]

    # Build subagent state from (prompt, parent_state) -> state kwargs dict
    state_builder: Callable[[str, Any], dict]

    # Extract results from subagent output dict -> result dict
    result_extractor: Callable[[dict], dict] | None = None

    # Optionally update parent state after subagent completes
    parent_state_updater: Callable[[Any, dict], None] | None = None


class AgentRegistry:
    """Registry of available agent types."""

    _instance: AgentRegistry | None = None
    _lock: RLock = RLock()

    def __new__(cls) -> AgentRegistry:
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
        self.agent_types: dict[str, AgentType] = {}

    @classmethod
    def instance(cls) -> AgentRegistry:
        return cls()

    @classmethod
    def register(cls, agent_type: AgentType) -> None:
        registry = cls.instance()
        if agent_type.name in registry.agent_types:
            logger.warning("Agent type '{}' already registered, replacing", agent_type.name)
        registry.agent_types[agent_type.name] = agent_type
        logger.debug("Registered agent type: {}", agent_type.name)

    @classmethod
    def get(cls, name: str) -> AgentType | None:
        return cls.instance().agent_types.get(name)

    @classmethod
    def list_types(cls) -> list[AgentType]:
        return list(cls.instance().agent_types.values())


class AgentToolInput(BaseModel):
    prompt: str = Field(
        description="Complete task description for the subagent. Be specific — include context, file paths, and what you need.",
    )
    subagent_type: str = Field(
        description="Type of agent to dispatch to. Available types listed in tool description.",
    )


class AgentTool(BaseTool):
    name = "Agent"
    description = ""  # Dynamically built from registered agent types
    input_schema = AgentToolInput
    prompt = (
        "# Agent tool usage\n"
        "- Launch a subagent for complex, self-contained tasks (paper search, coding).\n"
        "- Write a complete, self-contained prompt — the subagent has NO context from "
        "this conversation. Include file paths, specific details, and what you need.\n"
        "- Do NOT use Agent for simple tasks you can do directly (file reads, searches).\n"
        "- The subagent executes autonomously and returns results when done.\n"
    )

    @property
    def _dynamic_description(self) -> str:
        """Build description from registered agent types."""
        types = AgentRegistry.list_types()
        if not types:
            return "Launch a subagent to handle a task. No agent types are currently registered."

        type_list = "\n".join(f"  - {t.name}: {t.description}" for t in types)
        return (
            "Launch a specialized subagent to handle a complex task autonomously.\n\n"
            "Available agent types:\n"
            f"{type_list}\n\n"
            "Usage notes:\n"
            "- Write a complete, self-contained prompt — the agent has no context from this conversation\n"
            "- Include file paths, specific details, and what you need\n"
            "- The agent will execute and return results"
        )

    def to_json_schema(self) -> dict:
        """Override to inject dynamic description."""
        schema = super().to_json_schema()
        schema["function"]["description"] = self._dynamic_description
        # Also inject available types into subagent_type enum
        types = AgentRegistry.list_types()
        if types:
            schema["function"]["parameters"]["properties"]["subagent_type"]["enum"] = [
                t.name for t in types
            ]
        return schema

    def call(self, context: ToolContext, *, prompt: str, subagent_type: str) -> str:
        agent_type = AgentRegistry.get(subagent_type)
        if agent_type is None:
            available = [t.name for t in AgentRegistry.list_types()]
            return f"Error: Unknown agent type '{subagent_type}'. " f"Available types: {available}"

        logger.info("AgentTool dispatching to '{}': {}", subagent_type, prompt[:100])

        try:
            # Build subagent state from prompt + parent state
            parent_state = context.agent_state
            state_kwargs = agent_type.state_builder(prompt, parent_state)

            # Create and invoke subagent
            subagent_state = agent_type.state_cls(**state_kwargs)
            result_dict = agent_type.compiled_graph.invoke(subagent_state)

            # Extract results
            extractor = agent_type.result_extractor or (lambda r: r)
            extracted = extractor(result_dict)

            # Optionally update parent state
            if agent_type.parent_state_updater and parent_state is not None:
                try:
                    agent_type.parent_state_updater(parent_state, extracted)
                except Exception as e:
                    logger.warning("parent_state_updater failed for '{}': {}", subagent_type, e)

            logger.info("AgentTool '{}' completed successfully", subagent_type)
            return json.dumps(extracted, ensure_ascii=False, default=str)

        except Exception as e:
            logger.exception("AgentTool '{}' failed", subagent_type)
            return json.dumps({"error": f"Agent '{subagent_type}' failed: {e}"})


# Register the singleton AgentTool
register_new_tool(AgentTool())
