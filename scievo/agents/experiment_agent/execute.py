import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from functional import seq
from loguru import logger

from scievo import history_compression
from scievo.core import constant
from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.core.utils import wrap_dict_to_toon, wrap_text_with_block
from scievo.prompts import PROMPTS
from scievo.rbank.subgraph import mem_extraction, mem_retrieval
from scievo.tools import Tool, ToolRegistry

from .state import ExperimentAgentState

LLM_NAME = "execute"
AGENT_NAME = "experiment_executor"


def gateway_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    # NOTE: this node does nothing, it's just a placeholder for the conditional edges
    # Check `gateway_conditional` for the actual logic
    logger.trace("gateway_node of Agent {}", AGENT_NAME)
    return agent_state


def gateway_conditional(agent_state: ExperimentAgentState) -> str:

    last_msg = agent_state.patched_history[-1]
    if (tool_calls := last_msg.tool_calls) and len(tool_calls) > 0:
        return "tool_calling"

    match last_msg.role:
        case "tool":
            return "tool_calling"
        case "assistant":
            return "replanner"
        case _:
            raise ValueError(f"Unknown message role: {last_msg.role}")


def tool_calling_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    logger.debug("tool_calling_node of Agent experiment_executor")
    last_msg = agent_state.patched_history[-1]

    if not getattr(last_msg, "tool_calls", None):
        return agent_state

    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        logger.debug("No tool calls in last LLM message.")
        return agent_state

    tools: dict[str, Tool] = {}
    for toolset in agent_state.toolsets:
        tools.update(ToolRegistry.get_toolset(toolset))

    function_map = {tool.name: tool.func for tool in tools.values()}

    for tool_call in last_msg.tool_calls:
        tool_name = tool_call.function.name
        if tool_name not in function_map:
            error_msg = f"Tool {tool_name} not found."
            logger.error(error_msg)
            agent_state.add_message(
                Message(
                    role="tool",
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    content=error_msg,
                )
            )
            continue

        import json

        try:
            args = json.loads(tool_call.function.arguments)
            assert isinstance(args, dict)
        except Exception as e:
            error_msg = f"Invalid tool arguments: {e}"
            logger.error(error_msg)
            agent_state.add_message(
                Message(
                    role="tool",
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    content=error_msg,
                )
            )
            continue

        try:
            func = function_map[tool_name]

            result = func(**args)

            agent_state.add_message(
                Message(
                    role="tool",
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    content=str(result),
                )
            )

        except Exception as e:
            error_msg = f"Tool {tool_name} execution failed: {e}"
            logger.error(error_msg)
            agent_state.add_message(
                Message(
                    role="tool",
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    content=error_msg,
                )
            )

    return agent_state
