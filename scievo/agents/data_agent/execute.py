"""
Agent for data understanding and processing
"""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from functional import seq
from langgraph.graph import END
from loguru import logger

from scievo.core import constant
from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.core.utils import wrap_dict_to_toon, wrap_text_with_block
from scievo.prompts import PROMPTS
from scievo.rbank.subgraph import mem_extraction, mem_retrieval
from scievo.tools import Tool, ToolRegistry

from .state import DataAgentState

if TYPE_CHECKING:
    from scievo.rbank.memo import Memo

LLM_NAME = "data"
AGENT_NAME = "data"
MEM_EXTRACTION_CONTEXT_WINDOW = int(os.getenv("MEM_EXTRACTION_CONTEXT_WINDOW", 16))
MEM_EXTRACTION_ROUND_FREQ = int(os.getenv("MEM_EXTRACTION_ROUND_FREQ", 8))


def gateway_node(agent_state: DataAgentState) -> DataAgentState:
    # NOTE: this node does nothing, it's just a placeholder for the conditional edges
    # Check `gateway_conditional` for the actual logic
    logger.trace("gateway_node of Agent {}", AGENT_NAME)
    return agent_state


def gateway_conditional(agent_state: DataAgentState) -> str:
    agent_state = agent_state

    if (
        not agent_state.skip_mem_extraction
        and agent_state.round > 0
        and agent_state.round % MEM_EXTRACTION_ROUND_FREQ == 0
    ):
        return "mem_extraction"

    last_msg = agent_state.history[-1]
    if (tool_calls := last_msg.tool_calls) and len(tool_calls) > 0:
        return "tool_calling"

    match last_msg.role:
        case "user" | "tool":
            return "llm_chat"
        case "assistant":
            return "replanner"
        case _:
            raise ValueError(f"Unknown message role: {last_msg.role}")


mem_retrieval_subgraph = mem_retrieval.build()
mem_retrieval_subgraph_compiled = mem_retrieval_subgraph.compile()


def _memos_to_markdown(memos: list["Memo"]) -> str:
    ret = ""
    if len(memos) == 0:
        return "No memory retrieved."
    for i, memo in enumerate(memos):
        ret += f"# Memo {i + 1}\n\n{memo.to_markdown()}\n\n"
    return ret


def llm_chat_node(agent_state: DataAgentState) -> DataAgentState:
    logger.debug("llm_chat_node of Agent {}", AGENT_NAME)
    agent_state = agent_state
    agent_state.round += 1
    agent_state.skip_mem_extraction = False

    selected_state = {
        "working_dir": agent_state.local_env.working_dir,
        "toolsets": agent_state.toolsets,
    }

    # retrieve memos
    try:
        res = mem_retrieval_subgraph_compiled.invoke(
            mem_retrieval.MemRetrievalState(
                input_msgs=agent_state.history,
                mem_dirs=[agent_state.sess_dir / f"mem_{AGENT_NAME}"],  # TODO: more mem dirs
            )
        )
    except Exception as e:
        logger.exception("mem_retrieval_error")
        res = {"output_error": f"mem_retrieval_error with exception {e}"}

    if err := res.get("output_error", None):
        memory_text = err
    else:
        memos: list[Memo] = res.get("output_memos", [])
        memory_text = _memos_to_markdown(memos)

    # update system prompt
    system_prompt = PROMPTS.data.system_prompt.render(
        state_text=wrap_dict_to_toon(selected_state),
        toolsets_desc=ToolRegistry.get_toolsets_desc(["fs"]),
        memory_text=wrap_text_with_block(memory_text, "markdown"),
        current_plan=(
            agent_state.remaining_plans[0] if len(agent_state.remaining_plans) > 0 else None
        ),
    )

    tools: dict[str, Tool] = {}
    for toolset in agent_state.toolsets:
        tools.update(ToolRegistry.get_toolset(toolset))
    tools.update(ToolRegistry.get_toolset("todo"))
    tools.update(ToolRegistry.get_toolset("state"))
    msg = ModelRegistry.completion(
        LLM_NAME,
        seq(agent_state.history).filter_not(lambda msg: msg.hidden).to_list(),
        system_prompt,
        agent_sender=AGENT_NAME,
        tools=[tool.name for tool in tools.values()],
    ).with_log()
    agent_state.history.append(msg)

    return agent_state


def tool_calling_node(agent_state: DataAgentState) -> DataAgentState:
    """Execute tool calls from the last message and update the graph state"""
    logger.debug("tool_calling_node of Agent {}", AGENT_NAME)
    agent_state = agent_state
    # Get the last message which contains tool calls
    last_msg = agent_state.history[-1]

    if not last_msg.tool_calls:
        raise ValueError("No tool calls found in the last message")

    # Create a function map for available tools
    tools: dict[str, Tool] = {}
    for toolset in agent_state.toolsets:
        tools.update(ToolRegistry.get_toolset(toolset))
    tools.update(ToolRegistry.get_toolset("todo"))
    tools.update(ToolRegistry.get_toolset("state"))

    function_map = {tool.name: tool.func for tool in tools.values()}

    # Execute each tool call
    for tool_call in last_msg.tool_calls:
        tool_name = tool_call.function.name

        # Check if tool exists in function map
        if tool_name not in function_map:
            error_msg = f"Tool {tool_name} not found"
            tool_response = {
                "role": "tool",
                "tool_name": tool_name,
                "tool_call_id": tool_call.id,
                "content": error_msg,
            }
            agent_state.history.append(Message(**tool_response).with_log())
            continue

        # Parse tool arguments
        try:
            args = json.loads(tool_call.function.arguments)
            assert isinstance(args, dict)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in tool arguments: {e}"
            tool_response = {
                "role": "tool",
                "tool_name": tool_name,
                "tool_call_id": tool_call.id,
                "content": error_msg,
            }
            agent_state.history.append(Message(**tool_response).with_log())
            continue
        except AssertionError as e:
            error_msg = f"Invalid tool arguments: {e}"
            tool_response = {
                "role": "tool",
                "tool_name": tool_name,
                "tool_call_id": tool_call.id,
                "content": error_msg,
            }
            agent_state.history.append(Message(**tool_response).with_log())
            continue

        # Execute the tool
        try:
            # Pass the graph state to the tool function
            func = function_map[tool_name]

            # Check if function expects agent_state parameter
            import inspect

            sig = inspect.signature(func)
            if constant.__AGENT_STATE_NAME__ in sig.parameters:
                args.update({constant.__AGENT_STATE_NAME__: agent_state})
            if constant.__CTX_NAME__ in sig.parameters:
                args.update({constant.__CTX_NAME__: {"current_agent": AGENT_NAME}})

            # Execute the tool in the agent's local environment
            with agent_state.local_env:
                result = func(**args)

            # Create tool response message
            tool_response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": tool_name,
                "content": str(result),  # Ensure result is string
            }
            agent_state.history.append(Message(**tool_response).with_log())

        except Exception as e:
            error_msg = f"Tool {tool_name} execution failed: {e}"
            tool_response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": tool_name,
                "content": error_msg,
            }
            agent_state.history.append(Message(**tool_response).with_log())

    return agent_state


mem_extraction_subgraph = mem_extraction.build()
mem_extraction_subgraph_compiled = mem_extraction_subgraph.compile()


def mem_extraction_node(agent_state: DataAgentState) -> DataAgentState:
    logger.debug("mem_extraction_node of Agent {}", AGENT_NAME)
    agent_state = agent_state
    agent_state.skip_mem_extraction = True
    context_window = agent_state.history[-MEM_EXTRACTION_CONTEXT_WINDOW:]
    logger.info("Agent {} begins to Memory Extraction", AGENT_NAME)
    try:
        res = mem_extraction_subgraph_compiled.invoke(
            mem_extraction.MemExtractionState(
                save_dir=Path(agent_state.sess_dir) / f"mem_{AGENT_NAME}",
                input_msgs=context_window,
            )
        )
    except Exception as e:
        agent_state.history.append(
            Message(
                role="assistant",
                content=f"mem_extraction_error: {e}",
                agent_sender=AGENT_NAME,
            ).with_log()
        )
        return agent_state

    err = res.get("output_error", None)
    if err:
        agent_state.history.append(
            Message(
                role="assistant",
                content=f"mem_extraction_error: {err}",
                agent_sender=AGENT_NAME,
            ).with_log()
        )
    return agent_state
