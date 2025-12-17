"""
Summary Subagent - handles generating comprehensive experiment summaries
"""

import inspect
import json

from loguru import logger

from scievo.core import constant
from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.core.utils import wrap_dict_to_toon
from scievo.prompts import PROMPTS
from scievo.tools import Tool, ToolRegistry

from .state import SummaryAgentState

LLM_NAME = "execute_summary"
AGENT_NAME = "experiment_summary"

BUILTIN_TOOLSETS = [
    "state",
    "fs",  # Filesystem tools for reading files and listing directories
]
ALLOWED_TOOLSETS = [
    "history",
]


def gateway_node(agent_state: SummaryAgentState) -> SummaryAgentState:
    """Gateway node - placeholder for conditional routing logic"""
    logger.trace("gateway_node of Agent {}", AGENT_NAME)
    return agent_state


def gateway_conditional(agent_state: SummaryAgentState) -> str:
    """Determine the next node based on the last message"""
    last_msg = agent_state.patched_history[-1]

    # If the last message contains tool calls, execute them
    if (tool_calls := last_msg.tool_calls) and len(tool_calls) > 0:
        return "tool_calling"

    # Route based on message role
    match last_msg.role:
        case "user" | "tool":
            # User or tool message -> call LLM
            return "llm_chat"
        case "assistant":
            # Assistant responded without tool calls -> go to finalize
            return "finalize"
        case _:
            raise ValueError(f"Unknown message role: {last_msg.role}")


def llm_chat_node(agent_state: SummaryAgentState) -> SummaryAgentState:
    """LLM chat node - gets next action from the model"""
    logger.debug("llm_chat_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("llm_chat")

    selected_state = {
        "workspace": str(agent_state.workspace.working_dir),
        "output_path": agent_state.output_path,
        "current_activated_toolsets": agent_state.toolsets,
    }

    # Update system prompt
    system_prompt = PROMPTS.experiment_summary.system_prompt.render(
        state_text=wrap_dict_to_toon(selected_state),
        toolsets_desc=ToolRegistry.get_toolsets_desc(BUILTIN_TOOLSETS + ALLOWED_TOOLSETS),
    )

    # Construct tools
    tools: dict[str, Tool] = {}
    for toolset in agent_state.toolsets:
        tools.update(ToolRegistry.get_toolset(toolset))
    for toolset in BUILTIN_TOOLSETS:
        tools.update(ToolRegistry.get_toolset(toolset))

    # Get completion from LLM
    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.patched_history,
        system_prompt=(
            Message(role="system", content=system_prompt)
            .with_log(cond=constant.LOG_SYSTEM_PROMPT)
            .content
        ),
        agent_sender=AGENT_NAME,
        tools=[tool.name for tool in tools.values()],
    ).with_log()

    agent_state.add_message(msg)

    return agent_state


def tool_calling_node(agent_state: SummaryAgentState) -> SummaryAgentState:
    """Execute tool calls from the last message"""
    logger.debug("tool_calling_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("tool_calling")

    # Get the last message which contains tool calls
    last_msg = agent_state.patched_history[-1]

    if not last_msg.tool_calls:
        raise ValueError("No tool calls found in the last message")

    # Construct tools
    tools: dict[str, Tool] = {}
    for toolset in agent_state.toolsets:
        tools.update(ToolRegistry.get_toolset(toolset))
    for toolset in BUILTIN_TOOLSETS:
        tools.update(ToolRegistry.get_toolset(toolset))

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
            agent_state.add_message(Message(**tool_response).with_log())
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
            agent_state.add_message(Message(**tool_response).with_log())
            continue
        except AssertionError as e:
            error_msg = f"Invalid tool arguments: {e}"
            tool_response = {
                "role": "tool",
                "tool_name": tool_name,
                "tool_call_id": tool_call.id,
                "content": error_msg,
            }
            agent_state.add_message(Message(**tool_response).with_log())
            continue

        # Execute the tool
        try:
            func = function_map[tool_name]

            # Check if function expects agent_state parameter
            sig = inspect.signature(func)
            if constant.__AGENT_STATE_NAME__ in sig.parameters:
                args.update({constant.__AGENT_STATE_NAME__: agent_state})
            if constant.__CTX_NAME__ in sig.parameters:
                args.update({constant.__CTX_NAME__: {"current_agent": AGENT_NAME}})

            # Execute the tool
            result = func(**args)

            # Create tool response message
            tool_response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": tool_name,
                "content": str(result),  # Ensure result is string
            }

        except Exception as e:
            logger.exception(f"Tool {tool_name} execution failed")
            error_msg = f"Tool {tool_name} execution failed: {e}"
            tool_response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": tool_name,
                "content": error_msg,
            }

        agent_state.add_message(Message(**tool_response).with_log())

    return agent_state


def finalize_node(agent_state: SummaryAgentState) -> SummaryAgentState:
    """Generate final summary and save to file"""
    logger.debug("finalize_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("finalize")

    # Add summary generation prompt
    summary_prompt = Message(
        role="user",
        content=PROMPTS.experiment_summary.summary_prompt.render(),
        agent_sender=AGENT_NAME,
    )
    agent_state.add_message(summary_prompt)

    # Get summary from LLM
    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.patched_history,
        system_prompt=(
            Message(
                role="system",
                content=PROMPTS.experiment_summary.summary_system_prompt.render(),
            )
            .with_log(cond=constant.LOG_SYSTEM_PROMPT)
            .content
        ),
        agent_sender=AGENT_NAME,
        tools=None,  # No tools needed for final summary
    ).with_log()

    # Store the summary text
    agent_state.summary_text = msg.content or ""
    agent_state.add_message(msg)

    # Save summary to file
    if agent_state.output_path is not None:
        try:
            with open(agent_state.output_path, "w", encoding="utf-8") as f:
                f.write(agent_state.summary_text)
            logger.info(f"Summary saved to {agent_state.output_path}")
        except Exception as e:
            logger.error(f"Failed to save summary to {agent_state.output_path}: {e}")

    return agent_state
