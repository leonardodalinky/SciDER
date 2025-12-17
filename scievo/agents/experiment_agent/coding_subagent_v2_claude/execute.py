"""
Execution nodes for the Coding Subagent V2
"""

import json

from loguru import logger

from scievo.agents import critic_agent
from scievo.core import constant
from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.core.utils import wrap_dict_to_toon
from scievo.prompts import PROMPTS
from scievo.tools import Tool, ToolRegistry

from .state import CodingAgentState

LLM_NAME = "experiment_coding"
AGENT_NAME = "experiment_coding"

BUILTIN_TOOLSETS = [
    "state",
    "history",
]
ALLOWED_TOOLSETS = [
    "fs",
    "shell",
    "web",
    "claude_agent_sdk",
    "claude_code",
]  # Claude Agent SDK is the primary toolset


def gateway_node(agent_state: CodingAgentState) -> CodingAgentState:
    """Gateway node for routing decisions."""
    logger.trace("gateway_node of Agent {}", AGENT_NAME)
    return agent_state


def gateway_conditional(agent_state: CodingAgentState) -> str:
    """Determine the next node based on current state."""
    last_msg = agent_state.patched_history[-1]

    if (tool_calls := last_msg.tool_calls) and len(tool_calls) > 0:
        return "tool_calling"

    match last_msg.role:
        case "user" | "tool":
            return "llm_chat"
        case "assistant":
            return "critic_before_replan"
        case _:
            raise ValueError(f"Unknown message role: {last_msg.role}")


def llm_chat_node(agent_state: CodingAgentState) -> CodingAgentState:
    """LLM chat node for generating responses."""
    logger.debug("llm_chat_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("llm_chat")

    selected_state = {
        "workspace": agent_state.workspace.working_dir,
        "current_activated_toolsets": agent_state.toolsets,
    }

    # Update system prompt (no memory retrieval - unlike data_agent)
    system_prompt = PROMPTS.experiment_claude_coding_v2.system_prompt.render(
        state_text=wrap_dict_to_toon(selected_state),
        toolsets_desc=ToolRegistry.get_toolsets_desc(BUILTIN_TOOLSETS + ALLOWED_TOOLSETS),
        current_plan=(
            agent_state.remaining_plans[0] if len(agent_state.remaining_plans) > 0 else None
        ),
    )

    # Construct tools
    tools: dict[str, Tool] = {}
    for toolset in agent_state.toolsets:
        tools.update(ToolRegistry.get_toolset(toolset))
    for toolset in BUILTIN_TOOLSETS:
        tools.update(ToolRegistry.get_toolset(toolset))

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


def tool_calling_node(agent_state: CodingAgentState) -> CodingAgentState:
    """Execute tool calls from the last message."""
    logger.debug("tool_calling_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("tool_calling")

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
            with agent_state.workspace:
                result = func(**args)

            tool_response = {
                "role": "tool",
                "tool_name": tool_name,
                "tool_call_id": tool_call.id,
                "content": str(result),
            }
        except Exception as e:
            logger.exception(f"Tool execution error: {tool_name}")
            tool_response = {
                "role": "tool",
                "tool_name": tool_name,
                "tool_call_id": tool_call.id,
                "content": f"Error executing tool: {str(e)}",
            }

        agent_state.add_message(Message(**tool_response).with_log())

    return agent_state


# Build and invoke critic agent
critic_graph = critic_agent.build()
critic_compiled = critic_graph.compile()


def critic_node(agent_state: CodingAgentState) -> CodingAgentState:
    """Critic node to evaluate the plan step completion."""
    logger.debug("critic_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("critic")

    # Get current plan step
    current_plan = agent_state.remaining_plans[0] if len(agent_state.remaining_plans) > 0 else "N/A"

    critic_state = critic_agent.CriticAgentState(
        input_msgs=agent_state.patched_history[-10:],  # Last 10 messages for context
        plan=current_plan,
        is_data_agent=False,
        is_exp_agent=True,
        toolsets=["fs", "web"],
        sess_dir=agent_state.workspace.working_dir,
        long_term_mem_dir="",
        project_mem_dir="",
    )

    try:
        result = critic_compiled.invoke(critic_state)
        critic_msg = result.get("critic_msg")
        if critic_msg:
            agent_state.critic_feedback = critic_msg.content or ""
        else:
            agent_state.critic_feedback = "No critic feedback available."
    except Exception as e:
        logger.exception("Critic agent error")
        agent_state.critic_feedback = f"Critic evaluation failed: {str(e)}"

    # Log the critic feedback
    Message(
        role="assistant",
        content=f"[Critic Feedback] {agent_state.critic_feedback}",
        agent_sender="critic",
    ).with_log()

    return agent_state
