"""
Agent for criticizing and giving feedback on the agent's actions
"""

from typing import TYPE_CHECKING, TypeVar

from loguru import logger

from scievo.core import constant
from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.core.utils import wrap_dict_to_toon, wrap_text_with_block
from scievo.prompts import PROMPTS
from scievo.rbank.subgraph import mem_retrieval
from scievo.tools import Tool, ToolRegistry

from .state import CriticAgentState

if TYPE_CHECKING:
    from scievo.core.types import HistoryState, RBankState
    from scievo.rbank.memo import Memo

    MemHistoryMixin = TypeVar("MemHistoryMixin", HistoryState, RBankState)

LLM_NAME = "critic"
AGENT_NAME = "critic"

BUILTIN_TOOLSETS = [
    # "todo",
    "state",
    "history",
    "web",
]
ALLOWED_TOOLSETS = ["fs", "web"]


def create_first_user_msg_node(agent_state: CriticAgentState) -> CriticAgentState:
    logger.debug("create_first_user_msg_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("create_first_user_msg")

    # Stringify all input messages
    input_msgs_texts = []
    for i, msg in enumerate(agent_state.input_msgs):
        plain = msg.to_plain_text()
        input_msgs_texts.append(f"--- Message {i} Begin ---\n{plain}\n--- Message {i} End ---")
    trajectory_text: str = "\n".join(input_msgs_texts)

    # Format using user_prompt template
    user_prompt_content = PROMPTS.critic.user_prompt.render(
        plan_text=agent_state.plan,
        trajectory_text=trajectory_text,
        is_data_agent=agent_state.is_data_agent,
        is_exp_agent=agent_state.is_exp_agent,
    )

    # Add as first user message
    agent_state.add_message(
        Message(role="user", content=user_prompt_content, agent_sender=AGENT_NAME)
    )

    return agent_state


def gateway_node(agent_state: CriticAgentState) -> CriticAgentState:
    # NOTE: Same as data agent
    logger.trace("gateway_node of Agent {}", AGENT_NAME)
    return agent_state


def gateway_conditional(agent_state: CriticAgentState) -> str:
    # NOTE: Same as data agent
    last_msg = agent_state.patched_history[-1]
    if (tool_calls := last_msg.tool_calls) and len(tool_calls) > 0:
        return "tool_calling"

    match last_msg.role:
        case "user" | "tool":
            return "llm_chat"
        case "assistant":
            # finish this round of critic, go to "summary" node
            return "summary"
        case _:
            raise ValueError(f"Unknown message role: {last_msg.role}")


mem_retrieval_subgraph = mem_retrieval.build()
mem_retrieval_subgraph_compiled = mem_retrieval_subgraph.compile()


def llm_chat_node(agent_state: CriticAgentState) -> CriticAgentState:
    logger.debug("llm_chat_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("llm_chat")

    selected_state = {
        "current_activated_toolsets": agent_state.toolsets,
    }

    # retrieve memos
    if constant.REASONING_BANK_ENABLED:
        try:
            mem_dirs = [agent_state.sess_dir / "short_term"]
            if d := agent_state.long_term_mem_dir:
                mem_dirs.append(d)
            if d := agent_state.project_mem_dir:
                mem_dirs.append(d)
            res = mem_retrieval_subgraph_compiled.invoke(
                mem_retrieval.MemRetrievalState(
                    input_msgs=agent_state.patched_history,
                    mem_dirs=mem_dirs,
                    max_num_memos=constant.MEM_RETRIEVAL_MAX_NUM_MEMOS,
                )
            )
            memos: list[Memo] = res.get("output_memos", [])
            from scievo.agents.data_agent.execute import _memos_to_markdown

            memory_text = _memos_to_markdown(memos)
        except Exception:
            logger.exception("mem_retrieval_error")
            memory_text = None
    else:
        memory_text = None

    # update system prompt
    system_prompt = PROMPTS.critic.system_prompt.render(
        state_text=wrap_dict_to_toon(selected_state),
        toolsets_desc=ToolRegistry.get_toolsets_desc(BUILTIN_TOOLSETS + ALLOWED_TOOLSETS),
        memory_text=wrap_text_with_block(memory_text, "markdown"),
        is_data_agent=agent_state.is_data_agent,
        is_exp_agent=agent_state.is_exp_agent,
    )

    # construct tools
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


def tool_calling_node(agent_state: CriticAgentState) -> CriticAgentState:
    """Execute tool calls from the last message and update the graph state"""
    logger.debug("tool_calling_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("tool_calling")
    # Get the last message which contains tool calls
    last_msg = agent_state.patched_history[-1]

    if not last_msg.tool_calls:
        raise ValueError("No tool calls found in the last message")

    # construct tools
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
            agent_state.add_message(Message(**tool_response).with_log())

        except Exception as e:
            error_msg = f"Tool {tool_name} execution failed: {e}"
            tool_response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": tool_name,
                "content": error_msg,
            }
            agent_state.add_message(Message(**tool_response).with_log())

    return agent_state


def summary_node(agent_state: CriticAgentState) -> CriticAgentState:
    logger.debug("summary_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("summary")

    # update system prompt
    system_prompt = PROMPTS.critic.system_prompt.render(
        toolsets_desc={},
        is_data_agent=agent_state.is_data_agent,
        is_exp_agent=agent_state.is_exp_agent,
    )

    # Render the summary prompt
    summary_prompt_content = PROMPTS.critic.user_prompt_summary.render(
        is_data_agent=agent_state.is_data_agent,
        is_exp_agent=agent_state.is_exp_agent,
    )

    # Add summary request as user message
    agent_state.add_message(
        Message(role="user", content=summary_prompt_content, agent_sender=AGENT_NAME)
    )

    # Get AI summary response
    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.patched_history,
        system_prompt=system_prompt,
        agent_sender=AGENT_NAME,
    ).with_log()
    agent_state.add_message(msg)

    # Set the summary message as the output
    agent_state.critic_msg = msg

    return agent_state
