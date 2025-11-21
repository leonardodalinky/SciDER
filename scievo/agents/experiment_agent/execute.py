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

if TYPE_CHECKING:
    from scievo.rbank.memo import Memo

LLM_NAME = "execute"
AGENT_NAME = "experiment_executor"

BUILTIN_TOOLSETS = [
    "state",
    "history",
]
ALLOWED_TOOLSETS = ["shell", "fs", "cursor", "environment"]


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
        case "user" | "tool":
            logger.debug("user or tool message, going to llm_chat_node")
            return "llm_chat"
        case "assistant":
            if agent_state.remaining_plans and len(agent_state.remaining_plans) > 0:
                logger.debug("remaining plans, going to llm_chat_node")
                # Add user message to trigger execution of next step
                agent_state.add_message(
                    Message(
                        role="user",
                        content=PROMPTS.experiment.replanner_user_response.render(
                            next_step=agent_state.remaining_plans[0],
                            selected_action=None,
                            has_options=False,
                        ),
                    )
                )
                return "llm_chat"
            else:
                logger.debug("no remaining plans, going to replanner")
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


def llm_chat_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    logger.debug("llm_chat_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("llm_chat")

    selected_state = {
        "current_working_dir": agent_state.local_env.working_dir,
        "current_activated_toolsets": agent_state.toolsets,
        "repo_dir": str(agent_state.repo_dir) if agent_state.repo_dir else None,
    }

    # retrieve memos
    if constant.REASONING_BANK_ENABLED:
        try:
            mem_dirs = [agent_state.sess_dir / "short_term"]
            if hasattr(agent_state, "long_term_mem_dir") and agent_state.long_term_mem_dir:
                mem_dirs.append(agent_state.long_term_mem_dir)
            if hasattr(agent_state, "project_mem_dir") and agent_state.project_mem_dir:
                mem_dirs.append(agent_state.project_mem_dir)
            res = mem_retrieval_subgraph_compiled.invoke(
                mem_retrieval.MemRetrievalState(
                    input_msgs=agent_state.patched_history,
                    mem_dirs=mem_dirs,
                    max_num_memos=constant.MEM_RETRIEVAL_MAX_NUM_MEMOS,
                )
            )
            memos: list["Memo"] = res.get("output_memos", [])
            memory_text = _memos_to_markdown(memos)
        except Exception:
            logger.exception("mem_retrieval_error")
            memory_text = None
    else:
        memory_text = None

    # update system prompt
    system_prompt = PROMPTS.experiment.experiment_chat_system_prompt.render(
        state_text=wrap_dict_to_toon(selected_state),
        toolsets_desc=ToolRegistry.get_toolsets_desc(BUILTIN_TOOLSETS + ALLOWED_TOOLSETS),
        memory_text=wrap_text_with_block(memory_text, "markdown"),
        current_plan=(
            agent_state.remaining_plans[0] if len(agent_state.remaining_plans) > 0 else None
        ),
    )

    logger.debug(f"llm_chat_node of Agent {AGENT_NAME} system prompt: {system_prompt}")

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
    msg.role = "user"
    agent_state.add_message(msg)

    return agent_state


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
    for toolset in BUILTIN_TOOLSETS:
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
