"""
Agent for data understanding and processing
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from loguru import logger

from scievo import history_compression
from scievo.agents import critic_agent
from scievo.core import constant
from scievo.core.errors import sprint_chained_exception
from scievo.core.llms import ModelRegistry
from scievo.core.types import HistoryState, Message, RBankState
from scievo.core.utils import wrap_text_with_block
from scievo.prompts import PROMPTS
from scievo.rbank.subgraph import mem_extraction, mem_retrieval
from scievo.tools import Tool, ToolRegistry

from .state import DataAgentState

if TYPE_CHECKING:
    from scievo.rbank.memo import Memo

MemHistoryMixin = TypeVar("MemHistoryMixin", HistoryState, RBankState)

LLM_NAME = "data"
AGENT_NAME = "data"

BUILTIN_TOOLSETS = [
    # "todo",
    "state",
    "history",
    "fs",
]
ALLOWED_TOOLSETS = ["web"]


def gateway_node(agent_state: DataAgentState) -> DataAgentState:
    # NOTE: this node does nothing, it's just a placeholder for the conditional edges
    # Check `gateway_conditional` for the actual logic
    logger.trace("gateway_node of Agent {}", AGENT_NAME)
    return agent_state


def gateway_conditional(agent_state: DataAgentState) -> str:
    # compress history if needed
    if (
        constant.HISTORY_AUTO_COMPRESSION
        and "history_compression" not in agent_state.node_history[-2:]
        and agent_state.total_patched_tokens > constant.HISTORY_AUTO_COMPRESSION_TOKEN_THRESHOLD
    ):
        return "history_compression"

    if (
        constant.REASONING_BANK_ENABLED
        and len(agent_state.node_history) > 0
        and agent_state.node_history[-1] != "mem_extraction"
        and agent_state.round > 0
        and agent_state.round % constant.MEM_EXTRACTION_ROUND_FREQ == 0
    ):
        return "mem_extraction"

    if len(agent_state.patched_history) == 0:
        logger.warning("patched_history is empty, returning llm_chat")
        return "llm_chat"

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
    agent_state.add_node_history("llm_chat")

    selected_state = {
        "workspace": str(agent_state.workspace.working_dir),
        "current_activated_toolsets": list(set(agent_state.toolsets)),
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
            memory_text = _memos_to_markdown(memos)
        except Exception:
            logger.exception("mem_retrieval_error")
            memory_text = None
    else:
        memory_text = None

    # update system prompt
    import json

    system_prompt = PROMPTS.data.system_prompt.render(
        state_text=wrap_text_with_block(json.dumps(selected_state, indent=2), "json"),
        toolsets_desc=ToolRegistry.get_toolsets_desc(BUILTIN_TOOLSETS + ALLOWED_TOOLSETS),
        memory_text=wrap_text_with_block(memory_text, "markdown"),
        current_plan=(
            agent_state.remaining_plans[0] if len(agent_state.remaining_plans) > 0 else None
        ),
    )

    # construct tools
    tools: dict[str, Tool] = {}
    for toolset in agent_state.toolsets:
        tools.update(ToolRegistry.get_toolset(toolset))
    for toolset in BUILTIN_TOOLSETS:
        tools.update(ToolRegistry.get_toolset(toolset))

    # Ensure there's at least one non-system message for Anthropic API
    history = agent_state.patched_history
    if len(history) == 0 or all(msg.role == "system" for msg in history):
        # Add a dummy user message if history is empty or only contains system messages
        logger.warning(
            "patched_history is empty or only contains system messages, adding dummy user message"
        )
        history = [
            Message(
                role="user",
                content="Please continue with the task.",
                agent_sender=AGENT_NAME,
            )
        ]

    msg = ModelRegistry.completion(
        LLM_NAME,
        history,
        system_prompt=(
            Message(role="system", content=system_prompt)
            .with_log(cond=constant.LOG_SYSTEM_PROMPT)
            .content
        ),
        agent_sender=AGENT_NAME,
        tools=[tool.name for tool in tools.values()],
    ).with_log()
    agent_state.add_message(msg)

    llm_output = (
        msg.content
        if msg.content
        else ("[Tool calls: " + str(len(msg.tool_calls)) + "]" if msg.tool_calls else "[No output]")
    )

    agent_state.intermediate_state.append(
        {
            "node_name": "llm_chat",
            "output": llm_output,
        }
    )

    return agent_state


def tool_calling_node(agent_state: DataAgentState) -> DataAgentState:
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

    tool_results = []

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
            tool_results.append({"tool": tool_name, "result": error_msg})
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
            tool_results.append({"tool": tool_name, "result": error_msg})
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
            tool_results.append({"tool": tool_name, "result": error_msg})
            continue

        # Execute the tool
        result = None
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

            # Create tool response message
            tool_response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": tool_name,
                "content": str(result),  # Ensure result is string
            }
            tool_results.append(
                {"tool": tool_name, "result": str(result)[:1000] if result else "No result"}
            )

        except Exception as e:
            error_msg = f"Tool {tool_name} execution failed: {e}"
            tool_response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": tool_name,
                "content": error_msg,
            }
            tool_results.append({"tool": tool_name, "result": error_msg})

        tool_response_msg = Message(**tool_response).with_log()
        agent_state.add_message(tool_response_msg)

    tool_output_parts = []
    for tr in tool_results:
        tool_output_parts.append(f"Tool: {tr['tool']}\nResult: {tr['result']}")

    tool_output = "\n\n".join(tool_output_parts) if tool_output_parts else "No tool calls executed"

    agent_state.intermediate_state.append(
        {
            "node_name": "tool_calling",
            "output": tool_output,
        }
    )

    return agent_state


mem_extraction_subgraph = mem_extraction.build()
mem_extraction_subgraph_compiled = mem_extraction_subgraph.compile()


def mem_extraction_node(agent_state: MemHistoryMixin) -> MemHistoryMixin:
    logger.debug("mem_extraction_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("mem_extraction")
    context_window = agent_state.patched_history[-constant.MEM_EXTRACTION_CONTEXT_WINDOW :]
    logger.info("Agent {} begins to Memory Extraction", AGENT_NAME)
    mem_output = "Memory extraction completed"
    try:
        result = mem_extraction_subgraph_compiled.invoke(
            mem_extraction.MemExtractionState(
                mem_dir=Path(agent_state.sess_dir) / f"short_term",
                input_msgs=context_window,
                input_agent_name=AGENT_NAME,
            )
        )
        if isinstance(result, dict) and "output_memos" in result:
            mem_output = f"Extracted {len(result.get('output_memos', []))} memory entries"
    except Exception as e:
        error_msg = f"mem_extraction_error: {sprint_chained_exception(e)}"
        agent_state.add_message(
            Message(
                role="assistant",
                content=error_msg,
                agent_sender=AGENT_NAME,
            ).with_log()
        )
        mem_output = error_msg

    if isinstance(agent_state, DataAgentState):
        agent_state.intermediate_state.append(
            {
                "node_name": "mem_extraction",
                "output": mem_output,
            }
        )

    return agent_state


def history_compression_node(agent_state: DataAgentState) -> DataAgentState:
    logger.debug("history_compression_node of Agent {}", AGENT_NAME)

    history_before = len(agent_state.history)
    agent_state = history_compression.invoke_history_compression(agent_state)
    history_after = len(agent_state.history)

    compression_output = f"Compressed history: {history_before} -> {history_after} messages"
    if agent_state.history_patches:
        last_patch = agent_state.history_patches[-1]
        if last_patch.patched_message and last_patch.patched_message.content:
            compression_output = f"Compressed {last_patch.n_messages} messages into:\n{last_patch.patched_message.content[:500]}"

    agent_state.intermediate_state.append(
        {
            "node_name": "history_compression",
            "output": compression_output,
        }
    )

    return agent_state


def generate_summary_node(agent_state: DataAgentState) -> DataAgentState:
    """Generate analysis summary and store it in agent state"""
    logger.debug("generate_summary_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("generate_summary")

    try:
        # Construct a summary request message
        summary_system_prompt = PROMPTS.data.summary_system_prompt
        summary_user_prompt = PROMPTS.data.summary_user_prompt

        agent_state.add_message(
            Message(
                role="user",
                content=summary_user_prompt.render(),
            ).with_log(cond=constant.LOG_SYSTEM_PROMPT)
        )

        # Call LLM to generate summary
        summary_msg = ModelRegistry.completion(
            LLM_NAME,
            agent_state.patched_history,
            system_prompt=summary_system_prompt.render(),
            agent_sender=AGENT_NAME,
        ).with_log()

        agent_state.add_message(summary_msg)

        # Extract summary content
        if summary_msg.role != "assistant" or not summary_msg.content:
            raise ValueError("Failed to get summary from LLM")

        # Store summary in state
        agent_state.output_summary = summary_msg.content
        logger.info("Analysis summary generated successfully")

    except Exception as e:
        error_msg = f"Failed to generate analysis summary: {sprint_chained_exception(e)}"
        agent_state.add_message(
            Message(
                role="assistant",
                content=error_msg,
                agent_sender=AGENT_NAME,
            ).with_log()
        )
        logger.error("generate_summary_node failed: {}", error_msg)

    summary_output = (
        summary_msg.content
        if "summary_msg" in locals() and summary_msg.content
        else (error_msg if "error_msg" in locals() else "No summary generated")
    )

    agent_state.intermediate_state.append(
        {
            "node_name": "generate_summary",
            "output": summary_output,
        }
    )

    return agent_state


critic_subgraph = critic_agent.build()
critic_subgraph_compiled = critic_subgraph.compile()


def critic_node(agent_state: DataAgentState) -> DataAgentState:
    logger.trace("critic_node of Agent {}", AGENT_NAME)

    if not constant.CRITIC_ENABLED:
        return agent_state

    try:
        current_plan = (
            agent_state.remaining_plans[0] if len(agent_state.remaining_plans) > 0 else "N/A"
        )
        res = critic_subgraph_compiled.invoke(
            critic_agent.CriticAgentState(
                input_msgs=agent_state.patched_history[-constant.CRITIC_CONTEXT_WINDOW :],
                plan=agent_state.remaining_plans[0],
                is_data_agent=True,
                # RBankState
                sess_dir=agent_state.sess_dir,
                long_term_mem_dir=agent_state.long_term_mem_dir,
                project_mem_dir=agent_state.project_mem_dir,
            )
        )
        assert res.get("critic_msg", None) is not None, "critic_msg is None"
        critic_msg: Message = res.get("critic_msg")
        agent_state.add_message(critic_msg.with_log())
        critic_output = critic_msg.content if critic_msg.content else "No critic feedback"
    except Exception as e:
        error_msg = f"critic_error: {sprint_chained_exception(e)}"
        agent_state.add_message(
            Message(
                role="assistant",
                content=error_msg,
                agent_sender=AGENT_NAME,
            ).with_log()
        )
        critic_output = error_msg

    agent_state.intermediate_state.append(
        {
            "node_name": "critic",
            "output": critic_output if "critic_output" in locals() else "No critic output",
        }
    )

    return agent_state
