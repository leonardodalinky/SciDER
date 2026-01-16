"""
Subgraph for compressing conversation history.
"""

from typing import TypeVar

from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel

from scievo.core import constant
from scievo.core.errors import AgentError, sprint_chained_exception
from scievo.core.llms import ModelRegistry
from scievo.core.types import HistoryState, Message
from scievo.core.utils import parse_markdown_from_llm_response
from scievo.prompts import PROMPTS

LLM_NAME = "history"
AGENT_NAME = "history"


class HistoryCompressionState(BaseModel):
    """State for history compression subgraph."""

    # Input: the history state to compress
    hc_input_history_state: HistoryState

    # Input: whether to keep the original messages (default: 4)
    hc_keep_first_n_messages: int = 4

    # Output: the compressed message
    hc_output_patch: HistoryState.HistoryPatch | None = None


def validate_compression_input(
    state: HistoryCompressionState,
) -> HistoryCompressionState:
    """Validate the input parameters for compression."""
    logger.debug("validate_compression_input")

    if (
        state.hc_input_history_state.total_patched_tokens
        < constant.HISTORY_AUTO_COMPRESSION_TOKEN_THRESHOLD
    ):
        e = f"Input history state has {state.hc_input_history_state.total_patched_tokens} tokens, which is less than the threshold of {constant.HISTORY_AUTO_COMPRESSION_TOKEN_THRESHOLD}."
        logger.debug("Consolidation error: {}", e)
        raise AgentError(e, agent_name=AGENT_NAME)

    return state


def compress_history_node(state: HistoryCompressionState) -> HistoryCompressionState:
    """Compress a range of messages using LLM."""
    logger.debug("compress_history_node")

    try:
        match (
            last_kept_msg := state.hc_input_history_state.patched_history[
                state.hc_keep_first_n_messages - 1
            ]
        ).role:
            case "user":
                keep_first_n_messages = state.hc_keep_first_n_messages
            case "assistant":
                if last_kept_msg.tool_calls and len(last_kept_msg.tool_calls) > 0:
                    # if the last kept message has tool calls, backtrack one more message
                    keep_first_n_messages = state.hc_keep_first_n_messages - 1
                else:
                    keep_first_n_messages = state.hc_keep_first_n_messages
            case "tool":
                # find the last assistant message before this
                for i in range(state.hc_keep_first_n_messages - 2, -1, -1):
                    if state.hc_input_history_state.patched_history[i].role == "assistant":
                        keep_first_n_messages = i + 1
                        keep_first_n_messages -= (
                            1  # also drop the assistant msg that called the tool
                        )

        # Skip the first n messages as specified
        history_to_process = state.hc_input_history_state.patched_history[keep_first_n_messages:]

        messages_to_compress: list[Message] = []
        n_tokens = 0
        N_TOKENS_TO_COMPRESS = (
            1 - constant.HISTORY_AUTO_COMPRESSION_KEEP_RATIO
        ) * state.hc_input_history_state.total_patched_tokens
        for msg in history_to_process:
            if n_tokens > N_TOKENS_TO_COMPRESS:
                break
            messages_to_compress.append(msg)
            n_tokens += msg.n_tokens

        # if last msg has tool call, add those tool msgs too
        match (last_msg := messages_to_compress[-1]).role:
            case "user":
                pass
            case "assistant":
                if last_msg.tool_calls and len(last_msg.tool_calls) > 0:
                    messages_to_compress.extend(
                        history_to_process[
                            len(messages_to_compress) : len(messages_to_compress)
                            + len(last_msg.tool_calls)
                        ]
                    )
            case "tool":
                # include all the following tool messages
                for msg in history_to_process[len(messages_to_compress) :]:
                    if msg.role == "tool":
                        messages_to_compress.append(msg)
                    else:
                        break
            case _:
                pass

        assert len(messages_to_compress) > 0

        start_idx = keep_first_n_messages
        end_idx = keep_first_n_messages + len(messages_to_compress)

        # Format messages for LLM
        input_msgs_texts = []
        for i, msg in enumerate(messages_to_compress):
            plain = msg.to_plain_text(verbose_tool=False)
            input_msgs_texts.append(f"--- Message {i} Begin ---\n{plain}\n--- Message {i} End ---")
        input_msgs_text: str = "\n".join(input_msgs_texts)

        # Get prompts
        system_prompt = PROMPTS.history.compression_system_prompt.render()
        user_prompt = PROMPTS.history.compression_user_prompt.render(
            n_messages=len(messages_to_compress),
            message_text=input_msgs_text,
        )

        user_msg = Message(
            role="user",
            content=user_prompt,
        )

        # Use LLM to compress
        compressed_msg = ModelRegistry.completion(
            LLM_NAME,
            [user_msg],
            system_prompt=(
                Message(role="system", content=system_prompt)
                .with_log(cond=constant.LOG_SYSTEM_PROMPT)
                .content
            ),
            agent_sender=AGENT_NAME,
        )

        compressed_history_text = parse_markdown_from_llm_response(compressed_msg)

        # Create a system message with the compressed content
        patched_message = Message(
            role="assistant",
            content=PROMPTS.history.compressed_patch_template.render(
                patch_id=state.hc_input_history_state.next_patch_id(),
                n_messages=len(messages_to_compress),
                compressed_history_text=compressed_history_text,
            ),
            agent_sender=AGENT_NAME,
        ).with_log()

        state.hc_output_patch = HistoryState.HistoryPatch(
            patch_id=state.hc_input_history_state.next_patch_id(),
            start_idx=start_idx,
            end_idx=end_idx,
            patched_message=patched_message,
        )
        logger.debug(f"Successfully compressed {len(messages_to_compress)} messages")
        return state

    except Exception as e:
        logger.debug("Consolidation error: {}", e)
        raise AgentError(f"history_compression_error", agent_name=AGENT_NAME) from e


@logger.catch
def build():
    """Build the history compression subgraph."""
    g = StateGraph(HistoryCompressionState)

    # Add nodes
    g.add_node("validate", validate_compression_input)
    g.add_node("compress", compress_history_node)

    # Add edges
    g.add_edge(START, "validate")
    g.add_edge("validate", "compress")
    g.add_edge("compress", END)

    return g


########################
## Compiled subgraph instance
########################

T = TypeVar("T", bound=HistoryState)

history_compress_subgraph_compiled = build().compile()


def invoke_history_compression(agent_state: T) -> T:
    agent_state.add_node_history("history_compression")
    try:
        res = history_compress_subgraph_compiled.invoke(
            HistoryCompressionState(
                hc_input_history_state=agent_state,
            )
        )
    except Exception as e:
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"history_compression_error: {sprint_chained_exception(e)}",
                agent_sender=AGENT_NAME,
            ).with_log()
        )
        return agent_state

    output_patch = res.get("hc_output_patch", None)
    if output_patch:
        agent_state.history_patches.append(output_patch)
    else:
        agent_state.add_message(
            Message(
                role="assistant",
                content="history_compression_error: No output patch",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    return agent_state
