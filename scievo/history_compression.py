"""
Subgraph for compressing conversation history.
"""

from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel

from scievo.core import constant
from scievo.core.errors import AgentError
from scievo.core.llms import ModelRegistry
from scievo.core.types import HistoryState, Message
from scievo.core.utils import parse_markdown_from_llm_response
from scievo.prompts import PROMPTS

LLM_NAME = "history"
AGENT_NAME = "history"


class HistoryCompressionState(BaseModel):
    """State for history compression subgraph."""

    # Input: the history state to compress
    input_history_state: HistoryState

    # Output: the compressed message
    output_patch: HistoryState.HistoryPatch | None = None


def validate_compression_input(state: HistoryCompressionState) -> HistoryCompressionState:
    """Validate the input parameters for compression."""
    logger.debug("validate_compression_input")

    if (
        state.input_history_state.total_patched_tokens
        < constant.HISTORY_AUTO_COMPRESSION_TOKEN_THRESHOLD
    ):
        e = f"Input history state has {state.input_history_state.total_patched_tokens} tokens, which is less than the threshold of {constant.HISTORY_AUTO_COMPRESSION_TOKEN_THRESHOLD}."
        logger.debug("Consolidation error: {}", e)
        raise AgentError(e, agent_name=AGENT_NAME)

    return state


def compress_history_node(state: HistoryCompressionState) -> HistoryCompressionState:
    """Compress a range of messages using LLM."""
    logger.debug("compress_history_node")

    try:

        # Get messages to compress
        # messages = state.input_history_state.get_history_range(state.start_idx, state.end_idx)

        messages_to_compress: list[Message] = []
        n_tokens = 0
        N_TOKENS_TO_COMPRESS = (
            1 - constant.HISTORY_AUTO_COMPRESSION_KEEP_RATIO
        ) * state.input_history_state.total_patched_tokens
        for msg in state.input_history_state.patched_history:
            if n_tokens > N_TOKENS_TO_COMPRESS:
                break
            messages_to_compress.append(msg)
            n_tokens += msg.n_tokens

        # if last msg has tool call, add those tool msgs too
        if messages_to_compress[-1].tool_calls and len(messages_to_compress[-1].tool_calls) > 0:
            for msg in state.input_history_state.patched_history[len(messages_to_compress) :]:
                if msg.role != "tool":
                    break
                messages_to_compress.append(msg)
                n_tokens += msg.n_tokens

        assert len(messages_to_compress) > 0

        start_idx = 0
        end_idx = len(messages_to_compress)

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
                patch_id=state.input_history_state.next_patch_id(),
                n_messages=len(messages_to_compress),
                compressed_history_text=compressed_history_text,
            ),
            agent_sender=AGENT_NAME,
        ).with_log()

        state.output_patch = HistoryState.HistoryPatch(
            patch_id=state.input_history_state.next_patch_id(),
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
