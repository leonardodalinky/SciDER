"""
This subgraph is used to retrieve the memory based on the trajectory of messages.
"""

import json
from pathlib import Path

import numpy as np
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel

from scievo.core.errors import AgentError
from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.prompts import PROMPTS
from scievo.rbank.memo import Memo, MemoEmbeddings
from scievo.rbank.utils import cosine_similarity

LLM_NAME = "embed"
AGENT_NAME = "mem_retrieval"


class MemRetrievalState(BaseModel):
    input_msgs: list[Message]
    mem_dirs: list[str | Path]
    max_num_memos: int

    # intermediate
    summary_embedding: list[float] = []

    # output
    output_memos: list[Memo] = []


def format_input_msgs(
    input_msgs: list[Message], max_tokens: int = 6400, max_token_per_msg: int = 3200
) -> str:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")  # NOTE: hardcode for now
    # reverse the order of messages
    input_msgs = input_msgs[::-1]
    total_tokens = 0
    ret_text = ""
    for i, msg in enumerate(input_msgs):
        if total_tokens >= max_tokens:
            break
        msg_tokens = enc.encode(msg.to_plain_text())
        is_truncated = False
        if len(msg_tokens) > max_token_per_msg:
            msg_tokens = msg_tokens[:max_token_per_msg]
            is_truncated = True
        if (len(msg_tokens) + total_tokens) > max_tokens:
            msg_tokens = msg_tokens[: max_tokens - total_tokens]
            is_truncated = True
        total_tokens += len(msg_tokens)
        text = enc.decode(msg_tokens)
        ret_text += f"# Latest Message {i + 1}:\n{text}\n\n"
        if is_truncated:
            ret_text += "(truncated...)\n\n"

    return ret_text


def embedding_node(state: MemRetrievalState) -> MemRetrievalState:
    """Compute embeddings for the summary text."""
    logger.debug("Mem Retrieval embedding begin: {} input msgs", len(state.input_msgs))

    # Check if input_msgs is empty
    if len(state.input_msgs) == 0:
        logger.debug("No input messages, skipping embedding")
        state.summary_embedding = None
        return state

    try:
        formatted_text = format_input_msgs(state.input_msgs)
        # Check if formatted text is empty or only whitespace
        if not formatted_text or not formatted_text.strip():
            logger.debug("Formatted input is empty, skipping embedding")
            state.summary_embedding = None
            return state

        embeddings = ModelRegistry.embedding(LLM_NAME, [formatted_text])
        if embeddings and len(embeddings) > 0:
            state.summary_embedding = embeddings[0]
        else:
            logger.debug("Retrieval error: embedding returned empty result")
            state.summary_embedding = None
    except Exception as e:
        logger.debug("Retrieval error: embedding_error: {}", e)
        # Don't raise error, just set embedding to None
        state.summary_embedding = None

    logger.debug(
        "Mem Retrieval embedding end: {} dims",
        len(state.summary_embedding) if state.summary_embedding else 0,
    )
    return state


def retrieval_node(state: MemRetrievalState) -> MemRetrievalState:
    """Retrieve the top-k most relevant memos from the memory bank."""
    if not state.summary_embedding:
        logger.debug("Retrieval error: no summary embedding available")
        raise AgentError("no summary embedding available", agent_name=AGENT_NAME)

    # Convert query embedding to numpy array
    query_emb = np.array(state.summary_embedding, dtype=np.float32)

    # Collect all memo embeddings from JSON files
    memo_candidates: list[tuple[float, Path, Path]] = []  # (similarity, json_path, md_path)

    logger.debug("Retrieval scanning mem dirs: {}", state.mem_dirs)
    for mem_dir in state.mem_dirs:
        mem_dir_path = Path(mem_dir)
        if not mem_dir_path.exists() or not mem_dir_path.is_dir():
            logger.debug("Skip non-dir mem path: {}", mem_dir_path)
            continue

        # Find all JSON files (containing embeddings)
        json_files = list(mem_dir_path.glob("*.json"))

        # TODO: cache embeddings for each memo
        for json_path in json_files:
            try:
                # Load the embedding
                memo_emb = MemoEmbeddings.from_json_file(json_path)

                # Get the embedding for the same LLM
                emb_vec = memo_emb.get_embedding(LLM_NAME)
                if emb_vec is None:
                    continue

                # Compute cosine similarity
                similarity = cosine_similarity(query_emb, emb_vec)

                # Find corresponding markdown file
                md_path = json_path.with_suffix(".md")
                if md_path.exists():
                    memo_candidates.append((float(similarity), json_path, md_path))

            except Exception:
                # Skip files that can't be loaded or parsed
                continue

    # Sort by similarity (descending) and take top-k
    memo_candidates.sort(key=lambda x: x[0], reverse=True)
    top_k_candidates = memo_candidates[: state.max_num_memos]

    # Load the corresponding Memo objects
    retrieved_memos: list[Memo] = []
    for similarity, json_path, md_path in top_k_candidates:
        try:
            memo = Memo.from_markdown_file(md_path)
            retrieved_memos.append(memo)
        except Exception:
            # Skip memos that can't be loaded
            continue

    state.output_memos = retrieved_memos
    logger.debug(
        "Retrieval end: {} candidates, {} returned", len(memo_candidates), len(retrieved_memos)
    )
    return state


@logger.catch
def build():
    """Build the memory retrieval subgraph."""
    g = StateGraph(MemRetrievalState)

    g.add_node("embedding", embedding_node)
    g.add_node("retrieval", retrieval_node)

    g.add_edge(START, "embedding")
    g.add_edge("embedding", "retrieval")
    g.add_edge("retrieval", END)

    return g
