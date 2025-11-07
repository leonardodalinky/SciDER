"""
This subgraph is used to retrieve the memory based on the trajectory of messages.
"""

import json
from pathlib import Path

import numpy as np
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.prompts import PROMPTS
from scievo.rbank.memo import Memo, MemoEmbeddings
from scievo.rbank.utils import cosine_similarity

LLM_NAME_SUMMARIZE = "mem_retrieval_summarize"
LLM_NAME_EMBED = "embed"
AGENT_NAME = "mem_retrieval"


class MemRetrievalState(BaseModel):
    input_msgs: list[Message]
    mem_dirs: list[str | Path]
    max_num_memos: int = 3

    # intermediate
    summary_text: str = ""
    summary_embedding: list[float] = []

    # output
    output_memos: list[Memo] = []
    output_error: str | None = None


def summarize_node(state: MemRetrievalState) -> MemRetrievalState:
    """Summarize the input messages using the retrieval summarize prompt."""
    # Format input messages
    input_msgs_texts = []
    for i, msg in enumerate(state.input_msgs):
        plain = msg.to_plain_text()
        input_msgs_texts.append(f"--- Message {i} Begin ---\n{plain}\n--- Message {i} End ---")
    input_msgs_text: str = "\n".join(input_msgs_texts)

    # Use the retrieval summarize prompt
    system_prompt = PROMPTS.rbank.mem_retrieval_summarize_system_prompt
    user_prompt = PROMPTS.rbank.mem_retrieval_summarize_user_prompt.format(
        trajectory=input_msgs_text,
    )
    user_msg = Message(
        role="user",
        content=user_prompt,
    )

    # Get summary from LLM
    try:
        summary_msg = ModelRegistry.completion(
            LLM_NAME_SUMMARIZE,
            [user_msg],
            system_prompt,
            agent_sender=AGENT_NAME,
        )
        state.summary_text = summary_msg.content
    except Exception as e:
        state.output_error = f"summarize_error: {e}"
        return state

    return state


def embedding_node(state: MemRetrievalState) -> MemRetrievalState:
    """Compute embeddings for the summary text."""
    if state.output_error:
        return state

    if not state.summary_text:
        state.output_error = "no summary text to embed"
        return state

    try:
        embeddings = ModelRegistry.embedding(LLM_NAME_EMBED, [state.summary_text])
        if embeddings and len(embeddings) > 0:
            state.summary_embedding = embeddings[0]
        else:
            state.output_error = "embedding returned empty result"
    except Exception as e:
        state.output_error = f"embedding_error: {e}"

    return state


def retrieval_node(state: MemRetrievalState) -> MemRetrievalState:
    """Retrieve the top-k most relevant memos from the memory bank."""
    if state.output_error:
        return state

    if not state.summary_embedding:
        state.output_error = "no summary embedding available"
        return state

    # Convert query embedding to numpy array
    query_emb = np.array(state.summary_embedding, dtype=np.float32)

    # Collect all memo embeddings from JSON files
    memo_candidates: list[tuple[float, Path, Path]] = []  # (similarity, json_path, md_path)

    for mem_dir in state.mem_dirs:
        mem_dir_path = Path(mem_dir)
        if not mem_dir_path.exists() or not mem_dir_path.is_dir():
            continue

        # Find all JSON files (containing embeddings)
        json_files = list(mem_dir_path.glob("*.json"))

        for json_path in json_files:
            try:
                # Load the embedding
                with open(json_path, "r") as f:
                    memo_emb_data = json.load(f)
                memo_emb = MemoEmbeddings.model_validate(memo_emb_data)

                # Get the embedding for the same LLM
                emb_vec = memo_emb.get_embedding(LLM_NAME_EMBED)
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
    return state


def build():
    """Build the memory retrieval subgraph."""
    g = StateGraph(MemRetrievalState)

    g.add_node("summarize", summarize_node)
    g.add_node("embedding", embedding_node)
    g.add_node("retrieval", retrieval_node)

    g.add_edge(START, "summarize")
    g.add_edge("summarize", "embedding")
    g.add_edge("embedding", "retrieval")
    g.add_edge("retrieval", END)

    return g
