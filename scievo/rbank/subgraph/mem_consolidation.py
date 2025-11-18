"""
Memory consolidation subgraph
"""

from pathlib import Path
from typing import Tuple

from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel

from scievo.core.errors import AgentError
from scievo.core.llms import ModelRegistry
from scievo.rbank.memo import Memo, MemoEmbeddings

from .mem_extraction import LLM_NAME as EXTRACTION_LLM_NAME
from .mem_retrieval import LLM_NAME as RETRIEVAL_LLM_NAME

AGENT_NAME = "mem_consolidation"


MemPair = Tuple[Memo, MemoEmbeddings, str]


class MemConsolidationState(BaseModel):
    # dir of mems to consolidate (input)
    mem_dir: str | Path

    # save dirs (input & output)
    long_term_mem_dir: str | Path
    project_mem_dir: str | Path

    # intermediate mem pairs (output)
    mem_pairs_dict: dict[str, list[MemPair]] = {}
    long_term_mem_pairs: list[MemPair] = []
    project_mem_pairs: list[MemPair] = []


def _load_mem_pair(md_path: Path) -> MemPair:
    embed_path = md_path.with_suffix(".json")
    if not embed_path.exists():
        raise ValueError(f"embed_path does not exist: {embed_path}")

    # load memo
    memo = Memo.from_markdown_file(md_path)

    # load embedding
    try:
        embed = MemoEmbeddings.from_json_file(embed_path)
    except Exception as e:
        raise ValueError(f"Failed to load embedding from {embed_path}") from e

    base_name = md_path.name
    return (memo, embed, base_name)


def _identify_mem_type(md_path: Path) -> str:
    parts = md_path.name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid mem name: {md_path.name}")
    if "L" in parts:
        return "L"
    elif "P" in parts:
        return "P"
    else:
        raise ValueError(f"Unknown mem type: {md_path.name}")


def compute_embeddings_node(state: MemConsolidationState) -> MemConsolidationState:
    """Load memos from mem_dir and compute embeddings for them."""
    logger.debug("Memory Consolidation compute_embeddings begin: mem_dir={}", state.mem_dir)

    # Convert mem_dir to Path object
    mem_dir_path = Path(state.mem_dir)

    if not mem_dir_path.exists() or not mem_dir_path.is_dir():
        raise AgentError(
            f"mem_dir does not exist or is not a directory: {mem_dir_path}", agent_name=AGENT_NAME
        )

    # Find all markdown files in the directory
    md_files = list(mem_dir_path.glob("*.md"))

    if len(md_files) == 0:
        logger.debug("No markdown files found in mem_dir: {}", mem_dir_path)
        return state

    # long-term mems
    l_list: list[MemPair] = []
    # project mems
    p_list: list[MemPair] = []
    for md_path in md_files:
        try:
            mem_pair = _load_mem_pair(md_path)
        except Exception as e:
            logger.debug("Consolidation error: {}", e)
            raise AgentError(
                f"Failed to load mem pair from {md_path}", agent_name=AGENT_NAME
            ) from e

        if (typ := _identify_mem_type(md_path)) == "L":
            l_list.append(mem_pair)
        elif typ == "P":
            p_list.append(mem_pair)
        else:
            err_txt = f"Unknown mem type: {typ}"
            logger.debug("Consolidation error: {}", err_txt)
            raise AgentError(err_txt, agent_name=AGENT_NAME)

    state.mem_pairs_dict = {"L": l_list, "P": p_list}
    return state


def load_existing_mems_node(state: MemConsolidationState) -> MemConsolidationState:
    # load existing mem pairs from long_term_mem_dir and project_mem_dir
    long_term_mem_pairs: list[MemPair] = []
    project_mem_pairs: list[MemPair] = []

    # load long-term mems
    for md_path in Path(state.long_term_mem_dir).glob("*.md"):
        try:
            mem_pair = _load_mem_pair(md_path)
        except Exception as e:
            logger.debug("Consolidation error: {}", e)
            raise AgentError(
                f"Failed to load mem pair from {md_path}", agent_name=AGENT_NAME
            ) from e

        long_term_mem_pairs.append(mem_pair)

    # load project mems
    for md_path in Path(state.project_mem_dir).glob("*.md"):
        try:
            mem_pair = _load_mem_pair(md_path)
        except Exception as e:
            logger.debug("Consolidation error: {}", e)
            raise AgentError(
                f"Failed to load mem pair from {md_path}", agent_name=AGENT_NAME
            ) from e

        project_mem_pairs.append(mem_pair)

    state.long_term_mem_pairs = long_term_mem_pairs
    state.project_mem_pairs = project_mem_pairs
    return state


def merge_mems_node(state: MemConsolidationState) -> MemConsolidationState:
    Path(state.long_term_mem_dir).mkdir(parents=True, exist_ok=True)
    Path(state.project_mem_dir).mkdir(parents=True, exist_ok=True)

    # TODO: merge mem pairs. For now, just add the new mems to the existing mems
    # TODO: We may have more complex merging logic in the future
    for pair in state.mem_pairs_dict["L"]:
        memo, embed, base_name = pair
        memo.to_markdown_file(Path(state.long_term_mem_dir) / f"{base_name}.md")
        embed.to_json_file(Path(state.long_term_mem_dir) / f"{base_name}.json")

    for pair in state.mem_pairs_dict["P"]:
        memo, embed, base_name = pair
        memo.to_markdown_file(Path(state.project_mem_dir) / f"{base_name}.md")
        embed.to_json_file(Path(state.project_mem_dir) / f"{base_name}.json")

    return state


@logger.catch
def build():
    """Build the memory consolidation subgraph."""
    g = StateGraph(MemConsolidationState)

    g.add_node("compute_embeddings", compute_embeddings_node)
    g.add_node("load_existing_mems", load_existing_mems_node)
    g.add_node("merge_mems", merge_mems_node)

    g.add_edge(START, "compute_embeddings")
    g.add_edge("compute_embeddings", "load_existing_mems")
    g.add_edge("load_existing_mems", "merge_mems")
    g.add_edge("merge_mems", END)

    return g
