"""
Memory consolidation subgraph
"""

from pathlib import Path

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from scievo.rbank.memo import MemEntry

LLM_NAME = "mem"
AGENT_NAME = "mem_consolidation"


class MemConsolidationState(BaseModel):
    # dir of mems to consolidate
    mem_dir: str | Path

    # save dirs
    long_term_save_dir: str | Path
    project_save_dir: str | Path

    output_error: str | None = None

    # intermidiate results
    long_term_mems: list[MemEntry] = []
    project_mems: list[MemEntry] = []


def mem_consolidation_node(state: MemConsolidationState) -> MemConsolidationState:
    raise NotImplementedError


def persistence_node(state: MemConsolidationState) -> MemConsolidationState:
    raise NotImplementedError


def build():
    g = StateGraph(MemConsolidationState)
    raise NotImplementedError
