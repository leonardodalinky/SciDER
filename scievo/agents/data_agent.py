"""
Agent for data understanding and processing
"""

from langgraph.graph import END, START, StateGraph
from scievo_lg.prompts import PROMPTS

from ..llms import ModelRegistry
from ..tools.dummy_tool import dummy_func
from ..types import AllState


def main_loop(state: AllState) -> AllState:
    msg = ModelRegistry.completion(
        "data",
        state.data_msgs,
        PROMPTS.data.system_prompt,
        tools=[
            dummy_func,
        ],
    )
    state.data_msgs.append(msg)
    return state


def build():
    g = StateGraph(AllState)

    # nodes
    g.add_node("main_loop", main_loop)

    # edges
    g.add_edge(START, "main_loop")
    g.add_edge("main_loop", END)

    return g
