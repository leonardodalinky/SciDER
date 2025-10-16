from langgraph.graph import END, START, StateGraph

from scievo_lg.prompts import PROMPTS

from ..llms import ModelRegistry
from ..types import AllState


def say_hello(state: AllState) -> AllState:
    msg = ModelRegistry.completion(
        "dummy",
        state.data_msgs,
        PROMPTS.dummy.dummy_prompt,
    )
    state.data_msgs.append(msg)
    return state


def graph():
    g = StateGraph(AllState)
    g.add_node("dummy1", say_hello)
    g.add_node("dummy2", say_hello)

    g.add_edge(START, "dummy1")
    g.add_edge("dummy1", "dummy2")
    g.add_edge("dummy2", END)

    return g
