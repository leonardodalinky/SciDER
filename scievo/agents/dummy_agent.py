from langgraph.graph import END, START, StateGraph

from scievo.prompts import PROMPTS

from ..core.types import GraphState, Message

LLM_NAME = "dummy"
AGENT_NAME = "dummy"


def say_hello(graph_state: GraphState) -> GraphState:
    msg = Message(
        role="assistant",
        content="Hello",
        llm_sender=None,
        agent_sender=AGENT_NAME,
    )
    graph_state.agents[AGENT_NAME].data_msgs.append(msg)
    return graph_state


def build():
    g = StateGraph(GraphState)
    g.add_node("dummy1", say_hello)
    g.add_node("dummy2", say_hello)

    g.add_edge(START, "dummy1")
    g.add_edge("dummy1", "dummy2")
    g.add_edge("dummy2", END)

    return g
