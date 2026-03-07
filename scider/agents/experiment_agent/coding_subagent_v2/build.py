from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute
from .state import CodingAgentState


@logger.catch
def build():
    """Build the coding agent graph.

    This is a minimal graph that delegates all coding work to OpenHands SDK.
    Flow: START -> openhands_node -> summary_node -> END

    OpenHands has its own internal planning and execution, so no external
    LLM chat loop or tool calling is needed.
    """
    g = StateGraph(CodingAgentState)

    # Nodes - minimal: just OpenHands execution and summary
    g.add_node("openhands", execute.openhands_node)
    g.add_node("summary", execute.summary_node)

    # Simple linear flow
    g.add_edge(START, "openhands")
    g.add_edge("openhands", "summary")
    g.add_edge("summary", END)

    return g
