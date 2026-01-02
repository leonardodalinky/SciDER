from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute
from .state import ClaudeCodingAgentState


@logger.catch
def build():
    """Build the Claude coding agent graph.

    This is a minimal graph that delegates all coding work to Claude Agent SDK.
    Flow: START -> claude_node -> summary_node -> END

    Claude Agent SDK has its own internal planning and execution, so no external
    LLM chat loop or tool calling is needed.
    """
    g = StateGraph(ClaudeCodingAgentState)

    # Nodes - minimal: just OpenHands execution and summary
    g.add_node("claude", execute.claude_node)
    g.add_node("summary", execute.summary_node)

    # Simple linear flow
    g.add_edge(START, "claude")
    g.add_edge("claude", "summary")
    g.add_edge("summary", END)

    return g
