"""Build the ideation agent graph.

Flow: START → init → agent_loop → generate_report → END
"""

from langgraph.graph import END, START, StateGraph
from loguru import logger

from scider.core.types import Message

from . import execute
from .state import IdeationAgentState


def init_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    logger.debug("init_node of IdeationAgent")

    content = f"Generate novel research ideas for: {agent_state.user_query}"
    if agent_state.research_domain:
        content += f"\n\nResearch domain: {agent_state.research_domain}"

    agent_state.add_message(
        Message(role="user", content=content, agent_sender="ideation").with_log()
    )

    agent_state.intermediate_state.append({"node_name": "init", "output": "Starting ideation."})
    return agent_state


@logger.catch
def build():
    g = StateGraph(IdeationAgentState)

    g.add_node("init", init_node)
    g.add_node("agent_loop", execute.agent_loop_node)
    g.add_node("generate_report", execute.generate_report_node)

    g.add_edge(START, "init")
    g.add_edge("init", "agent_loop")
    g.add_edge("agent_loop", "generate_report")
    g.add_edge("generate_report", END)

    return g
