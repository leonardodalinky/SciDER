"""Build the writing agent graph.

Flow: START → init → agent_loop → generate_summary → END

No critic_review / user_review — the PaperOrchestra content-refinement-agent
skill already performs 3 rounds of simulated peer review inside the agent
loop, so an external critic pass would be redundant.
"""

from langgraph.graph import END, START, StateGraph
from loguru import logger

from scider.core.types import Message

from . import execute
from .state import WritingAgentState


def init_node(agent_state: WritingAgentState) -> WritingAgentState:
    """Inject the top-level task and a pointer to upstream materials."""
    logger.debug("init_node of WritingAgent")

    idea_preview = (agent_state.idea_summary or "").strip()
    if len(idea_preview) > 600:
        idea_preview = idea_preview[:600].rstrip() + "\n...[truncated]"

    exp_preview = (agent_state.experiment_summary or "").strip()
    if len(exp_preview) > 600:
        exp_preview = exp_preview[:600].rstrip() + "\n...[truncated]"

    content_parts = [
        "Run the full PaperOrchestra pipeline on the prepared paper workspace.",
        "",
        "# Task",
        agent_state.user_query or "(no user query provided)",
        "",
        "# Paper workspace",
        (
            f"Working directory: {agent_state.workspace.working_dir}\n"
            "The `inputs/` subdirectory already contains `idea.md`, `experimental_log.md`, "
            "`template.tex`, `conference_guidelines.md`, and (optionally) `figures/`. "
            "Do NOT overwrite them — they are your ground truth."
        ),
        "",
        "# Upstream idea (already copied to inputs/idea.md)",
        idea_preview or "(no idea summary provided)",
        "",
        "# Upstream experiment summary (source of inputs/experimental_log.md)",
        exp_preview or "(no experiment summary provided)",
        "",
        "Start by consulting the preloaded `paper-orchestra` skill, then walk through "
        "its six steps using the `Skill` tool to load each sub-skill as you reach it. "
        "Finish only when `final/paper.pdf` exists on disk.",
    ]
    content = "\n".join(content_parts)

    agent_state.add_message(
        Message(role="user", content=content, agent_sender="writing").with_log()
    )

    agent_state.intermediate_state.append(
        {"node_name": "init", "output": "Starting paper writing pipeline."}
    )
    return agent_state


@logger.catch
def build():
    """Build the writing agent graph."""
    g = StateGraph(WritingAgentState)

    g.add_node("init", init_node)
    g.add_node("agent_loop", execute.agent_loop_node)
    g.add_node("generate_summary", execute.generate_summary_node)

    g.add_edge(START, "init")
    g.add_edge("init", "agent_loop")
    g.add_edge("agent_loop", "generate_summary")
    g.add_edge("generate_summary", END)

    return g
