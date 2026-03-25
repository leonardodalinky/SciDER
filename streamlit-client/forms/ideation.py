"""Ideation workflow form and runner."""

import json

import streamlit as st

from scider.agents.ideation_agent.state import IdeationAgentState


def run_ideation(q, ideation_graph):
    """Run ideation workflow. Called from background thread."""
    s = IdeationAgentState(user_query=q)
    r = ideation_graph.invoke(s, {"recursion_limit": 50})
    rs = IdeationAgentState(**r)
    out = []
    if rs.output_summary:
        out.append("## Research Ideas Summary\n\n" + rs.output_summary)
    if rs.novelty_score is not None:
        out.append(
            "## Novelty Evaluation\n```json\n"
            + json.dumps(
                {
                    "novelty_score": rs.novelty_score,
                    "feedback": rs.novelty_feedback,
                },
                indent=2,
            )
            + "\n```"
        )
    if rs.research_ideas:
        out.append("## Generated Research Ideas\n")
        for i, idea in enumerate(rs.research_ideas[:5], 0):
            out.append(f"### {i}. {idea.get('title','')}\n{idea.get('description','')}")
    return ("\n\n".join(out) if out else "No result", rs.intermediate_state)


def render_form():
    """Render the ideation form. Returns workflow_config dict or None."""
    with st.form("ideation_form", clear_on_submit=True):
        st.markdown("### Ideation Workflow")
        topic = st.text_input("Research Topic", placeholder="Enter your research topic here...")
        submitted = st.form_submit_button("Run Ideation", type="primary")
        if submitted and topic:
            return {"type": "ideation", "query": topic}
    return None
