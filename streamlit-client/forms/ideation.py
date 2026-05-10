"""Ideation workflow form and runner."""

import json

import streamlit as st

from scider.agents.ideation_agent.state import IdeationAgentState


def run_ideation(wc, ideation_graph):
    """Run ideation workflow. Called from background thread."""
    query = wc.get("query") if isinstance(wc, dict) else wc
    idea_search_enabled = wc.get("idea_search_enabled", True) if isinstance(wc, dict) else True
    max_idea_search_calls = wc.get("max_idea_search_calls", 60) if isinstance(wc, dict) else 60

    s = IdeationAgentState(
        user_query=query,
        idea_search_enabled=idea_search_enabled,
        max_idea_search_calls=max_idea_search_calls,
    )
    r = ideation_graph.invoke(s, {"recursion_limit": 50})
    rs = IdeationAgentState(**r)
    out = []
    if rs.output_summary:
        out.append("## Research Ideas Summary\n\n" + rs.output_summary)

    # Only show the structured idea list when evolutionary search ran and produced
    # a different (evolved) set — otherwise it duplicates what's already in the summary.
    if rs.research_ideas and rs.idea_search_result:
        search_meta = rs.idea_search_result
        header = (
            f"## Evolved Research Ideas "
            f"({search_meta.get('iterations_completed', '?')} iterations, "
            f"{search_meta.get('llm_calls_used', '?')} LLM calls"
            + (" — budget reached" if search_meta.get('search_budget_hit') else "")
            + ")"
        )
        out.append(header)
        for i, idea in enumerate(rs.research_ideas[:5], 1):
            score = idea.get("composite_score") or idea.get("novelty_score")
            score_str = f" — score: {score:.2f}" if score is not None else ""
            out.append(f"### {i}. {idea.get('title', '')}{score_str}\n{idea.get('description', '')}")

    return ("\n\n".join(out) if out else "No result", rs.intermediate_state)


def render_form():
    """Render the ideation form. Returns workflow_config dict or None."""
    with st.form("ideation_form", clear_on_submit=True):
        st.markdown("### Generate Research Ideas")
        st.caption(
            "SciDER searches recent literature, generates seed ideas, then uses evolutionary "
            "search to improve and combine them — scoring each idea on novelty, feasibility, "
            "impact, and specificity."
        )
        topic = st.text_input(
            "Research Topic",
            placeholder="e.g. Efficient fine-tuning of large language models for low-resource languages",
        )
        idea_search = st.checkbox(
            "Enable evolutionary idea search",
            value=True,
            help=(
                "After generating seed ideas, runs a best-first search that improves and combines "
                "ideas across novelty, feasibility, impact, and specificity. Uses ~32 extra LLM calls."
            ),
        )
        max_calls = st.number_input(
            "Max LLM calls for idea search",
            min_value=8,
            max_value=200,
            value=60,
            step=4,
            help="Hard budget cap. ~32 calls for 3 iterations; increase for deeper search.",
            disabled=not idea_search,
        )
        submitted = st.form_submit_button("Generate Ideas")
        if submitted and topic:
            return {
                "type": "ideation",
                "query": topic,
                "idea_search_enabled": idea_search,
                "max_idea_search_calls": int(max_calls),
            }
    return None
