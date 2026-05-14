"""Ideation workflow form and runner."""

import html as _html

import streamlit as st

from scider.agents.ideation_agent.state import IdeationAgentState

# --- Rich result rendering -------------------------------------------------
# The result is delivered as a single chat message string that st.markdown
# renders with unsafe_allow_html=True. That means the string is run through a
# markdown pass, so any HTML block must contain NO blank lines and any
# LLM-authored text embedded in HTML must have its markdown specials
# neutralized — see _md_safe.

_DIM_LABELS = [
    ("score_novelty", "Novelty"),
    ("score_feasibility", "Feasibility"),
    ("score_impact", "Impact"),
    ("score_specificity", "Specificity"),
]

# operator -> (label, text color, background color)
_OP_BADGES = {
    "seed": ("🌱 seed", "#475569", "#f1f5f9"),
    "improve": ("⬆ improved", "#3a0ca3", "#eef0fd"),
    "combine": ("⚯ combined", "#6d28d9", "#f3e8ff"),
}


def _md_safe(text) -> str:
    """Make LLM text safe to embed inside an HTML block rendered by st.markdown.

    Collapses all whitespace (so no blank line can break the HTML block),
    HTML-escapes, then converts markdown special characters to HTML entities
    so the markdown pass can't italicize / codify / header-ify the text.
    """
    s = " ".join(str(text).split())
    s = _html.escape(s)
    # '#' must be replaced first — every numeric HTML entity below contains a
    # '#', so doing it later would corrupt the entities just inserted.
    for ch, ent in (
        ("#", "&#35;"),
        ("*", "&#42;"),
        ("_", "&#95;"),
        ("`", "&#96;"),
        ("~", "&#126;"),
        ("[", "&#91;"),
        ("]", "&#93;"),
        ("|", "&#124;"),
    ):
        s = s.replace(ch, ent)
    return s


def _score_color(v: float) -> str:
    if v >= 0.7:
        return "#16a34a"  # green
    if v >= 0.5:
        return "#f59e0b"  # amber
    return "#94a3b8"  # gray


def _score_bar(label: str, val) -> str:
    if val is None:
        return (
            f"<div class='sc-row'><span class='sc-label'>{label}</span>"
            f"<span class='sc-track'></span><span class='sc-val'>—</span></div>"
        )
    pct = max(0, min(100, round(float(val) * 100)))
    color = _score_color(float(val))
    return (
        f"<div class='sc-row'><span class='sc-label'>{label}</span>"
        f"<span class='sc-track'><span class='sc-fill' style='width:{pct}%;background:{color}'></span></span>"
        f"<span class='sc-val'>{float(val):.2f}</span></div>"
    )


def _idea_card(rank: int, idea: dict) -> str:
    """Render one ranked, scored idea as a single-line HTML card."""
    title = _md_safe(idea.get("title") or "Untitled idea")
    desc = _md_safe(idea.get("description") or "")
    composite = idea.get("composite_score")
    operator = idea.get("search_operator", "seed")
    op_label, op_fg, op_bg = _OP_BADGES.get(operator, _OP_BADGES["seed"])

    has_scores = any(idea.get(k) is not None for k, _ in _DIM_LABELS)
    if composite is not None:
        comp_str = f"{float(composite):.2f}"
        comp_color = _score_color(float(composite))
    else:
        comp_str, comp_color = "—", "#94a3b8"

    head = (
        "<div class='idea-card-head'>"
        f"<span class='idea-rank'>#{rank}</span>"
        f"<span class='idea-title'>{title}</span>"
        f"<span class='idea-op' style='color:{op_fg};background:{op_bg}'>{op_label}</span>"
        f"<span class='idea-composite' style='color:{comp_color}'>{comp_str}"
        "<small>composite</small></span>"
        "</div>"
    )

    bars = ""
    if has_scores:
        bars = "<div class='sc-bars'>" + "".join(
            _score_bar(lbl, idea.get(key)) for key, lbl in _DIM_LABELS
        ) + "</div>"

    extra = ""
    for key, lbl in (
        ("experiment", "Experiment"),
        ("contribution", "Contribution"),
    ):
        v = idea.get(key)
        if v:
            extra += (
                f"<div class='idea-extra'><span class='idea-extra-lbl'>{lbl}</span> "
                f"{_md_safe(v)}</div>"
            )

    return (
        "<div class='idea-card'>"
        + head
        + f"<div class='idea-desc'>{desc}</div>"
        + bars
        + extra
        + "</div>"
    )


def _simple_idea_card(rank: int, idea: dict) -> str:
    """Render an unscored idea (no evolutionary search) — title, desc, novelty."""
    title = _md_safe(idea.get("title") or "Untitled idea")
    desc = _md_safe(idea.get("description") or "")
    novelty = idea.get("novelty_score")
    badge = ""
    if novelty is not None:
        try:
            badge = f"<span class='idea-novelty'>novelty {float(novelty):.1f}/10</span>"
        except (TypeError, ValueError):
            badge = ""
    return (
        "<div class='idea-card'>"
        "<div class='idea-card-head'>"
        f"<span class='idea-rank'>#{rank}</span>"
        f"<span class='idea-title'>{title}</span>"
        f"{badge}"
        "</div>"
        f"<div class='idea-desc'>{desc}</div>"
        "</div>"
    )


def _search_metrics_panel(meta: dict, ideas: list[dict]) -> str:
    """Render the evolutionary-search metrics strip as a single HTML block."""
    iters = meta.get("iterations_completed", "?")
    calls = meta.get("llm_calls_used", "?")
    budget_hit = meta.get("search_budget_hit", False)
    initial = meta.get("initial_best_composite")
    best = max((i.get("composite_score") or 0.0 for i in ideas), default=0.0)

    # each item: (label, value, value-color or None)
    items = [
        ("Iterations", str(iters), None),
        ("LLM Calls", f"{calls}{' ⚠' if budget_hit else ''}", "#b45309" if budget_hit else None),
        ("Best Composite", f"{best:.2f}", _score_color(best)),
    ]
    if initial is not None and initial > 0:
        lift = best - initial
        items.append(
            (
                "Score Lift",
                f"{'+' if lift >= 0 else ''}{lift:.2f}",
                "#16a34a" if lift >= 0 else "#dc2626",
            )
        )

    cells = ""
    for lbl, v, color in items:
        style = f" style='color:{color}'" if color else ""
        cells += (
            f"<div class='sm-item'><div class='sm-val'{style}>{v}</div>"
            f"<div class='sm-lbl'>{lbl}</div></div>"
        )
    return f"<div class='search-metrics'>{cells}</div>"


def _build_ideation_output(rs: IdeationAgentState) -> str:
    """Assemble the rich ideation result message (HTML cards + markdown report)."""
    parts: list[str] = []

    if rs.research_ideas and rs.idea_search_result:
        # Evolutionary search ran — show metrics + fully-scored ranked cards.
        ideas = rs.research_ideas[:8]
        cards = "".join(_idea_card(i, idea) for i, idea in enumerate(ideas, 1))
        parts.append(
            "<div class='ideation-result'>"
            "<div class='ir-section-title'>🧬 Evolutionary Idea Search</div>"
            + _search_metrics_panel(rs.idea_search_result, rs.research_ideas)
            + "<div class='ir-section-title'>🏆 Ranked Research Ideas</div>"
            + cards
            + "</div>"
        )
    elif rs.research_ideas:
        # No evolutionary search — render the extracted ideas as simple cards.
        ideas = rs.research_ideas[:8]
        cards = "".join(_simple_idea_card(i, idea) for i, idea in enumerate(ideas, 1))
        parts.append(
            "<div class='ideation-result'>"
            "<div class='ir-section-title'>💡 Research Ideas</div>"
            + cards
            + "</div>"
        )

    if rs.output_summary:
        parts.append("---\n\n## 📄 Detailed Ideation Report\n\n" + rs.output_summary)

    return "\n\n".join(parts) if parts else "No result"


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

    return (_build_ideation_output(rs), rs.intermediate_state)


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
