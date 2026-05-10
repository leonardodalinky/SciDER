"""
Display Components for Streamlit Interface

Reusable components for displaying workflow outputs.
"""

import time

import streamlit as st

from scider.core.approval import ApprovalResponse, ApprovalResult


def render_live_intermediate(items: list[dict]) -> None:
    """Render intermediate state items as they arrive (real-time)."""
    if not items:
        return
    for item in items:
        node = item.get("node_name", "unknown")
        output = item.get("output", "")
        with st.status(f"Node: {node}", state="complete"):
            st.markdown(output[:800] if output else "(no output)")


_APPROVAL_CONTENT_TRUNCATE = 800


def render_approval_ui(handler) -> None:
    """Render approval buttons + feedback input for a pending approval request."""
    pending = handler.get_pending()
    if not pending:
        return

    summary = pending["summary"]
    node_name = pending["node_name"]
    title = pending.get("title", "")

    # Approval card header
    title_html = (
        f"<p class='approval-card-subtitle'>{title}</p>"
        if title
        else ""
    )
    st.markdown(
        f"""<div class='approval-card'>
        <p class='approval-card-title'>Review required: {node_name}</p>
        {title_html}
        </div>""",
        unsafe_allow_html=True,
    )

    # Approval content in a bordered container, collapsible if long
    if len(summary) > _APPROVAL_CONTENT_TRUNCATE:
        preview_lines = summary[:200].split("\n")
        label = " | ".join(line.strip() for line in preview_lines if line.strip())[:200]
        with st.expander(f"{label}...", expanded=True):
            st.markdown(summary)
    else:
        with st.container(border=True):
            st.markdown(summary)

    # Selection UI (radio buttons for single select)
    selected_index = None
    if pending.get("has_selection") and pending.get("items"):
        items = pending["items"]
        options = [f"{it.get('title', f'Item {i + 1}')}" for i, it in enumerate(items)]
        selected_label = st.radio("Select an idea:", options, key="idea_selection")
        selected_index = options.index(selected_label)

        # Show description of selected item
        selected_item = items[selected_index]
        if selected_item.get("description"):
            st.caption(selected_item["description"])

    # Use a counter-based key so each new approval request gets a fresh input
    approval_count = st.session_state.get("_approval_count", 0)
    feedback_text = st.text_input(
        "Feedback (required for feedback option)",
        key=f"approval_feedback_input_{approval_count}",
        placeholder="Enter your feedback here...",
    )

    # Buttons — use_container_width for equal sizing
    col1, col2, col3 = st.columns(3)
    with col1:
        label = "Approve selected" if selected_index is not None else "Approve"
        if st.button(f"✅ {label}", key="btn_approve", use_container_width=True):
            st.session_state.messages.append(
                {"role": "user", "content": f"✅ Approved [{node_name}]"}
            )
            st.session_state["_approval_count"] = approval_count + 1
            handler.submit_response(
                ApprovalResponse(ApprovalResult.APPROVED, selected_index=selected_index)
            )
            st.rerun()
    with col2:
        if st.button("❌ Reject", key="btn_reject", use_container_width=True):
            st.session_state.messages.append(
                {"role": "user", "content": f"❌ Rejected [{node_name}]"}
            )
            st.session_state["_approval_count"] = approval_count + 1
            handler.submit_response(ApprovalResponse(ApprovalResult.REJECTED))
            st.rerun()
    with col3:
        if st.button(
            "💬 Feedback",
            key="btn_feedback",
            use_container_width=True,
            disabled=not (feedback_text and feedback_text.strip()),
        ):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": f"💬 Feedback [{node_name}]: {feedback_text.strip()}",
                }
            )
            st.session_state["_approval_count"] = approval_count + 1
            handler.submit_response(
                ApprovalResponse(ApprovalResult.FEEDBACK, feedback=feedback_text.strip())
            )
            st.rerun()
