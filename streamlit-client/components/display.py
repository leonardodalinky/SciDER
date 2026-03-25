"""
Display Components for Streamlit Interface

Reusable components for displaying workflow outputs.
"""

import streamlit as st
from workflow.monitor import PhaseType, ProgressUpdate


def display_phase_header(phase_name: str, status: str = "running", icon: str = "🔄"):
    """Display a phase header with status."""
    status_colors = {
        "running": "🔄",
        "completed": "✅",
        "error": "❌",
        "pending": "⏳",
    }
    status_icon = status_colors.get(status, "🔄")
    st.markdown(f"### {icon} {phase_name} {status_icon}")


def display_ideation_progress(workflow_data: dict):
    """Display ideation agent progress."""
    with st.expander("💡 Research Ideation", expanded=True):
        cols = st.columns(4)

        with cols[0]:
            papers_count = len(workflow_data.get("ideation_papers", []))
            st.metric("Papers Found", papers_count)

        with cols[1]:
            ideas_count = len(workflow_data.get("research_ideas", []))
            st.metric("Ideas Generated", ideas_count)

        with cols[2]:
            novelty = workflow_data.get("novelty_score")
            if novelty:
                st.metric("Novelty Score", f"{novelty:.1f}/10")
            else:
                st.metric("Novelty Score", "Pending")

        with cols[3]:
            status = workflow_data.get("ideation_status", "Running")
            st.info(status)

        # Show ideation summary if available
        if workflow_data.get("ideation_summary"):
            st.markdown("**Summary:**")
            st.markdown(workflow_data["ideation_summary"][:500] + "...")

        # Show papers
        if workflow_data.get("ideation_papers"):
            with st.expander("📚 Papers Reviewed", expanded=False):
                for i, paper in enumerate(workflow_data["ideation_papers"][:5], 1):
                    st.markdown(f"**{i}. {paper.get('title', 'Untitled')}**")
                    if paper.get("abstract"):
                        st.caption(paper["abstract"][:200] + "...")


def display_data_agent_progress(workflow_data: dict):
    """Display data agent progress."""
    with st.expander("📊 Data Analysis", expanded=True):
        cols = st.columns(4)

        with cols[0]:
            papers_count = len(workflow_data.get("papers", []))
            st.metric("Papers Found", papers_count)

        with cols[1]:
            datasets_count = len(workflow_data.get("datasets", []))
            st.metric("Datasets Found", datasets_count)

        with cols[2]:
            metrics_count = len(workflow_data.get("metrics", []))
            st.metric("Metrics Found", metrics_count)

        with cols[3]:
            status = workflow_data.get("data_status", "Running")
            st.info(status)

        # Show data summary if available
        if workflow_data.get("data_summary"):
            st.markdown("**Data Summary:**")
            st.markdown(workflow_data["data_summary"][:500] + "...")

        # Show paper search summary
        if workflow_data.get("paper_search_summary"):
            with st.expander("🔍 Paper Search Summary", expanded=False):
                st.markdown(workflow_data["paper_search_summary"])


def display_experiment_progress(workflow_data: dict):
    """Display experiment agent progress."""
    with st.expander("🧪 Experiment Execution", expanded=True):
        # Revision info
        current_rev = workflow_data.get("current_revision", 0)
        max_rev = workflow_data.get("max_revisions", 5)
        current_phase = workflow_data.get("current_phase", "init")

        cols = st.columns(3)
        with cols[0]:
            st.metric("Current Revision", f"{current_rev + 1}/{max_rev}")
        with cols[1]:
            st.metric("Current Phase", current_phase)
        with cols[2]:
            results_count = len(workflow_data.get("execution_results", []))
            st.metric("Execution Results", results_count)

        # Show revision summaries
        if workflow_data.get("revision_summaries"):
            with st.expander("📝 Revision History", expanded=False):
                for i, summary in enumerate(workflow_data["revision_summaries"], 1):
                    st.markdown(f"**Revision {i}:**")
                    st.markdown(summary[:300] + "...")
                    st.markdown("---")


def display_progress_updates(updates: list[ProgressUpdate]):
    """Display a timeline of progress updates."""
    with st.expander("📋 Progress Log", expanded=False):
        for update in reversed(updates[-20:]):  # Show last 20 updates
            timestamp = time.strftime("%H:%M:%S", time.localtime(update.timestamp))

            # Status icon
            status_icons = {
                "started": "🚀",
                "progress": "⏳",
                "completed": "✅",
                "error": "❌",
            }
            icon = status_icons.get(update.status, "ℹ️")

            # Phase name
            phase_names = {
                PhaseType.IDEATION_LITERATURE_SEARCH: "Literature Search",
                PhaseType.IDEATION_ANALYZE_PAPERS: "Analyzing Papers",
                PhaseType.IDEATION_GENERATE_IDEAS: "Generating Ideas",
                PhaseType.IDEATION_NOVELTY_CHECK: "Novelty Check",
                PhaseType.IDEATION_REPORT: "Ideation Report",
                PhaseType.DATA_PLANNING: "Data Planning",
                PhaseType.DATA_EXECUTION: "Data Execution",
                PhaseType.DATA_PAPER_SEARCH: "Paper Search",
                PhaseType.DATA_FINALIZE: "Data Finalize",
                PhaseType.EXPERIMENT_INIT: "Experiment Init",
                PhaseType.EXPERIMENT_CODING: "Coding",
                PhaseType.EXPERIMENT_EXEC: "Execution",
                PhaseType.EXPERIMENT_SUMMARY: "Summary",
                PhaseType.EXPERIMENT_ANALYSIS: "Analysis",
                PhaseType.EXPERIMENT_REVISION: "Revision",
                PhaseType.COMPLETE: "Complete",
                PhaseType.ERROR: "Error",
            }
            phase_name = phase_names.get(update.phase, str(update.phase))

            st.markdown(f"`{timestamp}` {icon} **{phase_name}**: {update.message}")


def display_final_results(workflow):
    """Display final workflow results."""
    st.markdown("## 📊 Final Results")

    # Overall status
    status_icon = "✅" if workflow.final_status == "success" else "❌"
    st.markdown(f"### {status_icon} Status: {workflow.final_status}")

    # Tabs for different result sections
    tabs = st.tabs(["Summary", "Ideation", "Data Analysis", "Experiments", "Raw Data"])

    with tabs[0]:
        st.markdown("### Complete Summary")
        st.markdown(workflow.final_summary)

    with tabs[1]:
        st.markdown("### Research Ideation Results")
        if workflow.ideation_summary:
            st.markdown(workflow.ideation_summary)

            if workflow.novelty_score:
                st.metric("Novelty Score", f"{workflow.novelty_score:.2f}/10")

            if workflow.novelty_feedback:
                st.markdown("**Novelty Feedback:**")
                st.info(workflow.novelty_feedback)

            if workflow.ideation_papers:
                st.markdown(f"**Papers Reviewed:** {len(workflow.ideation_papers)}")
                with st.expander("Show Papers"):
                    st.json(workflow.ideation_papers)

    with tabs[2]:
        st.markdown("### Data Analysis Results")
        if workflow.data_summary:
            st.markdown(workflow.data_summary)

            cols = st.columns(3)
            with cols[0]:
                st.metric("Papers Found", len(workflow.papers))
            with cols[1]:
                st.metric("Datasets Found", len(workflow.datasets))
            with cols[2]:
                st.metric("Metrics Found", len(workflow.metrics))

            if workflow.paper_search_summary:
                with st.expander("Paper Search Summary"):
                    st.markdown(workflow.paper_search_summary)

    with tabs[3]:
        st.markdown("### Experiment Execution Results")
        if workflow.execution_results:
            st.markdown(f"**Total Executions:** {len(workflow.execution_results)}")

            for i, result in enumerate(workflow.execution_results, 1):
                with st.expander(f"Execution {i}"):
                    st.json(result)

    with tabs[4]:
        st.markdown("### Raw Workflow Data")
        st.json(
            {
                "current_phase": workflow.current_phase,
                "final_status": workflow.final_status,
                "workspace_path": str(workflow.workspace_path),
                "session_name": workflow.session_name,
                "run_data_workflow": workflow.run_data_workflow,
                "run_experiment_workflow": workflow.run_experiment_workflow,
            }
        )


import time

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


def render_approval_ui(handler) -> None:
    """Render approval buttons + feedback input for a pending approval request."""
    pending = handler.get_pending()
    if not pending:
        return

    st.warning(f"**Approval required: `{pending['node_name']}`**")
    st.markdown(pending["summary"])

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
            st.caption(selected_item["description"][:300])

    feedback_text = st.text_input(
        "Feedback (required for feedback option)",
        key="approval_feedback_input",
        placeholder="Enter your feedback here...",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        approve_label = "Approve selected" if selected_index is not None else "Approve"
        if st.button(approve_label, type="primary", key="btn_approve"):
            handler.submit_response(
                ApprovalResponse(ApprovalResult.APPROVED, selected_index=selected_index)
            )
            st.rerun()
    with col2:
        if st.button("Reject", key="btn_reject"):
            handler.submit_response(ApprovalResponse(ApprovalResult.REJECTED))
            st.rerun()
    with col3:
        if st.button(
            "Feedback",
            key="btn_feedback",
            disabled=not (feedback_text and feedback_text.strip()),
        ):
            handler.submit_response(
                ApprovalResponse(ApprovalResult.FEEDBACK, feedback=feedback_text.strip())
            )
            st.rerun()
