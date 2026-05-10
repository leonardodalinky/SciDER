"""Case study viewer — browse saved workflow conversations without API keys."""

import json
from pathlib import Path

import streamlit as st

# Search paths for saved conversations and case-study-memory examples
_APP_ROOT = Path(__file__).resolve().parent.parent  # streamlit-client/
_PROJECT_ROOT = _APP_ROOT.parent                    # SciDER/
_CASE_STUDY_DIRS = [
    _APP_ROOT / "saved_chats",           # conversations saved by the app's own workflows
    _PROJECT_ROOT / "case-study-memory", # pre-bundled example case studies
    _APP_ROOT / "case-study-memory",
]


def list_case_studies() -> list[tuple[Path, dict]]:
    """List all chat_history.json files in case-study-memory directories."""
    results = []
    seen = set()
    for base in _CASE_STUDY_DIRS:
        if not base.exists():
            continue
        dirs = sorted(
            [d for d in base.iterdir() if d.is_dir() and (d / "chat_history.json").exists()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        for memo_dir in dirs:
            chat_file = memo_dir / "chat_history.json"
            key = str(chat_file.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                with open(chat_file, encoding="utf-8") as f:
                    data = json.load(f)
                ts = data.get("timestamp", "")[:19].replace("T", " ")
                wf = data.get("workflow_type", "unknown")
                query = data.get("metadata", {}).get("query", "")
                label = (query[:80] + "...") if len(query) > 83 else (query or memo_dir.name)
                results.append((chat_file, {"label": label, "timestamp": ts, "workflow_type": wf}))
            except Exception:
                results.append(
                    (
                        chat_file,
                        {"label": memo_dir.name, "timestamp": "", "workflow_type": "unknown"},
                    )
                )
    return results


_WORKFLOW_LABELS = {
    "ideation": "Ideation",
    "data": "Data Analysis",
    "experiment": "Experiment",
    "full": "Full Pipeline",
}


def render_case_study_viewer():
    """Render the case study browser UI."""
    st.markdown(
        """<div class='scider-header'>
        <span class='scider-header-icon'>📚</span>
        <div>
            <div class='scider-header-title'>Case Studies</div>
            <div class='scider-header-sub'>Browse saved workflow conversations and workspace artifacts</div>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

    available = list_case_studies()
    if not available:
        st.info("No case studies found. Run a workflow first to save conversations.")
        return

    options = [f"{m['label']}  ({m['workflow_type']}, {m['timestamp']})" for _, m in available]
    idx = st.selectbox(
        "Select a case study",
        range(len(options)),
        format_func=lambda i: options[i],
        key="case_study_select",
    )

    selected_path = available[idx][0]
    selected_meta = available[idx][1]
    with open(selected_path, encoding="utf-8") as f:
        data = json.load(f)

    # Metadata card
    wf_type = selected_meta["workflow_type"]
    wf_label = _WORKFLOW_LABELS.get(wf_type, wf_type.title())
    query = data.get("metadata", {}).get("query", selected_meta["label"])
    ts = selected_meta["timestamp"]
    st.markdown(
        f"""<div class='approval-card'>
        <p class='approval-card-title'>{query}</p>
        <p class='approval-card-subtitle'>{wf_label} &nbsp;·&nbsp; {ts}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Workflow type badge + message count
    msg_count = len(data.get("messages", []))
    st.markdown(
        f"<span class='chat-agent-badge'>{wf_label}</span>&nbsp;&nbsp;"
        f"<span style='color:#64748b;font-size:0.85rem'>{msg_count} messages</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    messages = data.get("messages", [])
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Show workspace files if available
    from components.file_browser import render_file_browser, render_workspace_download

    case_dir = selected_path.parent
    ws_dir = case_dir / "workspace"
    browse_dir = ws_dir if ws_dir.exists() else case_dir
    render_file_browser(browse_dir, key_prefix="cs_fb")
    render_workspace_download(browse_dir, key_prefix="cs_dl")
