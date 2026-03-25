"""Case study viewer — browse saved workflow conversations without API keys."""

import json
from pathlib import Path

import streamlit as st

# Search paths for case-study-memory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CASE_STUDY_DIRS = [
    _PROJECT_ROOT / "case-study-memory",
    Path(__file__).resolve().parent.parent / "case-study-memory",
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


def render_case_study_viewer():
    """Render the case study browser UI."""
    st.title("SciDER — Case Studies")

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
    with open(selected_path, encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    st.divider()

    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Show workspace files if available
    case_dir = selected_path.parent
    ws_dir = case_dir / "workspace"
    browse_dir = ws_dir if ws_dir.exists() else case_dir
    files = [f for f in browse_dir.rglob("*") if f.is_file() and f.name != "chat_history.json"]
    if files:
        with st.expander(f"Workspace Files ({len(files)})", expanded=False):
            for f in sorted(files):
                rel = f.relative_to(browse_dir)
                st.text(str(rel))
