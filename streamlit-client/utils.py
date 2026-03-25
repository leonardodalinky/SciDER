"""Shared utilities for the Streamlit app."""

import json
import shutil
import tempfile
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import streamlit as st


def stream_markdown(text, delay=0.02):
    buf = ""
    slot = st.empty()
    for line in text.split("\n"):
        buf += line + "\n"
        slot.markdown(buf)
        time.sleep(delay)


def render_intermediate_state(intermediate_state):
    if not intermediate_state:
        return
    by_node = defaultdict(list)
    for item in intermediate_state:
        by_node[item.get("node_name", "unknown")].append(item.get("output", ""))

    st.divider()
    st.subheader("Intermediate States")
    for node, outputs in by_node.items():
        with st.expander(node, expanded=False):
            for i, content in enumerate(outputs, 1):
                st.markdown(f"**Step {i}**")
                st.markdown(content)


# --- File upload helpers ---


def get_upload_temp_dir() -> Path:
    """Return temp directory for uploaded files. Clean old dirs on startup."""
    base = Path(tempfile.gettempdir()) / "scider_uploads"
    base.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for d in base.iterdir():
        if d.is_dir() and (now - d.stat().st_mtime) > 3600:
            try:
                shutil.rmtree(d)
            except OSError:
                pass
    return base


def save_and_extract_upload(uploaded_file) -> Path | None:
    """Save uploaded zip to temp dir, extract it, return path to extracted dir."""
    if uploaded_file is None or not uploaded_file.name.lower().endswith(".zip"):
        return None
    base = get_upload_temp_dir()
    dest_dir = Path(tempfile.mkdtemp(dir=base))
    zip_path = dest_dir / uploaded_file.name
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    extract_dir = dest_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()
    return extract_dir.resolve()


def find_data_analysis_file(extract_dir: Path) -> Path | None:
    """Find data_analysis.md in extracted dir (root or first subdir)."""
    candidates = [extract_dir / "data_analysis.md", extract_dir / "analysis.md"]
    for c in candidates:
        if c.exists():
            return c
    for p in extract_dir.rglob("data_analysis.md"):
        return p
    for p in extract_dir.rglob("analysis.md"):
        return p
    return None


def _rm_upload_root(p: Path):
    """Remove the scider_uploads session dir (go up to find it)."""
    cur = Path(p).resolve().parent if Path(p).resolve().is_file() else Path(p).resolve()
    while cur != cur.parent:
        parent = cur.parent
        if parent.name == "scider_uploads":
            try:
                shutil.rmtree(cur)
            except OSError:
                pass
            return
        cur = parent


def cleanup_uploaded_data():
    """Remove temp uploaded data and restore workspace_path to default."""
    for key in ("uploaded_data_path", "uploaded_experiment_path", "uploaded_full_data_path"):
        path = st.session_state.get(key)
        if path and isinstance(path, (str, Path)):
            _rm_upload_root(Path(path))
            if key in st.session_state:
                del st.session_state[key]
    if "default_workspace_path" in st.session_state:
        st.session_state.workspace_path = st.session_state.default_workspace_path


# --- Chat history ---


def get_next_memo_number(memory_dir: Path) -> int:
    if not memory_dir.exists():
        return 1
    existing_memos = [
        d.name for d in memory_dir.iterdir() if d.is_dir() and d.name.startswith("memo_")
    ]
    if not existing_memos:
        return 1
    numbers = []
    for memo in existing_memos:
        try:
            num = int(memo.replace("memo_", ""))
            numbers.append(num)
        except ValueError:
            continue
    return max(numbers) + 1 if numbers else 1


def save_chat_history(messages: list, workflow_type: str, metadata: dict = None):
    base_dir = Path(__file__).parent / "case-study-memory"
    base_dir.mkdir(parents=True, exist_ok=True)
    memo_number = get_next_memo_number(base_dir)
    memo_dir = base_dir / f"memo_{memo_number}"
    memo_dir.mkdir(parents=True, exist_ok=True)
    chat_data = {
        "timestamp": datetime.now().isoformat(),
        "workflow_type": workflow_type,
        "metadata": metadata or {},
        "messages": messages,
    }
    chat_file = memo_dir / "chat_history.json"
    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)
    return memo_dir
