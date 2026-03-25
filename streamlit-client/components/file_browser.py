"""Workspace file browser and ZIP download components."""

import io
import zipfile
from pathlib import Path

import streamlit as st

MAX_PREVIEW_SIZE = 500_000  # 500KB
MAX_ZIP_FILE_SIZE = 2 * 1024 * 1024  # 2MB
MAX_DISPLAY_FILES = 200

EXCLUDED_DIRS = {
    ".venv",
    "__pycache__",
    "node_modules",
    ".git",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".ruff_cache",
}
EXCLUDED_FILES = {"uv.lock", ".DS_Store", "Thumbs.db"}

PREVIEWABLE_EXT = {
    ".py",
    ".js",
    ".ts",
    ".sh",
    ".sql",
    ".html",
    ".css",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".xml",
    ".cfg",
    ".ini",
    ".txt",
    ".md",
    ".rst",
    ".csv",
    ".tsv",
    ".log",
    ".ipynb",
    ".r",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}

LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".sh": "bash",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".css": "css",
    ".toml": "toml",
    ".sql": "sql",
    ".xml": "xml",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".r": "r",
}


def _collect_files(root: Path) -> list[Path]:
    """Collect visible, non-excluded files under root."""
    if not root.exists() or not root.is_dir():
        return []
    result = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        # Skip hidden files
        if f.name.startswith("."):
            continue
        # Skip files in excluded directories
        rel = f.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts[:-1]):
            continue
        # Skip excluded file names
        if f.name in EXCLUDED_FILES:
            continue
        result.append(f)
    return sorted(result, key=lambda p: p.relative_to(root))


def _format_size(n: int) -> str:
    """Human-readable file size."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / (1024 * 1024):.1f} MB"


def _render_preview(file_path: Path, key_prefix: str):
    """Render file preview with syntax highlighting."""
    ext = file_path.suffix.lower()

    if st.button("✕ Close preview", key=f"{key_prefix}_close"):
        del st.session_state[f"{key_prefix}_preview"]
        st.rerun()

    st.caption(f"📄 {file_path.name} ({_format_size(file_path.stat().st_size)})")

    if ext not in PREVIEWABLE_EXT:
        st.warning("Binary file — preview not available.")
        return

    if file_path.stat().st_size > MAX_PREVIEW_SIZE:
        st.warning(f"File too large to preview ({_format_size(file_path.stat().st_size)}).")
        return

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        st.error(f"Cannot read file: {e}")
        return

    if ext == ".json":
        try:
            import json

            st.json(json.loads(content))
        except Exception:
            st.code(content, language="json")
    elif ext == ".md":
        st.markdown(content)
    elif ext in (".csv", ".tsv"):
        try:
            import pandas as pd

            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(io.StringIO(content), sep=sep, nrows=200)
            st.dataframe(df)
        except Exception:
            st.code(content)
    else:
        lang = LANG_MAP.get(ext)
        st.code(content, language=lang)


def render_file_browser(root: Path, key_prefix: str = "fb"):
    """Render a file browser with preview for a workspace directory."""
    files = _collect_files(root)
    if not files:
        st.info("No files in workspace.")
        return

    truncated = len(files) > MAX_DISPLAY_FILES
    display_files = files[:MAX_DISPLAY_FILES]

    with st.expander(f"📁 Workspace Files ({len(files)})", expanded=False):
        for f in display_files:
            rel = str(f.relative_to(root))
            try:
                size = _format_size(f.stat().st_size)
            except OSError:
                size = "?"
            col_path, col_size, col_btn = st.columns([5, 1, 1])
            with col_path:
                st.text(f"📄 {rel}")
            with col_size:
                st.caption(size)
            with col_btn:
                if st.button("👁", key=f"{key_prefix}_{rel}", help="Preview"):
                    st.session_state[f"{key_prefix}_preview"] = str(f)
                    st.rerun()

        if truncated:
            st.caption(f"Showing first {MAX_DISPLAY_FILES} of {len(files)} files.")

    # Preview panel (outside expander so it stays visible)
    preview_path = st.session_state.get(f"{key_prefix}_preview")
    if preview_path:
        p = Path(preview_path)
        if p.exists():
            _render_preview(p, key_prefix)


def render_workspace_download(root: Path, key_prefix: str = "dl"):
    """Render a ZIP download button for workspace files."""
    files = _collect_files(root)
    if not files:
        return

    included = []
    excluded = []
    for f in files:
        rel = str(f.relative_to(root))
        try:
            size = f.stat().st_size
        except OSError:
            excluded.append((rel, "cannot read"))
            continue
        if size > MAX_ZIP_FILE_SIZE:
            excluded.append((rel, f"too large ({_format_size(size)})"))
        else:
            included.append(f)

    if not included:
        st.caption("No files small enough to package.")
        return

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in included:
            zf.write(f, f.relative_to(root))
    buf.seek(0)

    st.download_button(
        f"📦 Download Workspace ({len(included)} files)",
        data=buf.getvalue(),
        file_name="workspace.zip",
        mime="application/zip",
        key=f"{key_prefix}_zip",
    )

    if excluded:
        with st.expander(f"Excluded from download ({len(excluded)} files)", expanded=False):
            for path, reason in excluded:
                st.text(f"  {path} — {reason}")
