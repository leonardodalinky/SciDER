"""Data analysis workflow form and runner."""

from pathlib import Path

import streamlit as st
from loguru import logger
from utils import cleanup_uploaded_data, save_and_extract_upload

from scider.core import constant
from scider.workflows.data_workflow import DataWorkflow


def run_data(path, q, workspace_path):
    """Run data workflow. Called from background thread."""
    data_path = Path(path) if Path(path).exists() else path  # may be HF repo name
    logger.info(f"Running data analysis on path: {data_path}")

    w = DataWorkflow(
        data_path=Path(path) if Path(path).exists() else Path(str(path)),
        workspace_path=workspace_path,
        recursion_limit=100,
    )
    w.run()
    intermediate_state = getattr(w, "data_agent_intermediate_state", [])
    if w.final_status != "success":
        error_msg = w.error_message or "Data workflow failed"
        return f"Data workflow failed: {error_msg}", intermediate_state
    out = ["## Data Analysis Complete"]
    if w.data_summary:
        out.append(w.data_summary)
    return "\n\n".join(out), intermediate_state


def render_form():
    """Render the data analysis form. Returns workflow_config dict or None."""
    hf_enabled = constant.HF_DATASET_DOWNLOAD_ENABLED

    if hf_enabled:
        data_source = st.radio(
            "Data Source",
            ["Upload local file", "HuggingFace dataset"],
            horizontal=True,
            key="data_source_radio",
        )
    else:
        data_source = "Upload local file"

    with st.form("data_form", clear_on_submit=True):
        st.markdown("### Data Analysis Workflow")

        if data_source == "HuggingFace dataset":
            hf_repo = st.text_input(
                "HuggingFace Dataset Repo",
                placeholder="e.g. scikit-learn/iris",
                help="Enter a HuggingFace dataset repository name. It will be downloaded automatically.",
            )
        else:
            hf_repo = None
            st.caption("Upload a zip dataset or enter a path to existing data")
            uploaded_zip = st.file_uploader(
                "Upload ZIP dataset (optional)",
                type=["zip"],
                help="Upload a zip file containing your dataset. Extracted temporarily, deleted on reset.",
            )
            if st.session_state.get("uploaded_data_path"):
                st.info(f"Using uploaded data: `{st.session_state.uploaded_data_path}`")

        query = st.text_input("Query", placeholder="What would you like to analyze?")
        submitted = st.form_submit_button("Run Data Analysis", type="primary")

        if submitted and query:
            # HuggingFace mode
            if data_source == "HuggingFace dataset":
                if not hf_repo or not hf_repo.strip():
                    st.error("Please enter a HuggingFace dataset repository name.")
                    return None
                return {"type": "data", "path": hf_repo.strip(), "query": query}

            # Local file mode
            path_to_use = None
            if uploaded_zip:
                cleanup_uploaded_data()
                extracted = save_and_extract_upload(uploaded_zip)
                if extracted and extracted.exists():
                    extracted = extracted.resolve()
                    st.session_state.uploaded_data_path = str(extracted)
                    st.session_state.workspace_path = extracted.parent
                    path_to_use = str(extracted)
                    st.success(f"File uploaded and extracted to: {path_to_use}")
                else:
                    st.error(f"Failed to process uploaded zip file. Extracted path: {extracted}")
            elif st.session_state.get("uploaded_data_path"):
                path = Path(st.session_state.uploaded_data_path).resolve()
                if path.exists():
                    path_to_use = str(path)
                    st.session_state.workspace_path = path.parent
                else:
                    st.warning(f"Previously uploaded path no longer exists: {path}")
                    cleanup_uploaded_data()
            if path_to_use:
                verify_path = Path(path_to_use).resolve()
                if not verify_path.exists():
                    st.error(f"Path does not exist: {path_to_use}")
                else:
                    return {"type": "data", "path": str(verify_path), "query": query}
            else:
                st.error("Please upload a zip file or enter a data path.")
    return None
