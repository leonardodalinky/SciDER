"""Experiment workflow form and runner."""

from pathlib import Path

import streamlit as st
from loguru import logger
from utils import _rm_upload_root, find_data_analysis_file, save_and_extract_upload

from scider.workflows.experiment_workflow import ExperimentWorkflow


def run_experiment(q, path, workspace_path):
    """Run experiment workflow. Called from background thread."""
    if path:
        analysis_path = Path(path).resolve()
        if not analysis_path.exists():
            return f"Error: Analysis path does not exist: {analysis_path}", []
        logger.info(f"Running experiment with analysis file: {analysis_path}")
        w = ExperimentWorkflow.from_data_analysis_file(
            workspace_path=workspace_path,
            user_query=q,
            data_analysis_path=str(analysis_path),
            max_revisions=5,
            recursion_limit=100,
        )
    else:
        return "No data analysis file", []
    w.run()
    return w.final_summary or "Experiment finished", w.experiment_agent_intermediate_state


def render_form():
    """Render the experiment form. Returns workflow_config dict or None."""
    with st.form("experiment_form", clear_on_submit=True):
        st.markdown("### Experiment Workflow")
        st.caption("Upload a zip containing data_analysis.md or enter path manually")
        uploaded_exp_zip = st.file_uploader(
            "Upload ZIP with data analysis (optional)",
            type=["zip"],
            key="exp_upload",
            help="Zip containing data_analysis.md. Extracted temporarily, deleted on reset.",
        )
        if st.session_state.get("uploaded_experiment_path"):
            st.info(f"Using: `{st.session_state.uploaded_experiment_path}`")
        query = st.text_input("Experiment Query", placeholder="Describe your experiment...")
        submitted = st.form_submit_button(
            "Run Experiment",
        )
        if submitted:
            if not query or not query.strip():
                query = "Run an experiment based on the data analysis — build a model, evaluate it, and report results."
            path_to_use = None
            if uploaded_exp_zip:
                prev = st.session_state.get("uploaded_experiment_path")
                if prev:
                    _rm_upload_root(Path(prev))
                    if "uploaded_experiment_path" in st.session_state:
                        del st.session_state.uploaded_experiment_path
                extracted = save_and_extract_upload(uploaded_exp_zip)
                if extracted and extracted.exists():
                    extracted = extracted.resolve()
                    analysis_file = find_data_analysis_file(extracted)
                    if analysis_file and analysis_file.exists():
                        analysis_file = analysis_file.resolve()
                        st.session_state.uploaded_experiment_path = str(analysis_file)
                        st.session_state.workspace_path = analysis_file.parent
                        path_to_use = str(analysis_file)
                        st.success(f"Found analysis file: {path_to_use}")
                    else:
                        st.error(
                            f"Zip must contain data_analysis.md or analysis.md. Searched in: {extracted}"
                        )
                else:
                    st.error(f"Failed to process uploaded zip file. Extracted path: {extracted}")
            elif st.session_state.get("uploaded_experiment_path"):
                p = Path(st.session_state.uploaded_experiment_path).resolve()
                if p.exists():
                    path_to_use = str(p)
                    st.session_state.workspace_path = p.parent
                else:
                    st.warning(f"Previously uploaded path no longer exists: {p}")
                    if "uploaded_experiment_path" in st.session_state:
                        del st.session_state.uploaded_experiment_path
            if path_to_use:
                return {"type": "experiment", "query": query, "path": path_to_use}
            else:
                st.error("Please upload a zip with data_analysis.md or enter a data analysis path.")
    return None
