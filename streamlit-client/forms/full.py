"""Full workflow form and runner."""

from pathlib import Path

import streamlit as st
from loguru import logger
from utils import _rm_upload_root, save_and_extract_upload

from scider.core import constant
from scider.workflows.full_workflow_with_ideation import FullWorkflowWithIdeation


def run_full(cfg, workspace_path):
    """Run full workflow. Called from background thread."""
    data_path = None
    if cfg.get("data_path"):
        dp = cfg["data_path"]
        data_path = Path(dp) if Path(dp).exists() else Path(str(dp))

    if data_path:
        logger.info(f"Running full workflow with data path: {data_path}")

    run_ideation = cfg.get("run_ideation", True)
    run_paper_writing = cfg.get("run_paper_writing", False)

    w = FullWorkflowWithIdeation(
        user_query=cfg["query"],
        workspace_path=workspace_path,
        data_path=data_path,
        run_data_workflow=cfg["run_data"],
        run_experiment_workflow=cfg["run_exp"],
        max_revisions=5,
        skip_ideation=not run_ideation,
        run_paper_writing=run_paper_writing,
    )
    w.run()
    return w.final_summary or "Workflow finished", []


def render_form():
    """Render the full workflow form. Returns workflow_config dict or None."""
    hf_enabled = constant.HF_DATASET_DOWNLOAD_ENABLED

    if hf_enabled:
        data_source = st.radio(
            "Data Source",
            ["Upload local file", "HuggingFace dataset"],
            horizontal=True,
            key="full_data_source_radio",
        )
    else:
        data_source = "Upload local file"

    with st.form("full_form", clear_on_submit=True):
        st.markdown("### Full Workflow")
        topic = st.text_input("Research Topic", placeholder="Enter your research topic...")

        if data_source == "HuggingFace dataset":
            hf_repo = st.text_input(
                "HuggingFace Dataset Repo",
                placeholder="e.g. scikit-learn/iris",
                help="Enter a HuggingFace dataset repository name.",
            )
        else:
            hf_repo = None
            st.caption("Data (for Data Analysis): upload zip or enter path")
            uploaded_full_zip = st.file_uploader(
                "Upload ZIP dataset (optional)",
                type=["zip"],
                key="full_upload",
                help="Zip dataset for Data Analysis. Extracted temporarily, deleted on reset.",
            )
            if st.session_state.get("uploaded_full_data_path"):
                st.info(f"Using: `{st.session_state.uploaded_full_data_path}`")

        run_ideation = st.checkbox("Run Ideation (literature search & idea generation)", value=True)
        run_data = st.checkbox("Run Data Analysis", value=True)
        run_exp = st.checkbox("Run Experiment", value=True)
        run_paper_writing = st.checkbox(
            "Run Paper Writing (LaTeX + PDF via PaperOrchestra)",
            value=False,
            help=(
                "After data + experiment finish, bootstrap a paper workspace under "
                "`<workspace>/paper/` and run the writing agent through the six "
                "PaperOrchestra steps to produce `final/paper.pdf`. Requires "
                "`latexmk` on the host."
            ),
        )
        submitted = st.form_submit_button(
            "Run Full Workflow",
        )

        if submitted and topic:
            data_path_to_use = None

            if run_data:
                # HuggingFace mode
                if data_source == "HuggingFace dataset":
                    if not hf_repo or not hf_repo.strip():
                        st.error("Please enter a HuggingFace dataset repository name.")
                        return None
                    data_path_to_use = hf_repo.strip()
                else:
                    # Local file mode
                    if uploaded_full_zip:
                        prev = st.session_state.get("uploaded_full_data_path")
                        if prev:
                            _rm_upload_root(Path(prev))
                            if "uploaded_full_data_path" in st.session_state:
                                del st.session_state.uploaded_full_data_path
                        extracted = save_and_extract_upload(uploaded_full_zip)
                        if extracted and extracted.exists():
                            extracted = extracted.resolve()
                            st.session_state.uploaded_full_data_path = str(extracted)
                            st.session_state.workspace_path = extracted.parent
                            data_path_to_use = str(extracted)
                            st.success(f"File uploaded and extracted to: {data_path_to_use}")
                        else:
                            st.error(
                                f"Failed to process uploaded zip file. Extracted path: {extracted}"
                            )
                            data_path_to_use = None
                    elif st.session_state.get("uploaded_full_data_path"):
                        p = Path(st.session_state.uploaded_full_data_path).resolve()
                        if p.exists():
                            data_path_to_use = str(p)
                            st.session_state.workspace_path = p.parent
                        else:
                            st.warning(f"Previously uploaded path no longer exists: {p}")
                            if "uploaded_full_data_path" in st.session_state:
                                del st.session_state.uploaded_full_data_path

                    if not data_path_to_use:
                        st.error(
                            "Run Data Analysis requires uploading a zip or entering a data path."
                        )
                        data_path_to_use = None

            if data_path_to_use is not None or not run_data:
                return {
                    "type": "full",
                    "query": topic,
                    "data_path": data_path_to_use,
                    "run_ideation": run_ideation,
                    "run_data": run_data,
                    "run_exp": run_exp,
                    "run_paper_writing": run_paper_writing,
                }
    return None
