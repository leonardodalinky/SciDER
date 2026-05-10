"""Data analysis workflow form and runner."""

from pathlib import Path

import streamlit as st
from loguru import logger
from utils import cleanup_uploaded_data, save_and_extract_upload

from scider.core import constant
from scider.workflows.data_workflow import DataWorkflow
from scider.workflows.hypo_data_workflow import HypoDataWorkflow


def run_data(path, q, workspace_path):
    """Run data workflow. Called from background thread."""
    data_path = Path(path) if Path(path).exists() else Path(str(path))
    logger.info(f"Running data analysis on path: {data_path}")

    w = DataWorkflow(
        data_path=data_path,
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


def run_hypo_data(feature_desc, num_rows, query, workspace_path):
    """Run hypothetical data workflow. Called from background thread."""
    logger.info(f"Running hypothetical data generation: {feature_desc[:100]}...")

    w = HypoDataWorkflow(
        feature_desc=feature_desc,
        workspace_path=workspace_path,
        num_rows=num_rows,
        user_query=query,
        recursion_limit=100,
    )
    w.run()

    if w.final_status != "success":
        error_msg = w.error_message or "Hypothetical data workflow failed"
        return f"Workflow failed: {error_msg}", []
    out = ["## Hypothetical Data Analysis Complete"]
    if w.data_summary:
        out.append(w.data_summary)
    return "\n\n".join(out), []


def render_form():
    """Render the data analysis form. Returns workflow_config dict or None."""
    hf_enabled = constant.HF_DATASET_DOWNLOAD_ENABLED

    source_options = ["Upload local file"]
    if hf_enabled:
        source_options.append("HuggingFace dataset")
    source_options.append("Generate hypothetical data")

    data_source = st.radio(
        "Data Source",
        source_options,
        horizontal=True,
        key="data_source_radio",
    )

    with st.form("data_form", clear_on_submit=True):
        st.markdown("### Analyze Your Data")
        st.caption(
            "Upload a dataset, enter a HuggingFace repo name, or let SciDER generate synthetic data. "
            "The AI agent explores structure, runs statistical analysis, and searches for related metrics in the literature."
        )

        if data_source == "Generate hypothetical data":
            feature_desc = st.text_area(
                "Describe the data you want to generate",
                placeholder=(
                    "e.g. A dataset about house prices with features: "
                    "square footage (1000-5000 sq ft), number of bedrooms (1-6), "
                    "age of house (0-100 years), price ($100k-$1M)"
                ),
                height=150,
                help="Describe the features, their ranges, and the domain of the dataset.",
            )
            num_rows = st.number_input(
                "Number of rows",
                min_value=10,
                max_value=100000,
                value=1000,
                step=100,
            )
            query = st.text_input(
                "Analysis query",
                placeholder="What would you like to analyze about this data?",
            )
            submitted = st.form_submit_button("Generate & Analyze")
            if submitted:
                if not feature_desc or not feature_desc.strip():
                    st.error("Please describe the data you want to generate.")
                    return None
                return {
                    "type": "data_hypo",
                    "feature_desc": feature_desc.strip(),
                    "num_rows": num_rows,
                    "query": query
                    or f"Analyze this synthetic dataset: {feature_desc.strip()[:200]}",
                }

        elif data_source == "HuggingFace dataset":
            hf_repo = st.text_input(
                "HuggingFace Dataset Repo",
                placeholder="e.g. scikit-learn/iris",
                help="Enter a HuggingFace dataset repository name. It will be downloaded automatically.",
            )
            query = st.text_input(
                "Query",
                placeholder="e.g. What features most strongly predict the target variable?",
            )
            submitted = st.form_submit_button("Analyze Dataset")
            if submitted:
                if not hf_repo or not hf_repo.strip():
                    st.error("Please enter a HuggingFace dataset repository name.")
                    return None
                if not query or not query.strip():
                    query = "Analyze this dataset — explore its structure, key patterns, and notable findings."
                return {"type": "data", "path": hf_repo.strip(), "query": query}

        else:
            st.caption("Upload a zip file containing your dataset")
            uploaded_zip = st.file_uploader(
                "Upload ZIP dataset (optional)",
                type=["zip"],
                help="Upload a zip file containing your dataset. Extracted temporarily, deleted on reset.",
            )
            if st.session_state.get("uploaded_data_path"):
                st.info(f"Using uploaded data: `{st.session_state.uploaded_data_path}`")
            query = st.text_input(
                "Query",
                placeholder="e.g. What features most strongly predict the target variable?",
            )
            submitted = st.form_submit_button("Analyze Dataset")
            if submitted:
                if not query or not query.strip():
                    query = "Analyze this dataset — explore its structure, key patterns, and notable findings."
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
                        st.error(
                            f"Failed to process uploaded zip file. Extracted path: {extracted}"
                        )
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
