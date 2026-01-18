"""
SciEvo Streamlit Chat Interface - Enhanced Version

A ChatGPT-like interface with real-time progress tracking and intermediate outputs.
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Set environment variables before importing scievo to ensure Claude coding agent is used
os.environ["CODING_AGENT_VERSION"] = "v3"  # Use Claude coding agent (v3)
os.environ.setdefault("SCIEVO_ENABLE_OPENHANDS", "0")  # Disable OpenHands by default

# Add parent directory to path to import scievo
sys.path.insert(0, str(Path(__file__).parent.parent))

import display_components as dc
from workflow_monitor import get_monitor, reset_monitor

from scievo.core.llms import ModelRegistry
from scievo.workflows.full_workflow_with_ideation import FullWorkflowWithIdeation


# ==================== Model Registration ====================
def register_all_models():
    """Register all required models for SciEvo workflows."""
    # Get API keys from environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    # Default model (can be overridden via environment variable)
    default_model = os.getenv("SCIEVO_DEFAULT_MODEL", "gemini/gemini-2.5-flash-lite")
    default_api_key = gemini_api_key
    print("default_api_key: ", default_api_key)
    if not default_api_key:
        st.error(
            "❌ No API key found! Please set GEMINI_API_KEY or OPENAI_API_KEY in your environment."
        )
        st.stop()

    # List of all models that need to be registered
    models_to_register = [
        # Ideation workflow
        ("ideation", default_model, default_api_key),
        # Data workflow
        ("data", default_model, default_api_key),
        ("plan", default_model, default_api_key),
        ("history", default_model, default_api_key),
        # Experiment workflow
        ("experiment_agent", default_model, default_api_key),
        ("experiment_coding", default_model, default_api_key),
        ("experiment_execute", default_model, default_api_key),
        ("experiment_summary", default_model, default_api_key),
        # Critic agent (enabled by default)
        ("critic", default_model, default_api_key),
        # RBank (enabled by default)
        ("mem", default_model, default_api_key),
    ]

    # Register embedding model (can use different model/API key)
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-004")
    embed_api_key = os.getenv("EMBED_API_KEY", openai_api_key or default_api_key)
    models_to_register.append(("embed", embed_model, embed_api_key))

    # Register all models
    registered_models = []
    failed_models = []

    for model_name, model, api_key in models_to_register:
        try:
            ModelRegistry.register(
                name=model_name,
                model=model,
                api_key=api_key,
            )
            registered_models.append(model_name)
        except Exception as e:
            failed_models.append((model_name, str(e)))

    # Log registration status (only show errors in UI)
    if failed_models:
        st.warning(f"⚠️ Some models failed to register: {', '.join([m[0] for m in failed_models])}")
        for model_name, error in failed_models:
            st.error(f"Failed to register {model_name}: {error}")

    return len(registered_models), len(failed_models)


# ==================== Configuration ====================
st.set_page_config(
    page_title="SciEvo Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================== Session State Initialization ====================
def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "workflow_running" not in st.session_state:
        st.session_state.workflow_running = False
    if "current_workflow" not in st.session_state:
        st.session_state.current_workflow = None
    if "workflow_result" not in st.session_state:
        st.session_state.workflow_result = None
    if "workflow_data" not in st.session_state:
        st.session_state.workflow_data = {}
    if "show_advanced" not in st.session_state:
        st.session_state.show_advanced = False


# ==================== Sidebar Configuration ====================
def render_sidebar():
    """Render the sidebar with workflow configuration."""
    with st.sidebar:
        st.title("🔬 SciEvo Configuration")

        st.markdown("---")
        st.subheader("Research Query")
        user_query = st.text_area(
            "Research Topic/Query",
            placeholder="E.g., transformer models for time series forecasting",
            help="Your research topic or experimental objective",
            key="user_query",
            height=100,
        )

        research_domain = st.text_input(
            "Research Domain (optional)",
            placeholder="E.g., machine learning, chemistry",
            help="Specify the research domain for better context",
            key="research_domain",
        )

        st.markdown("---")
        st.subheader("Workspace Settings")

        workspace_path = st.text_input(
            "Workspace Path",
            value="./workspace",
            help="Directory where results will be saved",
            key="workspace_path",
        )

        session_name = st.text_input(
            "Session Name (optional)",
            placeholder="Auto-generated if empty",
            help="Custom name for this workflow session",
            key="session_name",
        )

        st.markdown("---")
        st.subheader("Workflow Stages")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**💡 Ideation**")
            st.caption("Always enabled")
        with col2:
            st.success("✓ Enabled")

        run_data_workflow = st.checkbox(
            "📊 Data Analysis",
            value=False,
            help="Analyze input data and search for papers/datasets",
            key="run_data_workflow",
        )

        data_path = None
        data_desc = None
        if run_data_workflow:
            with st.container():
                st.markdown("**Data Settings:**")
                data_path = st.text_input(
                    "Data File Path",
                    placeholder="./data/dataset.csv",
                    help="Path to your data file",
                    key="data_path",
                )
                data_desc = st.text_area(
                    "Data Description (optional)",
                    placeholder="Additional context about your data",
                    key="data_desc",
                    height=80,
                )

        run_experiment_workflow = st.checkbox(
            "🧪 Experiment Execution",
            value=False,
            help="Generate and execute experimental code",
            key="run_experiment_workflow",
        )

        repo_source = None
        max_revisions = 5
        if run_experiment_workflow:
            with st.container():
                st.markdown("**Experiment Settings:**")
                repo_source = st.text_input(
                    "Repository Source (optional)",
                    placeholder="./code or git URL",
                    help="Local path or git URL for code repository",
                    key="repo_source",
                )
                max_revisions = st.slider(
                    "Max Revisions",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Maximum revision loops",
                    key="max_revisions",
                )

        st.markdown("---")

        # Advanced settings in expander
        with st.expander("⚙️ Advanced Settings"):
            st.markdown("**Recursion Limits:**")
            ideation_limit = st.number_input(
                "Ideation Agent",
                min_value=10,
                max_value=200,
                value=50,
                key="ideation_limit",
            )
            data_limit = st.number_input(
                "Data Agent",
                min_value=10,
                max_value=200,
                value=100,
                key="data_limit",
            )
            experiment_limit = st.number_input(
                "Experiment Agent",
                min_value=10,
                max_value=200,
                value=100,
                key="experiment_limit",
            )

        return {
            "user_query": user_query,
            "research_domain": research_domain or None,
            "workspace_path": workspace_path,
            "session_name": session_name or None,
            "run_data_workflow": run_data_workflow,
            "data_path": data_path,
            "data_desc": data_desc,
            "run_experiment_workflow": run_experiment_workflow,
            "repo_source": repo_source,
            "max_revisions": max_revisions,
            "ideation_agent_recursion_limit": ideation_limit,
            "data_agent_recursion_limit": data_limit,
            "experiment_agent_recursion_limit": experiment_limit,
        }


# ==================== Workflow Runner ====================
class WorkflowRunner:
    """Manages workflow execution with progress tracking."""

    def __init__(self, config: dict):
        self.config = config
        self.workflow = None
        self.monitor = get_monitor()

    def run(self):
        """Run the workflow with progress tracking."""
        try:
            # Create workflow
            self.workflow = FullWorkflowWithIdeation(
                user_query=self.config["user_query"],
                workspace_path=Path(self.config["workspace_path"]),
                research_domain=self.config["research_domain"],
                data_path=Path(self.config["data_path"]) if self.config["data_path"] else None,
                run_data_workflow=self.config["run_data_workflow"],
                run_experiment_workflow=self.config["run_experiment_workflow"],
                repo_source=self.config["repo_source"],
                max_revisions=self.config["max_revisions"],
                ideation_agent_recursion_limit=self.config["ideation_agent_recursion_limit"],
                data_agent_recursion_limit=self.config["data_agent_recursion_limit"],
                experiment_agent_recursion_limit=self.config["experiment_agent_recursion_limit"],
                session_name=self.config["session_name"],
                data_desc=self.config["data_desc"],
            )

            # Run workflow
            result = self.workflow.run()
            return result

        except Exception as e:
            st.error(f"Error running workflow: {str(e)}")
            raise


# ==================== Main App ====================
def main():
    """Main application entry point."""
    init_session_state()

    # Register models on first run (only once per session)
    if "models_registered" not in st.session_state:
        try:
            registered_count, failed_count = register_all_models()
            st.session_state.models_registered = True
            st.session_state.models_registered_count = registered_count
            if failed_count == 0:
                # Only show success message once, don't clutter UI
                pass  # Success is silent, errors will be shown
        except Exception as e:
            st.error(f"❌ Failed to register models: {e}")
            st.exception(e)
            st.stop()

    # Render sidebar and get configuration
    config = render_sidebar()

    # Header
    st.title("🔬 SciEvo Research Assistant")
    st.markdown(
        """
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <p style="margin: 0; color: #31333F;">
            <strong>SciEvo</strong> is your AI-powered research assistant that automates the entire research workflow:
            💡 Research Ideation → 📊 Data Analysis → 🧪 Experiment Execution
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Main content area
    if not st.session_state.workflow_running and st.session_state.workflow_result is None:
        # Show start button
        st.markdown("### 🚀 Ready to Start")
        st.markdown("Configure your research settings in the sidebar and click below to begin.")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🚀 Start Research Workflow",
                type="primary",
                use_container_width=True,
                key="start_btn",
            ):
                # Validate inputs
                if not config["user_query"]:
                    st.error("❌ Please provide a research topic/query")
                elif config["run_data_workflow"] and not config["data_path"]:
                    st.error("❌ Data path is required when Data Workflow is enabled")
                else:
                    st.session_state.workflow_running = True
                    st.session_state.workflow_result = None
                    st.session_state.workflow_data = {
                        "ideation_papers": [],
                        "research_ideas": [],
                        "novelty_score": None,
                        "ideation_summary": None,
                        "ideation_status": "Starting...",
                        "papers": [],
                        "datasets": [],
                        "metrics": [],
                        "data_summary": None,
                        "data_status": "Pending",
                        "current_revision": 0,
                        "max_revisions": config["max_revisions"],
                        "current_phase": "init",
                        "execution_results": [],
                        "revision_summaries": [],
                    }
                    reset_monitor()
                    st.rerun()

    elif st.session_state.workflow_running:
        # Show workflow progress
        st.markdown("### 🔄 Workflow Running")

        # Progress containers
        progress_container = st.container()

        with progress_container:
            # Run workflow
            try:
                runner = WorkflowRunner(config)

                # Show initial status
                status_placeholder = st.empty()
                status_placeholder.info("🚀 Initializing workflow...")

                # Progress tracking
                progress_bar = st.progress(0)
                phase_text = st.empty()

                # Run workflow
                with st.spinner("Running workflow..."):
                    result = runner.run()

                # Update progress to complete
                progress_bar.progress(100)
                phase_text.success("✅ Workflow completed!")

                # Store result
                st.session_state.workflow_result = result
                st.session_state.workflow_running = False

                # Show intermediate results during execution
                if result.current_phase in ["ideation", "data_analysis", "experiment"]:
                    # Update workflow data
                    st.session_state.workflow_data.update(
                        {
                            "ideation_papers": result.ideation_papers,
                            "novelty_score": result.novelty_score,
                            "ideation_summary": result.ideation_summary,
                            "papers": result.papers,
                            "datasets": result.datasets,
                            "metrics": result.metrics,
                            "data_summary": result.data_summary,
                        }
                    )

                    # Display progress based on phase
                    if result.current_phase == "ideation":
                        dc.display_ideation_progress(st.session_state.workflow_data)
                    elif result.current_phase == "data_analysis":
                        dc.display_data_agent_progress(st.session_state.workflow_data)
                    elif result.current_phase == "experiment":
                        if hasattr(result, "_experiment_workflow") and result._experiment_workflow:
                            st.session_state.workflow_data.update(
                                {
                                    "current_revision": result._experiment_workflow.current_revision,
                                    "current_phase": result._experiment_workflow.current_phase,
                                    "execution_results": result._experiment_workflow.all_execution_results,
                                    "revision_summaries": result._experiment_workflow.revision_summaries,
                                }
                            )
                        dc.display_experiment_progress(st.session_state.workflow_data)

                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
                st.session_state.workflow_running = False

                if st.button("🔄 Reset and Try Again"):
                    st.session_state.workflow_result = None
                    st.rerun()

    else:
        # Show results
        result = st.session_state.workflow_result

        if result:
            st.markdown("### ✅ Workflow Complete")

            # Display final results
            dc.display_final_results(result)

            # Action buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("💾 Save Summary", use_container_width=True):
                    saved_path = result.save_summary()
                    st.success(f"Saved to: {saved_path}")

            with col2:
                if st.button("🔄 Start New Research", use_container_width=True):
                    st.session_state.workflow_result = None
                    st.session_state.workflow_running = False
                    st.session_state.workflow_data = {}
                    st.rerun()

            with col3:
                if st.button("📂 Open Workspace", use_container_width=True):
                    st.info(f"Workspace: {result.workspace_path}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p style="margin: 0;">SciEvo Research Assistant v1.0 | Powered by Multi-Agent AI</p>
            <p style="margin: 0; font-size: 0.8rem;">
                💡 Ideation → 📊 Data Analysis → 🧪 Experiment Execution
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
