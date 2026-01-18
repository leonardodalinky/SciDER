"""
SciEvo Streamlit Chat Interface

A ChatGPT-like interface for running the full SciEvo workflow with ideation agent.
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Try to load environment variables from env file (if exists)
try:
    from dotenv import load_dotenv

    # Load from parent directory's env file
    env_file = Path(__file__).parent.parent / "env"
    if env_file.exists():
        load_dotenv(env_file, override=False)  # Don't override existing env vars
except ImportError:
    # python-dotenv not installed, skip loading .env file
    pass

# Set environment variables before importing scievo to ensure Claude coding agent is used
os.environ["CODING_AGENT_VERSION"] = "v3"  # Use Claude coding agent (v3)
os.environ.setdefault("SCIEVO_ENABLE_OPENHANDS", "0")  # Disable OpenHands by default

# Add parent directory to path to import scievo
sys.path.insert(0, str(Path(__file__).parent.parent))

from scievo.core.llms import ModelRegistry
from scievo.workflows.full_workflow_with_ideation import FullWorkflowWithIdeation


# ==================== API Key Validation ====================
def test_api_key(model: str, api_key: str) -> tuple[bool, str]:
    """Test if an API key is valid by making a simple request."""
    try:
        from litellm import completion

        # For Gemini models, test with a simple completion
        if model.startswith("gemini/"):
            response = completion(
                model=model,
                messages=[{"role": "user", "content": "test"}],
                api_key=api_key,
                max_tokens=5,
            )
            return True, "API key is valid"
        else:
            # For other models, just check if we can initialize
            return True, "API key format looks valid"
    except Exception as e:
        error_msg = str(e)
        if (
            "API key" in error_msg.lower()
            or "authentication" in error_msg.lower()
            or "invalid" in error_msg.lower()
        ):
            return False, f"Invalid API key: {error_msg}"
        else:
            # Other errors might be network-related, but API key might be valid
            return True, f"API key format valid (test request failed: {error_msg})"


# ==================== Model Registration ====================
def register_all_models(user_api_key: str | None = None, user_model: str | None = None):
    """Register all required models for SciEvo workflows.

    Args:
        user_api_key: API key from user input (takes priority over environment variables)
        user_model: Model name from user input (takes priority over environment variables)
    """
    # Get API keys from environment variables (as fallback)
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    # Determine model to use (user input takes priority)
    default_model = user_model or os.getenv("SCIEVO_DEFAULT_MODEL", "gemini/gemini-2.5-flash-lite")

    # Use user-provided API key if available, otherwise use environment variables
    if user_api_key and user_api_key.strip():
        # User provided API key - determine which type based on model
        if default_model.startswith("gemini/"):
            default_api_key = user_api_key.strip()
            gemini_api_key = default_api_key  # Update for embedding model
        elif (
            default_model.startswith("gpt-")
            or default_model.startswith("o1-")
            or default_model.startswith("gpt-5")
        ):
            default_api_key = user_api_key.strip()
            openai_api_key = default_api_key  # Update for embedding model
        elif default_model.startswith("claude-"):
            default_api_key = user_api_key.strip()
            anthropic_api_key = default_api_key
        else:
            default_api_key = user_api_key.strip()
    else:
        # No user input, use environment variables

        # Determine which API key to use based on the model
        if default_model.startswith("gemini/"):
            default_api_key = gemini_api_key
            if not default_api_key:
                st.error(
                    "❌ GEMINI_API_KEY not found! Please enter your API key in the sidebar or set GEMINI_API_KEY in your environment."
                )
                st.stop()
        elif (
            default_model.startswith("gpt-")
            or default_model.startswith("o1-")
            or default_model.startswith("gpt-5")
        ):
            default_api_key = openai_api_key
            if not default_api_key:
                st.error(
                    "❌ OPENAI_API_KEY not found! Please enter your API key in the sidebar or set OPENAI_API_KEY in your environment."
                )
                st.stop()
        elif default_model.startswith("claude-"):
            default_api_key = anthropic_api_key
            if not default_api_key:
                st.error(
                    "❌ ANTHROPIC_API_KEY not found! Please enter your API key in the sidebar or set ANTHROPIC_API_KEY in your environment."
                )
                st.stop()
        else:
            # Fallback: try any available key
            default_api_key = gemini_api_key or openai_api_key or anthropic_api_key
            if not default_api_key:
                st.error(
                    "❌ No API key found! Please enter your API key in the sidebar or set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in your environment."
                )
                st.stop()

    # Validate API key format (basic check)
    if default_api_key:
        # Check if API key looks valid (not empty, has reasonable length)
        if len(default_api_key.strip()) < 10:
            st.error(f"❌ API key appears to be invalid (too short). Please check your API key.")
            st.stop()

        # For Gemini, check if it starts with "AIza" (typical Gemini API key prefix)
        if default_model.startswith("gemini/") and not default_api_key.startswith("AIza"):
            st.warning(
                f"⚠️ Warning: Gemini API key doesn't start with 'AIza'. Please verify your GEMINI_API_KEY is correct."
            )

        # Test API key validity (optional, can be disabled for faster startup)
        if os.getenv("SCIEVO_TEST_API_KEY", "false").lower() == "true":
            is_valid, msg = test_api_key(default_model, default_api_key)
            if not is_valid:
                st.error(f"❌ API key validation failed: {msg}")
                st.stop()
            else:
                st.success(f"✅ API key validated: {msg}")

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

    # Also set environment variable for litellm (some models may require it)
    if default_model.startswith("gemini/"):
        os.environ["GEMINI_API_KEY"] = default_api_key
    elif (
        default_model.startswith("gpt-")
        or default_model.startswith("o1-")
        or default_model.startswith("gpt-5")
    ):
        os.environ["OPENAI_API_KEY"] = default_api_key
    elif default_model.startswith("claude-"):
        os.environ["ANTHROPIC_API_KEY"] = default_api_key

    # Register all models
    registered_models = []
    failed_models = []

    for model_name, model, api_key in models_to_register:
        try:
            # Ensure API key is not None and not empty
            if not api_key or not api_key.strip():
                failed_models.append((model_name, "API key is empty"))
                continue

            # Clean API key
            clean_api_key = api_key.strip()

            # Set environment variable for this specific model type if needed
            if model.startswith("gemini/"):
                os.environ["GEMINI_API_KEY"] = clean_api_key
            elif model.startswith("gpt-") or model.startswith("o1-") or model.startswith("gpt-5"):
                os.environ["OPENAI_API_KEY"] = clean_api_key
            elif model.startswith("claude-"):
                os.environ["ANTHROPIC_API_KEY"] = clean_api_key

            ModelRegistry.register(
                name=model_name,
                model=model,
                api_key=clean_api_key,  # Pass API key explicitly
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


# Note: Model registration will happen in main() function to ensure it runs in streamlit context


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
    if "workflow_outputs" not in st.session_state:
        st.session_state.workflow_outputs = {}
    if "models_registered" not in st.session_state:
        st.session_state.models_registered = False


# ==================== Sidebar Configuration ====================
def render_sidebar():
    """Render the sidebar with workflow configuration."""
    with st.sidebar:
        st.title("🔬 SciEvo Configuration")

        st.markdown("---")
        st.subheader("🔑 API Configuration")

        # Model selection
        model_options = {
            "Gemini 2.5 Flash Lite": "gemini/gemini-2.5-flash-lite",
            "Gemini 2.5 Flash": "gemini/gemini-2.5-flash",
            "Gemini 2.5 Pro": "gemini/gemini-2.5-pro",
            "GPT-4o": "gpt-4o",
            "GPT-4o-mini": "gpt-4o-mini",
            "GPT-3.5 Turbo": "gpt-3.5-turbo",
            "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
            "Claude 3 Opus": "claude-3-opus-20240229",
        }

        selected_model_display = st.selectbox(
            "Model",
            options=list(model_options.keys()),
            index=0,  # Default to Gemini 2.5 Flash Lite
            help="Select the LLM model to use",
            key="selected_model",
        )
        selected_model = model_options[selected_model_display]

        # API Key input
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="Enter your API key here",
            help="Enter your API key. For Gemini models, use GEMINI_API_KEY. For OpenAI models, use OPENAI_API_KEY. For Claude models, use ANTHROPIC_API_KEY.",
            key="api_key_input",
        )

        # Show API key status
        if api_key:
            # Basic validation
            if selected_model.startswith("gemini/"):
                if api_key.startswith("AIza"):
                    st.success("✅ Gemini API key format looks valid")
                else:
                    st.warning("⚠️ Gemini API keys usually start with 'AIza'")
            elif (
                selected_model.startswith("gpt-")
                or selected_model.startswith("o1-")
                or selected_model.startswith("gpt-5")
            ):
                if api_key.startswith("sk-"):
                    st.success("✅ OpenAI API key format looks valid")
                else:
                    st.warning("⚠️ OpenAI API keys usually start with 'sk-'")
            elif selected_model.startswith("claude-"):
                if api_key.startswith("sk-ant-"):
                    st.success("✅ Anthropic API key format looks valid")
                else:
                    st.warning("⚠️ Anthropic API keys usually start with 'sk-ant-'")
        else:
            # Try to load from environment as fallback
            env_key = None
            if selected_model.startswith("gemini/"):
                env_key = os.getenv("GEMINI_API_KEY")
            elif (
                selected_model.startswith("gpt-")
                or selected_model.startswith("o1-")
                or selected_model.startswith("gpt-5")
            ):
                env_key = os.getenv("OPENAI_API_KEY")
            elif selected_model.startswith("claude-"):
                env_key = os.getenv("ANTHROPIC_API_KEY")

            if env_key:
                st.info("ℹ️ Using API key from environment variables")
            else:
                st.warning("⚠️ Please enter your API key above")

        st.markdown("---")
        st.subheader("Research Query")
        user_query = st.text_area(
            "Research Topic/Query",
            placeholder="E.g., transformer models for time series forecasting",
            help="Your research topic or experimental objective",
            key="user_query",
        )

        research_domain = st.text_input(
            "Research Domain (optional)",
            placeholder="E.g., machine learning, chemistry, biology",
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
            placeholder="Leave empty for auto-generated timestamp",
            help="Custom name for this workflow session",
            key="session_name",
        )

        st.markdown("---")
        st.subheader("Data Workflow (Optional)")

        run_data_workflow = st.checkbox(
            "Enable Data Analysis",
            value=False,
            help="Run DataAgent to analyze input data",
            key="run_data_workflow",
        )

        data_path = None
        data_desc = None
        if run_data_workflow:
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
            )

        st.markdown("---")
        st.subheader("Experiment Workflow (Optional)")

        run_experiment_workflow = st.checkbox(
            "Enable Experiment Execution",
            value=False,
            help="Run ExperimentAgent to generate and execute code",
            key="run_experiment_workflow",
        )

        repo_source = None
        max_revisions = 5
        if run_experiment_workflow:
            repo_source = st.text_input(
                "Repository Source (optional)",
                placeholder="./code or https://github.com/user/repo",
                help="Local path or git URL for code repository",
                key="repo_source",
            )
            max_revisions = st.number_input(
                "Max Revisions",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum revision loops for experiment refinement",
                key="max_revisions",
            )

        st.markdown("---")
        st.subheader("Advanced Settings")

        with st.expander("Recursion Limits"):
            ideation_limit = st.number_input(
                "Ideation Agent Limit",
                min_value=10,
                max_value=200,
                value=50,
                key="ideation_limit",
            )
            data_limit = st.number_input(
                "Data Agent Limit",
                min_value=10,
                max_value=200,
                value=100,
                key="data_limit",
            )
            experiment_limit = st.number_input(
                "Experiment Agent Limit",
                min_value=10,
                max_value=200,
                value=100,
                key="experiment_limit",
            )

        return {
            "api_key": api_key,
            "model": selected_model,
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


# ==================== Display Functions ====================
def display_message(role: str, content: str, avatar: str = None):
    """Display a message in the chat interface."""
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)


def display_phase_output(phase_name: str, data: dict):
    """Display output from a specific workflow phase."""
    with st.expander(f"📊 {phase_name}", expanded=True):
        for key, value in data.items():
            if value:
                st.markdown(f"**{key}:**")
                if isinstance(value, str):
                    st.markdown(value)
                elif isinstance(value, list):
                    if value:
                        st.json(value)
                elif isinstance(value, dict):
                    st.json(value)
                else:
                    st.write(value)


def display_workflow_progress(workflow: FullWorkflowWithIdeation):
    """Display current workflow progress."""
    phase_mapping = {
        "init": "🔄 Initializing",
        "ideation": "💡 Research Ideation",
        "data_analysis": "📊 Data Analysis",
        "experiment": "🧪 Running Experiments",
        "complete": "✅ Complete",
        "failed": "❌ Failed",
    }

    current_phase_display = phase_mapping.get(workflow.current_phase, workflow.current_phase)

    progress_container = st.container()
    with progress_container:
        st.markdown(f"### Current Phase: {current_phase_display}")

        # Show phase-specific progress
        if workflow.current_phase == "ideation":
            cols = st.columns(5)
            with cols[0]:
                st.metric("Papers Found", len(workflow.ideation_papers))
            with cols[1]:
                st.metric("Analyzed Papers", len(getattr(workflow, "analyzed_papers", [])))
            with cols[2]:
                st.metric("Ideas Generated", len(getattr(workflow, "research_ideas", [])))
            with cols[3]:
                if workflow.novelty_score:
                    st.metric("Novelty Score", f"{workflow.novelty_score:.1f}/10")
            with cols[4]:
                st.info("🔍 Generating Research Ideas...")

        elif workflow.current_phase == "data_analysis":
            st.info("📊 Analyzing data and searching for relevant papers/datasets...")

        elif workflow.current_phase == "experiment":
            if hasattr(workflow, "_experiment_workflow") and workflow._experiment_workflow:
                exp = workflow._experiment_workflow
                st.info(
                    f"🧪 Revision {exp.current_revision + 1}/{exp.max_revisions} - Phase: {exp.current_phase}"
                )


# ==================== Workflow Runner ====================
def run_workflow_with_updates(config: dict, output_container):
    """Run the workflow and update the UI with progress."""
    try:
        # Validate required inputs
        if not config["user_query"]:
            st.error("❌ Please provide a research topic/query")
            st.session_state.workflow_running = False
            return

        if config["run_data_workflow"] and not config["data_path"]:
            st.error("❌ Data path is required when Data Workflow is enabled")
            st.session_state.workflow_running = False
            return

        # Create workflow
        with output_container:
            st.info("🚀 Starting SciEvo workflow...")

        workflow = FullWorkflowWithIdeation(
            user_query=config["user_query"],
            workspace_path=Path(config["workspace_path"]),
            research_domain=config["research_domain"],
            data_path=Path(config["data_path"]) if config["data_path"] else None,
            run_data_workflow=config["run_data_workflow"],
            run_experiment_workflow=config["run_experiment_workflow"],
            repo_source=config["repo_source"],
            max_revisions=config["max_revisions"],
            ideation_agent_recursion_limit=config["ideation_agent_recursion_limit"],
            data_agent_recursion_limit=config["data_agent_recursion_limit"],
            experiment_agent_recursion_limit=config["experiment_agent_recursion_limit"],
            session_name=config["session_name"],
            data_desc=config["data_desc"],
        )

        st.session_state.current_workflow = workflow

        # Run the workflow
        with output_container:
            with st.spinner("Running workflow..."):
                result = workflow.run()

        st.session_state.current_workflow = result

        # Display results
        with output_container:
            st.success(f"✅ Workflow completed: {result.final_status}")

            # Display ideation results
            if result.ideation_summary:
                display_phase_output(
                    "Research Ideation Results",
                    {
                        "Summary": result.ideation_summary,
                        "Novelty Score": (
                            f"{result.novelty_score:.2f}/10" if result.novelty_score else "N/A"
                        ),
                        "Novelty Feedback": result.novelty_feedback,
                        "Papers Reviewed": (
                            result.ideation_papers[:5] if result.ideation_papers else []
                        ),
                    },
                )

            # Display data analysis results
            if result.run_data_workflow and result.data_summary:
                display_phase_output(
                    "Data Analysis Results",
                    {
                        "Data Summary": result.data_summary,
                        "Papers Found": result.papers[:3] if result.papers else [],
                        "Datasets Found": result.datasets[:3] if result.datasets else [],
                        "Metrics Found": result.metrics[:3] if result.metrics else [],
                        "Paper Search Summary": result.paper_search_summary,
                    },
                )

            # Display experiment results
            if result.run_experiment_workflow and result.execution_results:
                display_phase_output(
                    "Experiment Results",
                    {
                        "Execution Results": result.execution_results,
                        "Number of Revisions": len(result.execution_results),
                    },
                )

            # Display final summary
            st.markdown("### 📋 Final Summary")
            st.markdown(result.final_summary)

            # Save summary button
            if st.button("💾 Save Summary to File"):
                saved_path = result.save_summary()
                st.success(f"Summary saved to: {saved_path}")

    except Exception as e:
        with output_container:
            st.error(f"❌ Error running workflow: {str(e)}")
            st.exception(e)

    finally:
        st.session_state.workflow_running = False


# ==================== Main App ====================
def main():
    """Main application entry point."""
    init_session_state()

    # Render sidebar and get configuration first (to get API key)
    config = render_sidebar()

    # Check if we need to register/re-register models
    current_api_key = config.get("api_key", "")
    current_model = config.get("model", "")
    last_api_key = st.session_state.get("last_registered_api_key", "")
    last_model = st.session_state.get("last_registered_model", "")

    # Register models if:
    # 1. Not registered yet, OR
    # 2. API key changed, OR
    # 3. Model changed
    needs_registration = (
        not st.session_state.get("models_registered", False)
        or current_api_key != last_api_key
        or current_model != last_model
    )

    if needs_registration:
        # Validate API key before registering
        if not current_api_key or not current_api_key.strip():
            # Try to use environment variable as fallback
            if current_model.startswith("gemini/"):
                env_key = os.getenv("GEMINI_API_KEY")
            elif (
                current_model.startswith("gpt-")
                or current_model.startswith("o1-")
                or current_model.startswith("gpt-5")
            ):
                env_key = os.getenv("OPENAI_API_KEY")
            elif current_model.startswith("claude-"):
                env_key = os.getenv("ANTHROPIC_API_KEY")
            else:
                env_key = (
                    os.getenv("GEMINI_API_KEY")
                    or os.getenv("OPENAI_API_KEY")
                    or os.getenv("ANTHROPIC_API_KEY")
                )

            if not env_key:
                st.warning("⚠️ Please enter your API key in the sidebar to continue.")
                st.stop()
            else:
                current_api_key = env_key

        try:
            registered_count, failed_count = register_all_models(
                user_api_key=current_api_key if current_api_key else None,
                user_model=current_model if current_model else None,
            )
            st.session_state.models_registered = True
            st.session_state.models_registered_count = registered_count
            st.session_state.last_registered_api_key = current_api_key
            st.session_state.last_registered_model = current_model

            if failed_count == 0:
                st.success(f"✅ Successfully registered {registered_count} models")
            else:
                st.warning(f"⚠️ Registered {registered_count} models, {failed_count} failed")
        except Exception as e:
            st.error(f"❌ Failed to register models: {e}")
            st.exception(e)
            st.stop()

    # Main content area
    st.title("🔬 SciEvo Research Assistant")
    st.markdown(
        """
    Welcome to SciEvo! This interface allows you to run the complete research workflow:
    - 💡 **Research Ideation**: Generate novel research ideas through literature review
    - 📊 **Data Analysis**: Analyze your data and find relevant papers/datasets
    - 🧪 **Experiment Execution**: Automatically generate and execute experimental code
    """
    )

    st.markdown("---")

    # Display chat history
    for message in st.session_state.messages:
        display_message(message["role"], message["content"])

    # Start workflow button
    if not st.session_state.workflow_running:
        # Validate API key before allowing workflow start
        api_key_valid = (
            config.get("api_key")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
        )

        if st.button("🚀 Start Research Workflow", type="primary", use_container_width=True):
            if not api_key_valid:
                st.error(
                    "❌ Please enter your API key in the sidebar before starting the workflow."
                )
            elif not config.get("user_query"):
                st.error("❌ Please provide a research topic/query.")
            else:
                st.session_state.workflow_running = True
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": f"**Research Query:** {config['user_query']}\n\n"
                        f"**Research Domain:** {config['research_domain'] or 'Not specified'}\n\n"
                        f"**Model:** {config.get('model', 'N/A')}\n\n"
                        f"**Data Workflow:** {'Enabled' if config['run_data_workflow'] else 'Disabled'}\n\n"
                        f"**Experiment Workflow:** {'Enabled' if config['run_experiment_workflow'] else 'Disabled'}",
                    }
                )
                st.rerun()
    else:
        # Show running status
        with st.spinner("Workflow is running..."):
            output_container = st.container()

            # Display current workflow progress if available
            if st.session_state.current_workflow:
                display_workflow_progress(st.session_state.current_workflow)

            # Run workflow in background
            run_workflow_with_updates(config, output_container)

            # Add result to messages
            if st.session_state.current_workflow:
                result = st.session_state.current_workflow
                st.session_state.messages.append(
                    {"role": "assistant", "content": result.final_summary}
                )

        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666;">
        <p>SciEvo Research Assistant | Powered by Multi-Agent AI</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
