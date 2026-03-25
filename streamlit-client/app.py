"""SciDER Research Assistant — Streamlit App (main entry point)."""

import os
import sys
import time
from pathlib import Path

import streamlit as st

os.environ["CODING_AGENT_VERSION"] = "v3"
os.environ.setdefault("SCIDER_ENABLE_OPENHANDS", "0")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from components.display import render_approval_ui
from forms.data import render_form as data_form
from forms.data import run_data
from forms.experiment import render_form as experiment_form
from forms.experiment import run_experiment
from forms.full import render_form as full_form
from forms.full import run_full
from forms.ideation import render_form as ideation_form
from forms.ideation import run_ideation
from forms.settings import render_settings_form
from settings import has_settings, load_settings, save_settings
from utils import cleanup_uploaded_data, save_chat_history
from workflow.approval import StreamlitApprovalHandler
from workflow.runner import WorkflowRunner

from scider.agents import ideation_agent
from scider.core import approval as approval_module
from scider.core.brain import Brain
from scider.core.llms import ModelRegistry
from scider.core.types import set_on_message_callback

# ==================== Page config ====================

st.set_page_config(page_title="SciDER Chat", layout="centered")

st.markdown(
    """
    <style>
    h1, h2, h3, h4, h5, h6 { color: #384166 !important; }
    [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] h4,
    [data-testid="stChatMessage"] h5, [data-testid="stChatMessage"] h6 {
        color: inherit !important;
    }
    .chat-bubble {
        padding: 10px 16px;
        border-radius: 12px;
        margin: 6px 0;
        max-width: 85%;
        word-wrap: break-word;
        line-height: 1.5;
        font-size: 14px;
    }
    .chat-bubble-user {
        background-color: #d4edda;
        color: #1a1a1a;
        margin-left: auto;
        text-align: right;
    }
    .chat-bubble-assistant {
        background-color: #f8f9fa;
        color: #1a1a1a;
    }
    .chat-bubble-tool {
        background-color: #e8d5f5;
        color: #1a1a1a;
        font-family: monospace;
        font-size: 13px;
    }
    .chat-row-right {
        display: flex;
        justify-content: flex-end;
    }
    .chat-row-left {
        display: flex;
        justify-content: flex-start;
    }
    .chat-bubble details summary {
        cursor: pointer;
        color: #666;
        font-size: 12px;
        margin-top: 4px;
    }
    .chat-bubble details summary:hover {
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_TRUNCATE_LEN = 300


def render_chat_message(role: str, content: str):
    """Render a chat message with colored bubble. Long messages are truncated."""
    import html as _html

    if role == "user":
        css_class = "chat-bubble chat-bubble-user"
        row_class = "chat-row-right"
    elif role == "tool":
        css_class = "chat-bubble chat-bubble-tool"
        row_class = "chat-row-left"
    else:
        css_class = "chat-bubble chat-bubble-assistant"
        row_class = "chat-row-left"

    escaped = _html.escape(content)

    if len(content) > _TRUNCATE_LEN:
        preview = _html.escape(content[:_TRUNCATE_LEN]).replace("\n", "<br>")
        full = escaped.replace("\n", "<br>")
        body = (
            f"{preview}..." f"<details><summary>Show more</summary>" f"<div>{full}</div></details>"
        )
    else:
        body = escaped.replace("\n", "<br>")

    st.markdown(
        f'<div class="{row_class}"><div class="{css_class}">{body}</div></div>',
        unsafe_allow_html=True,
    )


# ==================== Model registration ====================


def register_all_models(settings: dict):
    """Register models from saved settings. Returns True on success."""
    api_key = settings.get("api_key")
    if not api_key:
        return False

    role_models = settings.get("model_roles", {})

    # Register each role with its assigned model
    all_roles = [
        "ideation",
        "data",
        "plan",
        "history",
        "experiment_agent",
        "experiment_coding",
        "experiment_execute",
        "experiment_summary",
        "experiment_monitor",
        "paper_search",
        "metric_search",
        "critic",
        "mem",
    ]
    for role in all_roles:
        model = role_models.get(role)
        if not model:
            # Fallback: use first available model from provider
            provider = settings.get("model_provider", "Gemini")
            model = "gemini/gemini-2.5-flash-lite" if provider == "Gemini" else "gpt-5-nano"
        ModelRegistry.register(name=role, model=model, api_key=api_key)

    # Embedding: requires OpenAI key
    openai_key = settings.get("openai_api_key", "")
    if openai_key:
        ModelRegistry.register(name="embed", model="text-embedding-3-small", api_key=openai_key)

    return True


# ==================== Settings gate ====================

# Load .env for non-secret vars (BRAIN_DIR, logging, etc.) — but NOT for API keys
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except Exception:
    pass

# --- First visit: show settings page ---
if not has_settings():
    st.title("SciDER Research Assistant")
    st.info("Welcome! Please configure your API keys to get started.")
    new_settings = render_settings_form()
    if new_settings:
        save_settings(new_settings)
        st.rerun()
    st.stop()

# --- Load saved settings ---
_settings = load_settings()

# --- Settings page (when user clicks Settings button) ---
if st.session_state.get("show_settings"):
    st.title("SciDER Research Assistant — Settings")
    new_settings = render_settings_form(current_settings=_settings)
    if new_settings:
        save_settings(new_settings)
        st.session_state.show_settings = False
        # Force re-initialization with new settings
        if "initialized" in st.session_state:
            del st.session_state.initialized
        st.rerun()
    if st.button("Cancel"):
        st.session_state.show_settings = False
        st.rerun()
    st.stop()


# ==================== Apply settings ====================

# Set env vars from saved settings
if _settings.get("anthropic_api_key"):
    os.environ["ANTHROPIC_API_KEY"] = _settings["anthropic_api_key"]

# --- Title bar ---
col_title, col_settings, col_reset = st.columns([5, 1, 1])
with col_title:
    st.title("SciDER Research Assistant")
with col_settings:
    if st.button("Settings", key="btn_settings"):
        st.session_state.show_settings = True
        st.rerun()
with col_reset:
    if st.button("Reset", help="Clear chat history", key="btn_reset"):
        cleanup_uploaded_data()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello. I can run ideation, data analysis, experiments, or a full workflow.\n\nPlease select a workflow type below to get started.",
            }
        ]
        if "selected_workflow" in st.session_state:
            st.session_state.selected_workflow = None
        st.rerun()

# --- One-time initialization ---
if "initialized" not in st.session_state:
    if not os.getenv("BRAIN_DIR"):
        os.environ["BRAIN_DIR"] = str(Path.cwd() / "tmp_brain")

    # Memory: only enable if user toggled on AND has OpenAI key
    if _settings.get("memory_enabled") and _settings.get("openai_api_key"):
        os.environ["REASONING_BANK_ENABLED"] = "true"
    else:
        os.environ["REASONING_BANK_ENABLED"] = "false"

    Brain()
    if register_all_models(_settings):
        st.session_state.ideation_graph = ideation_agent.build().compile()
        st.session_state.initialized = True
    else:
        st.error("Failed to register models. Please check your API key in Settings.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello. I can run ideation, data analysis, experiments, or a full workflow.\n\nPlease select a workflow type below to get started.",
        }
    ]

if "workspace_path" not in st.session_state:
    st.session_state.workspace_path = Path.cwd() / "workspace"
if "default_workspace_path" not in st.session_state:
    st.session_state.default_workspace_path = Path.cwd() / "workspace"
_ws = st.session_state.workspace_path
if isinstance(_ws, (str, Path)) and "scider_uploads" in str(_ws) and not Path(_ws).exists():
    cleanup_uploaded_data()

if "selected_workflow" not in st.session_state:
    st.session_state.selected_workflow = None


# ==================== Workflow selection ====================

st.subheader("Select Workflow Type")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Ideation", use_container_width=True, key="btn_ideation"):
        st.session_state.selected_workflow = "ideation"
        st.rerun()
with col2:
    if st.button("Data Analysis", use_container_width=True, key="btn_data"):
        st.session_state.selected_workflow = "data"
        st.rerun()
with col3:
    if st.button("Experiment", use_container_width=True, key="btn_experiment"):
        st.session_state.selected_workflow = "experiment"
        st.rerun()
with col4:
    if st.button("Full Workflow", use_container_width=True, key="btn_full"):
        st.session_state.selected_workflow = "full"
        st.rerun()

st.divider()

# --- Chat history (skip if workflow is running — polling loop handles rendering) ---
if "workflow_runner" not in st.session_state:
    for m in st.session_state.messages:
        render_chat_message(m["role"], m["content"])


# ==================== Workflow forms ====================

workflow_config = None

if st.session_state.selected_workflow == "ideation":
    workflow_config = ideation_form()
elif st.session_state.selected_workflow == "data":
    workflow_config = data_form()
elif st.session_state.selected_workflow == "experiment":
    workflow_config = experiment_form()
elif st.session_state.selected_workflow == "full":
    workflow_config = full_form()

if workflow_config:
    st.session_state.selected_workflow = None


# ==================== Background workflow execution ====================


def _build_user_msg(wc):
    if wc["type"] == "ideation":
        return f"Ideation: {wc['query']}"
    elif wc["type"] == "data":
        return f"Data Analysis: {wc['path']} - {wc['query']}"
    elif wc["type"] == "experiment":
        msg = f"Experiment: {wc['query']}"
        if wc.get("path"):
            msg += f" (Data: {wc['path']})"
        return msg
    else:
        msg = f"Full Workflow: {wc['query']}"
        if wc.get("data_path"):
            msg += f" (Data: {wc['data_path']})"
        if wc.get("run_data"):
            msg += " [Data Analysis]"
        if wc.get("run_exp"):
            msg += " [Experiment]"
        return msg


def _run_workflow_func(wc, ideation_graph, workspace_path):
    """Execute the appropriate workflow function (called from background thread)."""
    wtype = wc["type"]
    if wtype == "ideation":
        return run_ideation(wc.get("query"), ideation_graph)
    elif wtype == "data":
        return run_data(wc["path"], wc["query"], workspace_path)
    elif wtype == "experiment":
        return run_experiment(wc["query"], wc.get("path"), workspace_path)
    elif wtype == "full":
        return run_full(wc, workspace_path)
    return "Unknown workflow type", []


# --- Launch background workflow ---
if workflow_config and "workflow_runner" not in st.session_state:
    st.session_state.messages.append({"role": "user", "content": _build_user_msg(workflow_config)})

    _graph = st.session_state.ideation_graph
    _wspath = st.session_state.workspace_path

    handler = StreamlitApprovalHandler()
    st.session_state.approval_handler = handler
    approval_module.set_handler(handler)

    # Hook every add_message() call to push to UI
    def _on_msg(msg):
        if msg.content:
            handler.push_message(msg.role or "assistant", msg.content)

    set_on_message_callback(_on_msg)

    runner = WorkflowRunner()
    st.session_state.workflow_runner = runner
    st.session_state.workflow_config_active = workflow_config

    runner.start(_run_workflow_func, workflow_config, _graph, _wspath)
    st.rerun()

# --- Poll running workflow ---
if "workflow_runner" in st.session_state:
    runner = st.session_state.workflow_runner
    handler = st.session_state.approval_handler

    # Drain live messages from background thread → chat history
    for msg in handler.drain_messages():
        st.session_state.messages.append(msg)

    # Re-render all messages (including newly drained ones)
    for m in st.session_state.messages:
        render_chat_message(m["role"], m["content"])

    if handler.has_pending():
        render_approval_ui(handler)
    elif runner.is_done:
        if runner.error:
            resp = f"Workflow failed: {runner.error}"
        else:
            resp, _ = runner.result or ("No result", [])

        st.session_state.messages.append({"role": "assistant", "content": resp})
        render_chat_message("assistant", resp)

        wc = st.session_state.workflow_config_active
        metadata = {
            "workflow_type": wc["type"],
            "query": wc.get("query"),
            "path": wc.get("path"),
        }
        if wc["type"] == "full":
            metadata.update(
                {
                    "data_path": wc.get("data_path"),
                    "run_data": wc.get("run_data"),
                    "run_exp": wc.get("run_exp"),
                }
            )
        memo_dir = save_chat_history(
            st.session_state.messages, workflow_type=wc["type"], metadata=metadata
        )
        st.session_state.last_saved_memo = str(memo_dir)

        set_on_message_callback(None)
        del st.session_state.workflow_runner
        del st.session_state.workflow_config_active
        del st.session_state.approval_handler
        st.rerun()
    else:
        render_chat_message("assistant", "Workflow is running...")
        time.sleep(2)
        st.rerun()
