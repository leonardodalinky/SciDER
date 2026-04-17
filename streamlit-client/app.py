"""SciDER Research Assistant — Streamlit App (main entry point)."""

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Load .env BEFORE setting any defaults so .env values take precedence.
# override=True ensures .env wins over any inherited shell env vars — otherwise
# a stale `export CODING_AGENT_VERSION=claude_sdk` in the parent shell would
# silently override what's in .env.
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        load_dotenv()
except Exception:
    pass

os.environ.setdefault("CODING_AGENT_VERSION", "claude_sdk")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from components.display import render_approval_ui
from forms.case_study import render_case_study_viewer
from forms.data import render_form as data_form
from forms.data import run_data, run_hypo_data
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
from scider.core.types import set_on_message_callback
from scider.default.models import ModelCatalog, parse_model_spec, register_role

# ==================== Page config ====================

st.set_page_config(page_title="SciDER Chat", page_icon="🍎", layout="centered")

st.markdown(
    """
    <style>
    h1, h2, h3, h4, h5, h6 { color: #384166 !important; }
    [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] h4,
    [data-testid="stChatMessage"] h5, [data-testid="stChatMessage"] h6 {
        color: inherit !important;
    }
    .chat-agent-badge {
        font-size: 12px;
        color: #777;
        margin-bottom: 2px;
        font-weight: 600;
    }
    .chat-meta-msg {
        color: #888;
        font-size: 13px;
        border-left: 3px solid #ddd;
        padding-left: 10px;
        margin: 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_TRUNCATE_LEN = 800

_AGENT_LABELS = {
    # Ideation
    "ideation": "💡 Ideation Agent",
    # Data
    "data": "📊 Data Agent",
    "data_agent": "📊 Data Agent",
    "paper_search": "📖 Paper Search",
    "metric_search": "📏 Metric Search",
    "metric_extractor": "📏 Metric Extractor",
    "dataset_search": "🗃️ Dataset Search",
    # Experiment
    "experiment": "🧪 Experiment Agent",
    "experiment_agent": "🧪 Experiment Agent",
    "experiment_coding": "💻 Coding Subagent",
    # Writing
    "writing": "📝 Writing Agent",
    "writing_agent": "📝 Writing Agent",
    # Coding backends
    "claude_agent_sdk": "🤖 Claude Agent SDK",
    "claude_code": "🤖 Claude Code",
    "summary_node": "📝 Summary",
    # Critic & system
    "critic": "🧐 Critic Agent",
    "history": "🗜️ History Compression",
    "user_approval": "👤 User Approval",
}


def _agent_label(agent: str | None) -> str:
    if not agent:
        return ""
    return _AGENT_LABELS.get(agent, agent)


def _render_tool_images(images: list[dict]) -> None:
    """Decode base64 tool-result images and render them under the chat bubble."""
    import base64 as _b64

    for i, img in enumerate(images):
        data = img.get("data")
        media_type = img.get("media_type", "image/png")
        if not data:
            continue
        try:
            raw = _b64.b64decode(data)
        except Exception:
            st.caption(f"(image {i + 1} could not be decoded)")
            continue
        st.image(raw, caption=f"{media_type} ({len(raw):,} bytes)")


def render_chat_message(
    role: str,
    content: str,
    agent: str | None = None,
    is_meta: bool = False,
    is_tool: bool = False,
    tool_name: str | None = None,
    images: list[dict] | None = None,
):
    """Render a single chat message as a Streamlit chat_message with markdown."""
    # Map role for st.chat_message (only supports "user", "assistant", "ai", "human")
    display_role = role
    if role == "tool":
        display_role = "assistant"

    with st.chat_message(display_role):
        # Badge: tool result or agent name
        if is_tool and tool_name:
            st.markdown(
                f"<div class='chat-agent-badge'>🔧 {tool_name}</div>",
                unsafe_allow_html=True,
            )
        elif agent:
            label = _agent_label(agent)
            st.markdown(f"<div class='chat-agent-badge'>{label}</div>", unsafe_allow_html=True)

        # Meta messages: render as muted system info
        if is_meta:
            st.markdown(
                f"<div style='color: #888; font-size: 13px; border-left: 3px solid #ddd; "
                f"padding-left: 10px; margin: 4px 0;'>{content}</div>",
                unsafe_allow_html=True,
            )
        elif len(content) > _TRUNCATE_LEN:
            preview_lines = content[:200].split("\n")
            label = " | ".join(line.strip() for line in preview_lines if line.strip())[:200]
            with st.expander(f"{label}...", expanded=False):
                st.markdown(content)
        elif content:
            st.markdown(content)

        # Tool-result images (e.g. from Read on a PNG) rendered below text
        if images:
            _render_tool_images(images)


def render_chat_messages(messages: list[dict]):
    """Render messages, grouping consecutive messages from the same agent."""
    if not messages:
        return

    groups: list[list[dict]] = []
    for m in messages:
        agent = m.get("agent")
        role = m["role"]
        # Group consecutive assistant messages from the same agent (non-None)
        if (
            groups
            and agent
            and role == "assistant"
            and groups[-1][0]["role"] == "assistant"
            and groups[-1][0].get("agent") == agent
        ):
            groups[-1].append(m)
        else:
            groups.append([m])

    for group in groups:
        if len(group) == 1:
            m = group[0]
            render_chat_message(
                m["role"],
                m["content"],
                m.get("agent"),
                is_meta=m.get("is_meta", False),
                is_tool=m.get("is_tool", False),
                tool_name=m.get("tool_name"),
                images=m.get("images"),
            )
        else:
            # Multiple consecutive messages from same agent — collapse
            agent = group[0].get("agent")
            label = _agent_label(agent) or "Assistant"
            with st.expander(f"{label} ({len(group)} messages)", expanded=True):
                for m in group:
                    render_chat_message(
                        m["role"],
                        m["content"],
                        is_meta=m.get("is_meta", False),
                        is_tool=m.get("is_tool", False),
                        tool_name=m.get("tool_name"),
                        images=m.get("images"),
                    )


# ==================== Model registration ====================


def register_all_models(settings: dict) -> bool:
    """Apply provider keys + register roles via the unified ModelCatalog.

    Keys are passed directly to register_role() per session — never written to
    os.environ (which is shared across all Streamlit sessions).
    """
    # Build {env_var: key} from this session's settings.
    key_map: dict[str, str] = {}
    for env_var, settings_key in (
        ("GEMINI_API_KEY", "gemini_api_key"),
        ("OPENAI_API_KEY", "openai_api_key"),
        ("ANTHROPIC_API_KEY", "anthropic_api_key"),
    ):
        val = settings.get(settings_key, "")
        if val:
            key_map[env_var] = val

    if not key_map:
        return False

    ModelCatalog.load()

    def _resolve_key(model_id: str) -> str | None:
        """Resolve the API key for a catalog model from this session's keys."""
        base_id, _ = parse_model_spec(model_id)
        entry = ModelCatalog.get(base_id)
        if entry is None:
            return None
        for env_var in entry.requires_env:
            key = key_map.get(env_var) or os.getenv(env_var)
            if key:
                return key
        return None

    def _model_has_key(model_id: str) -> bool:
        return _resolve_key(model_id) is not None

    # Register role defaults (only those for which we have a key).
    try:
        import yaml

        from scider.default.models.catalog import DEFAULT_ROLE_DEFAULTS_PATH

        data = yaml.safe_load(DEFAULT_ROLE_DEFAULTS_PATH.read_text()) or {}
        for role, model_spec in (data.get("defaults") or {}).items():
            spec = str(model_spec)
            if _model_has_key(spec):
                register_role(role, spec, api_key=_resolve_key(spec))
    except Exception as e:
        from loguru import logger as _lg

        _lg.warning("Failed to register role defaults: {}", e)

    role_models = settings.get("model_roles", {})

    # CLAUDE_SDK_MODEL env var drives the claude_sdk coding backend.
    # This is the one env var we must set because the SDK reads it directly.
    coding_version = os.getenv("CODING_AGENT_VERSION", "claude_sdk")
    coding_model_id = role_models.get("experiment_coding", "")
    if coding_version in ("v3", "claude_sdk") and coding_model_id:
        entry = ModelCatalog.get(coding_model_id)
        if entry is not None:
            os.environ["CLAUDE_SDK_MODEL"] = entry.litellm_id.split("/", 1)[-1]

    # Register user overrides.
    for role, mid in role_models.items():
        if not mid:
            continue
        key = _resolve_key(mid)
        if key is None:
            from loguru import logger as _lg

            _lg.warning("Skipping role '{}' -> '{}': no API key available", role, mid)
            continue
        register_role(role, mid, api_key=key)

    return True


# ==================== Settings gate ====================
# .env is loaded at module top so CODING_AGENT_VERSION etc. are honored.

# --- Case study view mode (before settings gates so it works without API keys) ---
if st.session_state.get("view_mode") == "case_study":
    render_case_study_viewer()
    if st.button("⬅️ Back", key="case_study_back"):
        st.session_state.view_mode = None
        st.rerun()
    st.stop()

# --- First visit: show settings page ---
if not has_settings():
    _logo = Path(__file__).parent.parent / "static" / "images" / "scider_logo.webp"
    _logo_src = None
    if _logo.exists():
        _logo_src = str(_logo)
    else:
        _logo_url = "https://raw.githubusercontent.com/leonardodalinky/SciDER/main/static/images/scider_logo.webp"
        try:
            import urllib.request

            urllib.request.urlopen(_logo_url, timeout=3)
            _logo_src = _logo_url
        except Exception:
            pass
    if _logo_src:
        _col_l, _col_c, _col_r = st.columns([1, 2, 1])
        with _col_c:
            st.image(_logo_src, width=300)
    st.title("SciDER Research Assistant")
    if st.button("📂 Browse Case Studies", key="case_study_from_setup"):
        st.session_state.view_mode = "case_study"
        st.rerun()
    st.divider()
    st.info("Configure your API keys to get started.")
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
    if st.button("📂 Browse Case Studies", key="case_study_from_settings"):
        st.session_state.show_settings = False
        st.session_state.view_mode = "case_study"
        st.rerun()
    st.divider()
    new_settings = render_settings_form(current_settings=_settings)
    if new_settings:
        save_settings(new_settings)
        st.session_state.show_settings = False
        # Clear caches so next rerun re-registers models with new keys
        st.session_state.pop("_settings_cache", None)
        st.session_state.pop("ideation_graph", None)
        st.rerun()
    col_cancel, col_reset = st.columns(2)
    with col_cancel:
        if st.button("⬅️ Cancel", key="btn_cancel_settings", use_container_width=True):
            st.session_state.show_settings = False
            st.rerun()
    with col_reset:
        if st.button(
            "🗑️ Reset All Settings",
            key="btn_clear_settings",
            use_container_width=True,
        ):
            from settings import clear_settings

            clear_settings()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    st.stop()


# ==================== Apply settings ====================

# --- Title bar ---
col_title, col_settings, col_reset = st.columns([4, 1.2, 1])
with col_title:
    st.title("SciDER Research Assistant")
with col_settings:
    if st.button("\u2699\ufe0f Settings", key="btn_settings"):
        st.session_state.show_settings = True
        st.rerun()
with col_reset:
    if st.button("\U0001f504 Reset", help="Clear chat history", key="btn_reset"):
        cleanup_uploaded_data()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello. I can run ideation, data analysis, experiments, or a full workflow.\n\nPlease select a workflow type below to get started.",
            }
        ]
        if "selected_workflow" in st.session_state:
            st.session_state.selected_workflow = None
        st.session_state.show_workspace_result = False
        st.rerun()

# --- Per-session initialization ---
if "initialized" not in st.session_state:
    st.session_state.initialized = True

# Re-register models on EVERY rerun so each session uses its own keys.
# ModelRegistry is a process-level singleton — without this, one user's
# keys would leak into another user's session.
if not register_all_models(_settings):
    st.error("Failed to register models. Please check your API key in Settings.")
    st.stop()

if "ideation_graph" not in st.session_state:
    st.session_state.ideation_graph = ideation_agent.build().compile()

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
    if st.button("\U0001f4a1 Ideation", use_container_width=True, key="btn_ideation"):
        st.session_state.selected_workflow = "ideation"
        st.rerun()
with col2:
    if st.button("\U0001f4ca Data Analysis", use_container_width=True, key="btn_data"):
        st.session_state.selected_workflow = "data"
        st.rerun()
with col3:
    if st.button("\U0001f9ea Experiment", use_container_width=True, key="btn_experiment"):
        st.session_state.selected_workflow = "experiment"
        st.rerun()
with col4:
    if st.button("\U0001f680 Full Workflow", use_container_width=True, key="btn_full"):
        st.session_state.selected_workflow = "full"
        st.rerun()

st.divider()

# --- Chat history (skip if workflow is running — polling loop handles rendering) ---
if "workflow_runner" not in st.session_state:
    render_chat_messages(st.session_state.messages)

    # Show workspace files after experiment/full workflow completion
    if st.session_state.get("show_workspace_result"):
        from components.file_browser import render_file_browser, render_workspace_download

        _ws_result = st.session_state.workspace_path
        if _ws_result and Path(_ws_result).exists():
            st.divider()
            st.subheader("📁 Workspace Output")
            render_file_browser(Path(_ws_result), key_prefix="wf_fb")
            render_workspace_download(Path(_ws_result), key_prefix="wf_dl")


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
    elif wtype == "data_hypo":
        return run_hypo_data(wc["feature_desc"], wc["num_rows"], wc["query"], workspace_path)
    elif wtype == "experiment":
        return run_experiment(wc["query"], wc.get("path"), workspace_path)
    elif wtype == "full":
        return run_full(wc, workspace_path)
    return "Unknown workflow type", []


# --- Launch background workflow ---
if workflow_config and "workflow_runner" not in st.session_state:
    st.session_state.show_workspace_result = False
    st.session_state.messages.append({"role": "user", "content": _build_user_msg(workflow_config)})

    _graph = st.session_state.ideation_graph
    _wspath = st.session_state.workspace_path

    handler = StreamlitApprovalHandler()
    st.session_state.approval_handler = handler
    approval_module.set_handler(handler)

    # Hook every add_message() call to push to UI
    def _on_msg(msg):
        images = getattr(msg, "tool_result_images", None)
        if msg.content or images:
            handler.push_message(
                msg.role or "assistant",
                msg.content or "",
                getattr(msg, "agent_sender", None),
                is_meta=getattr(msg, "is_meta", False),
                is_tool=msg.role == "tool",
                tool_name=getattr(msg, "tool_name", None),
                images=images,
            )

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
    render_chat_messages(st.session_state.messages)

    if handler.has_pending():
        render_approval_ui(handler)
    elif runner.is_done:
        if runner.error:
            resp = f"Workflow failed: {runner.error}"
        else:
            resp, _ = runner.result or ("No result", [])

        st.session_state.messages.append({"role": "assistant", "content": resp})
        render_chat_message("assistant", resp, None)

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

        # Show workspace browser for any workflow that writes files (data/experiment/full)
        if wc["type"] in ("data", "data_hypo", "experiment", "full"):
            st.session_state.show_workspace_result = True

        set_on_message_callback(None)
        del st.session_state.workflow_runner
        del st.session_state.workflow_config_active
        del st.session_state.approval_handler
        st.rerun()
    else:
        render_chat_message("assistant", "Workflow is running...", None)
        time.sleep(2)
        st.rerun()
        render_chat_message("assistant", "Workflow is running...", None)
        time.sleep(2)
        st.rerun()
