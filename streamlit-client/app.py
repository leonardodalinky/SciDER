"""SciDER Research Assistant — Streamlit App (main entry point)."""

import json
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
from workflow.usage_tracker import UsageTracker

from scider.agents import ideation_agent
from scider.core import approval as approval_module
from scider.core.types import add_message_listener, remove_message_listener, set_on_message_callback
from scider.default.models import ModelCatalog, parse_model_spec, register_role

# ==================== Page config ====================

st.set_page_config(page_title="SciDER — Research Assistant", page_icon="🍎", layout="centered")

st.markdown(
    """
    <style>
    /* ══════════════════════════════════════════════════════════
       BUTTONS  —  the biggest visual change
       Streamlit 1.30+ buttons need direct testid targeting
       ══════════════════════════════════════════════════════════ */

    /* All regular (secondary) buttons → vibrant blue outline
       Multiple selectors for Streamlit version compatibility */
    [data-testid="stBaseButton-secondary"],
    [data-testid="baseButton-secondary"],
    .stButton > button,
    button[kind="secondary"] {
        background: white !important;
        color: #4361ee !important;
        border: 2px solid #4361ee !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: background 0.15s ease, color 0.15s ease,
                    box-shadow 0.15s ease, transform 0.1s ease !important;
    }
    [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="baseButton-secondary"]:hover,
    .stButton > button:hover {
        background: #4361ee !important;
        color: white !important;
        border-color: #4361ee !important;
        box-shadow: 0 4px 14px rgba(67, 97, 238, 0.35) !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stBaseButton-secondary"]:active,
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 1px 4px rgba(67, 97, 238, 0.2) !important;
    }

    /* Form submit buttons → vibrant gradient */
    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primaryFormSubmit"],
    [data-testid="baseButton-primary"],
    .stButton > button[kind="primary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 10px rgba(67, 97, 238, 0.4) !important;
        transition: box-shadow 0.15s ease, transform 0.1s ease !important;
    }
    [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stBaseButton-primaryFormSubmit"]:hover,
    [data-testid="baseButton-primary"]:hover {
        box-shadow: 0 5px 20px rgba(67, 97, 238, 0.55) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary form submit (e.g. Cancel) → outlined */
    [data-testid="stBaseButton-secondaryFormSubmit"],
    [data-testid="baseButton-secondaryFormSubmit"] {
        background: white !important;
        color: #4361ee !important;
        border: 2px solid #4361ee !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stBaseButton-secondaryFormSubmit"]:hover {
        background: #eef0fd !important;
    }

    /* ══════════════════════════════════════════════════════════
       HEADINGS
       ══════════════════════════════════════════════════════════ */
    h1, h2, h3, h4, h5, h6 {
        color: #1e2a4a !important;
        letter-spacing: -0.3px;
    }
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4,
    [data-testid="stChatMessage"] h5,
    [data-testid="stChatMessage"] h6 {
        color: inherit !important;
        letter-spacing: normal;
    }

    /* ══════════════════════════════════════════════════════════
       BRANDED HEADER BAND
       ══════════════════════════════════════════════════════════ */
    .scider-header {
        background: linear-gradient(135deg, #1e2a4a 0%, #384166 60%, #4361ee 100%);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .scider-header-icon { font-size: 30px; line-height: 1; }
    .scider-header-title {
        color: white !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }
    .scider-header-sub {
        color: #a5b4e8;
        font-size: 12.5px;
        margin: 3px 0 0;
        line-height: 1.3;
    }

    /* ══════════════════════════════════════════════════════════
       WORKFLOW SECTION
       ══════════════════════════════════════════════════════════ */
    .workflow-section-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #4361ee;
        margin-bottom: 6px;
    }

    /* ══════════════════════════════════════════════════════════
       FORMS — colored top border
       ══════════════════════════════════════════════════════════ */
    [data-testid="stForm"] {
        border: 1px solid #d9dcf8 !important;
        border-radius: 12px !important;
        border-top: 4px solid #4361ee !important;
        padding-top: 8px !important;
    }

    /* ══════════════════════════════════════════════════════════
       CHAT MESSAGES
       ══════════════════════════════════════════════════════════ */
    .chat-agent-badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        color: #4361ee;
        background: #eef0fd;
        border: 1px solid #c5c9f8;
        border-radius: 999px;
        padding: 2px 10px;
        margin-bottom: 6px;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }

    .chat-meta-msg {
        color: #64748b;
        font-size: 13px;
        background: #f5f6ff;
        border-left: 3px solid #c5c9f8;
        border-radius: 0 4px 4px 0;
        padding: 6px 12px;
        margin: 6px 0;
    }

    /* ══════════════════════════════════════════════════════════
       HERO / SETUP CARD  — teal-to-blue gradient for warmth
       ══════════════════════════════════════════════════════════ */
    .scider-hero {
        background: linear-gradient(135deg, #e8f4f8 0%, #eef0fd 60%, #f0f4ff 100%);
        border: 1px solid #b8d8e8;
        border-top: 4px solid #0891b2;
        border-radius: 12px;
        padding: 24px 28px;
        margin: 16px 0 24px;
    }
    .scider-hero h2 {
        color: #0c4a6e !important;
        margin-top: 0;
        font-size: 22px;
        letter-spacing: -0.4px;
    }
    .scider-hero p {
        color: #334155;
        font-size: 15px;
        line-height: 1.65;
        margin: 0;
    }

    /* ══════════════════════════════════════════════════════════
       RUNNING INDICATOR  — amber/warm to show active state
       ══════════════════════════════════════════════════════════ */
    .running-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        background: linear-gradient(90deg, #fff7ed 0%, #fffbf5 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 14px;
        color: #78350f;
        font-weight: 500;
        margin: 8px 0;
    }

    /* ══════════════════════════════════════════════════════════
       APPROVAL CARD  — indigo accent
       ══════════════════════════════════════════════════════════ */
    .approval-card {
        background: linear-gradient(90deg, #eef0fd 0%, #f5f6ff 100%);
        border-left: 4px solid #6366f1;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .approval-card-title {
        font-size: 16px;
        font-weight: 700;
        color: #1e2a4a;
        margin: 0 0 4px;
    }
    .approval-card-subtitle {
        font-size: 14px;
        color: #6366f1;
        font-weight: 600;
        margin: 0;
    }

    /* ══════════════════════════════════════════════════════════
       SUCCESS / INFO CALLOUTS  — green accent
       ══════════════════════════════════════════════════════════ */
    [data-testid="stSuccess"] {
        background: #f0fdf4 !important;
        border-left: 4px solid #22c55e !important;
        border-radius: 0 8px 8px 0 !important;
    }
    [data-testid="stInfo"] {
        background: #eff6ff !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 0 8px 8px 0 !important;
    }
    [data-testid="stWarning"] {
        background: #fffbeb !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 0 8px 8px 0 !important;
    }
    [data-testid="stError"] {
        background: #fef2f2 !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 0 8px 8px 0 !important;
    }

    /* ══════════════════════════════════════════════════════════
       MISC
       ══════════════════════════════════════════════════════════ */
    hr { border-color: #e2e5f8 !important; }
    /* Sidebar-less layout: add subtle page border */
    [data-testid="stAppViewContainer"] > section:first-child {
        border-right: 1px solid #e8eaf0;
    }

    /* ══════════════════════════════════════════════════════════
       WORKFLOW STATUS BANNERS
       ══════════════════════════════════════════════════════════ */
    .status-banner {
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
        font-weight: 700;
        font-size: 14px;
    }
    .status-success { background: #f0fdf4; border-left: 4px solid #22c55e; color: #15803d; }
    .status-warn    { background: #fffbeb; border-left: 4px solid #f59e0b; color: #b45309; }
    .status-fail    { background: #fef2f2; border-left: 4px solid #ef4444; color: #b91c1c; }

    /* ══════════════════════════════════════════════════════════
       IDEATION RESULT — search metrics + ranked idea cards
       ══════════════════════════════════════════════════════════ */
    .ir-section-title {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.6px;
        color: #4361ee;
        text-transform: uppercase;
        margin: 16px 0 8px;
    }
    .search-metrics {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 4px;
    }
    .sm-item {
        flex: 1;
        min-width: 92px;
        text-align: center;
        background: linear-gradient(135deg, #eef0fd 0%, #f5f6ff 100%);
        border: 1px solid #d9dcf8;
        border-radius: 10px;
        padding: 10px 8px;
    }
    .sm-val { font-size: 21px; font-weight: 800; color: #1e2a4a; line-height: 1.1; }
    .sm-lbl {
        font-size: 10px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 3px;
    }
    .idea-card {
        border: 1px solid #e2e5f8;
        border-left: 4px solid #4361ee;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        background: #ffffff;
    }
    .idea-card-head {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .idea-rank {
        font-size: 13px;
        font-weight: 800;
        color: #4361ee;
        background: #eef0fd;
        border-radius: 6px;
        padding: 1px 7px;
    }
    .idea-title { font-size: 15px; font-weight: 700; color: #1e2a4a; flex: 1; }
    .idea-op {
        font-size: 10px;
        font-weight: 700;
        border-radius: 999px;
        padding: 2px 9px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .idea-novelty {
        font-size: 10px;
        font-weight: 700;
        color: #3a0ca3;
        background: #eef0fd;
        border-radius: 999px;
        padding: 2px 9px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .idea-composite {
        font-size: 20px;
        font-weight: 800;
        line-height: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .idea-composite small {
        font-size: 8px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-top: 2px;
    }
    .idea-desc { font-size: 13.5px; color: #334155; line-height: 1.55; margin: 8px 0 2px; }
    .sc-bars { margin: 8px 0 2px; }
    .sc-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
    .sc-label { font-size: 11px; font-weight: 600; color: #64748b; width: 80px; flex-shrink: 0; }
    .sc-track {
        flex: 1;
        height: 7px;
        background: #eef0f6;
        border-radius: 999px;
        overflow: hidden;
    }
    .sc-fill { display: block; height: 100%; border-radius: 999px; }
    .sc-val {
        font-size: 11px;
        font-weight: 700;
        color: #475569;
        width: 32px;
        text-align: right;
        flex-shrink: 0;
    }
    .idea-extra { font-size: 12px; color: #475569; margin-top: 5px; line-height: 1.5; }
    .idea-extra-lbl {
        font-weight: 700;
        color: #4361ee;
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 0.3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_TRUNCATE_LEN = 800

_AGENT_LABELS = {
    # Ideation
    "ideation": "💡 Ideation Agent",
    "idea_search": "🧬 Idea Search",
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


def _format_tool_call(tc) -> str:
    """Render a tool_call as a compact one-line invocation summary (HTML)."""
    try:
        fn = tc.function.name
    except Exception:
        fn = "tool"
    args_str = ""
    try:
        args = json.loads(tc.function.arguments or "{}")
        if isinstance(args, dict):
            parts = []
            for k, v in list(args.items())[:4]:
                vs = str(v).replace("\n", " ")
                if len(vs) > 60:
                    vs = vs[:60] + "…"
                parts.append(f"{k}={vs}")
            args_str = ", ".join(parts)
    except Exception:
        pass
    return f"🔧 <b>{fn}</b>(<code>{args_str}</code>)"


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
    truncate: bool = True,
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
                f"<div class='chat-meta-msg'>{content}</div>",
                unsafe_allow_html=True,
            )
        elif truncate and len(content) > _TRUNCATE_LEN:
            first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
            label = (first_line[:117] + "…") if len(first_line) > 117 else (first_line or "Response")
            with st.expander(label, expanded=False):
                st.markdown(content)
        elif content:
            st.markdown(content, unsafe_allow_html=True)

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
                truncate=not m.get("is_report", False),
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


# ==================== Usage stats display ====================

_PROVIDER_COLORS = {
    "gemini": "#1a73e8",
    "anthropic": "#d97706",
    "openai": "#16a34a",
}


def _model_display(model_id: str) -> str:
    """Strip provider prefix: 'gemini/gemini-2.5-pro' → 'gemini-2.5-pro'."""
    return model_id.split("/", 1)[-1] if "/" in model_id else model_id


def render_usage_stats(rows: list[dict]) -> None:
    if not rows:
        return
    total_in = sum(r["input_tokens"] for r in rows)
    total_out = sum(r["output_tokens"] for r in rows)
    total_calls = sum(r["calls"] for r in rows)
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    total_cost_str = f"~${sum(costs):.4f}" if costs else "cost unavailable"

    with st.expander(
        f"📊 API Usage — {total_in + total_out:,} tokens · {total_cost_str}",
        expanded=False,
    ):
        # Column headers
        h1, h2, h3, h4, h5, h6 = st.columns([2.5, 3, 1.5, 1.5, 1, 1.5])
        h1.markdown("**Agent**")
        h2.markdown("**Model**")
        h3.markdown("**Input**")
        h4.markdown("**Output**")
        h5.markdown("**Calls**")
        h6.markdown("**Est. Cost**")
        st.divider()

        for r in rows:
            c1, c2, c3, c4, c5, c6 = st.columns([2.5, 3, 1.5, 1.5, 1, 1.5])
            label = _agent_label(r["role"]) or r["role"]
            model_short = _model_display(r["model"])
            cost_str = f"${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "N/A"
            c1.caption(label)
            c2.caption(f"`{model_short}`")
            c3.caption(f"{r['input_tokens']:,}")
            c4.caption(f"{r['output_tokens']:,}")
            c5.caption(str(r["calls"]))
            c6.caption(cost_str)

        st.divider()
        t1, t2, t3, t4, t5, t6 = st.columns([2.5, 3, 1.5, 1.5, 1, 1.5])
        t1.markdown("**Total**")
        t2.markdown("")
        t3.markdown(f"**{total_in:,}**")
        t4.markdown(f"**{total_out:,}**")
        t5.markdown(f"**{total_calls}**")
        t6.markdown(f"**{total_cost_str}**")

        st.caption(
            "Cost estimates use litellm's published per-token pricing and may differ from your "
            "actual bill. Input tokens shown are per-request totals (system + full history)."
        )


# ==================== Model registration ====================


def register_all_models(settings: dict) -> bool:
    """Apply provider keys + register roles via the unified ModelCatalog.

    Keys are passed directly to register_role() per session — never written to
    os.environ (which is shared across all Streamlit sessions).
    """
    # Wire the Semantic Scholar key into the module-level constant that
    # paper_search.py reads at call time. constant.S2_API_KEY is resolved once
    # at import from the env, so setting os.environ alone is too late — we must
    # mutate the attribute directly. (Env is also set for any subprocesses.)
    from scider.core import constant as _constant

    _s2_key = settings.get("s2_api_key", "")
    _constant.S2_API_KEY = _s2_key
    if _s2_key:
        os.environ["S2_API_KEY"] = _s2_key

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
    st.markdown(
        """<div class='scider-header' style='margin-bottom:16px'>
          <span class='scider-header-icon'>🍎</span>
          <div>
            <div class='scider-header-title'>SciDER</div>
            <div class='scider-header-sub'>AI-powered scientific research assistant</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )
    if _logo_src:
        _col_l, _col_c, _col_r = st.columns([1, 2, 1])
        with _col_c:
            st.image(_logo_src, width=220)
    if st.button("📂 Browse Case Studies", key="case_study_from_setup"):
        st.session_state.view_mode = "case_study"
        st.rerun()
    st.divider()
    st.markdown(
        """<div class='scider-hero'>
        <h2>Get started with SciDER</h2>
        <p>SciDER automates the full scientific research loop — from literature search and idea generation
        to data analysis, ML experiments, and paper writing. Configure at least one AI provider
        API key below to begin. You can mix providers across different agent roles.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    new_settings = render_settings_form()
    if new_settings:
        save_settings(new_settings)
        st.rerun()
    st.stop()

# --- Load saved settings ---
_settings = load_settings()

# --- Settings page (when user clicks Settings button) ---
if st.session_state.get("show_settings"):
    st.markdown(
        """<div class='scider-header' style='margin-bottom:16px'>
          <span class='scider-header-icon'>⚙️</span>
          <div>
            <div class='scider-header-title'>Settings</div>
            <div class='scider-header-sub'>Configure API keys and model assignments — stored locally only</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )
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

# --- Branded header ---
_hdr_col, _btn_col1, _btn_col2 = st.columns([4, 1.2, 1])
with _hdr_col:
    st.markdown(
        """<div class='scider-header'>
          <span class='scider-header-icon'>🍎</span>
          <div>
            <div class='scider-header-title'>SciDER</div>
            <div class='scider-header-sub'>AI-powered scientific research assistant</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )
with _btn_col1:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    if st.button("\u2699\ufe0f Settings", key="btn_settings", use_container_width=True):
        st.session_state.show_settings = True
        st.rerun()
with _btn_col2:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    if st.button("🔄 Reset", help="Clear chat history", key="btn_reset", use_container_width=True):
        cleanup_uploaded_data()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome back to **SciDER** — your AI-powered research assistant.\n\n"
                    "I can help you with four research workflows:\n"
                    "- **Ideation** — search the literature and generate novel, scored research ideas\n"
                    "- **Data Analysis** — explore datasets with AI-guided analysis and metric search\n"
                    "- **Experiment** — implement, run, and iterate on ML experiments with a coding agent\n"
                    "- **Full Pipeline** — chain all phases end-to-end, optionally producing a LaTeX paper draft\n\n"
                    "Select a workflow below to get started."
                ),
            }
        ]
        if "selected_workflow" in st.session_state:
            st.session_state.selected_workflow = None
        st.session_state.show_workspace_result = False
        st.session_state.pop("last_usage_stats", None)
        st.rerun()

# --- Per-session initialization ---
if "initialized" not in st.session_state:
    st.session_state.initialized = True

# Re-register models on each rerun so every session uses its own keys —
# ModelRegistry is a process-level singleton, so without this one user's keys
# would leak into another user's session. Skip while a workflow is running:
# the poll loop reruns every 2s, and re-registering there is pure waste (the
# background thread is already using the registry).
if "workflow_runner" not in st.session_state:
    if not register_all_models(_settings):
        st.error("Failed to register models. Please check your API key in Settings.")
        st.stop()

if "ideation_graph" not in st.session_state:
    st.session_state.ideation_graph = ideation_agent.build().compile()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome to **SciDER** — your AI-powered end-to-end scientific research assistant.\n\n"
                "I can help you with four research workflows:\n"
                "- **Ideation** — search the literature and generate novel, scored research ideas\n"
                "- **Data Analysis** — explore datasets with AI-guided analysis and metric search\n"
                "- **Experiment** — implement, run, and iterate on ML experiments with a coding agent\n"
                "- **Full Pipeline** — chain all phases end-to-end, optionally producing a LaTeX paper draft\n\n"
                "Select a workflow below to get started."
            ),
        }
    ]

# Per-session workspace dir so concurrent sessions don't clobber each other's
# files (the process-wide ModelRegistry is shared, but the filesystem is not).
if "_session_id" not in st.session_state:
    import uuid

    st.session_state._session_id = uuid.uuid4().hex[:8]
_session_workspace = Path.cwd() / "workspace" / st.session_state._session_id
if "workspace_path" not in st.session_state:
    st.session_state.workspace_path = _session_workspace
if "default_workspace_path" not in st.session_state:
    st.session_state.default_workspace_path = _session_workspace
_ws = st.session_state.workspace_path
if isinstance(_ws, (str, Path)) and "scider_uploads" in str(_ws) and not Path(_ws).exists():
    cleanup_uploaded_data()

if "selected_workflow" not in st.session_state:
    st.session_state.selected_workflow = None


# ==================== Workflow selection ====================

st.markdown(
    "<p class='workflow-section-label'>🚀 Choose a Workflow</p>",
    unsafe_allow_html=True,
)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💡 Ideation", use_container_width=True, key="btn_ideation"):
        st.session_state.selected_workflow = "ideation"
        st.rerun()
    st.caption("Search literature and evolve novel research ideas using 4D scoring.")
with col2:
    if st.button("📊 Data Analysis", use_container_width=True, key="btn_data"):
        st.session_state.selected_workflow = "data"
        st.rerun()
    st.caption("Explore datasets, run AI analysis, and extract key findings.")
with col3:
    if st.button("🧪 Experiment", use_container_width=True, key="btn_experiment"):
        st.session_state.selected_workflow = "experiment"
        st.rerun()
    st.caption("Implement and iterate on ML experiments with a Claude coding agent.")
with col4:
    if st.button("🚀 Full Pipeline", use_container_width=True, key="btn_full"):
        st.session_state.selected_workflow = "full"
        st.rerun()
    st.caption("Chain all phases end-to-end, optionally producing a LaTeX paper.")

st.divider()

# --- Chat history (skip if workflow is running — polling loop handles rendering) ---
if "workflow_runner" not in st.session_state:
    render_chat_messages(st.session_state.messages)

    # Show API usage stats from the last completed workflow
    if st.session_state.get("last_usage_stats"):
        render_usage_stats(st.session_state.last_usage_stats)

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
        return run_ideation(wc, ideation_graph)
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

    tracker = UsageTracker()
    st.session_state.usage_tracker = tracker
    add_message_listener(tracker.on_message)

    # Hook every add_message() call to push to UI
    def _on_msg(msg):
        images = getattr(msg, "tool_result_images", None)
        agent = getattr(msg, "agent_sender", None)
        content = msg.content or ""

        # Surface the agent's reasoning/thinking as a muted preview line.
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning and reasoning.strip():
            preview = reasoning.strip()
            if len(preview) > 400:
                preview = preview[:400] + "…"
            handler.push_message("assistant", f"🧠 <i>Thinking:</i> {preview}", agent, is_meta=True)

        # Surface tool invocations. Assistant messages whose only payload is
        # tool_calls (empty content) were previously dropped entirely, so the
        # UI showed tool *results* with no visible cause — no file path, no
        # bash command, no search query.
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                handler.push_message("assistant", _format_tool_call(tc), agent, is_meta=True)

        if content or images:
            handler.push_message(
                msg.role or "assistant",
                content,
                agent,
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
        tb_msg = None
        if runner.error:
            resp = f"❌ Workflow failed: {runner.error}"
            if runner.traceback:
                # Persist the full traceback as its own message — it lands in
                # a collapsed expander (long content) so it's available for
                # debugging without dominating the chat.
                tb_msg = {
                    "role": "assistant",
                    "content": f"⚠️ Error traceback\n\n```\n{runner.traceback}\n```",
                }
        else:
            resp, _ = runner.result or ("No result", [])

        st.session_state.messages.append(
            {"role": "assistant", "content": resp, "is_report": True}
        )
        if tb_msg:
            st.session_state.messages.append(tb_msg)
        render_chat_message("assistant", resp, None, truncate=False)

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

        # Collect and store usage stats before tearing down
        tracker = st.session_state.pop("usage_tracker", None)
        if tracker:
            remove_message_listener(tracker.on_message)
            st.session_state.last_usage_stats = tracker.get_rows()

        set_on_message_callback(None)
        del st.session_state.workflow_runner
        del st.session_state.workflow_config_active
        del st.session_state.approval_handler
        st.rerun()
    else:
        # Live status: current agent, elapsed time, running token/cost total.
        elapsed = int(runner.elapsed)
        mins, secs = divmod(elapsed, 60)
        elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        last_agent = next(
            (m.get("agent") for m in reversed(st.session_state.messages) if m.get("agent")),
            None,
        )
        agent_str = _agent_label(last_agent) if last_agent else "starting up…"
        parts = [f"⚙️ <b>{agent_str}</b>", f"⏱ {elapsed_str}"]
        tracker = st.session_state.get("usage_tracker")
        if tracker is not None:
            toks = tracker.total_tokens
            if toks:
                cost = tracker.total_cost
                cost_str = f" · ~${cost:.4f}" if cost else ""
                parts.append(f"{toks:,} tokens{cost_str}")
        render_chat_message(
            "assistant",
            "<div class='running-indicator'>" + " &nbsp;·&nbsp; ".join(parts) + "</div>",
            None,
        )
        time.sleep(2)
        st.rerun()
