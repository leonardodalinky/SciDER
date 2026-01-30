"""
SciEvo Streamlit Chat Interface - Enhanced Version

A ChatGPT-like interface with unified conversation view showing all agent messages.

## Conversation Logging

All agents and subagents log their messages to a unified conversation log that is
displayed in real-time during workflow execution. Messages are captured via the
WorkflowMonitor callback system.

To log agent messages from within the workflow code:
    from workflow_monitor import get_monitor, PhaseType

    monitor = get_monitor()
    monitor.log_update(
        phase=PhaseType.DATA_EXECUTION,
        status="progress",
        message="Analyzing dataset structure...",
        agent_name="Data Agent",
        message_type="thought"  # Options: status, thought, action, result, error
    )

Message types:
- status: General status updates (blue background)
- thought: Agent reasoning/planning (purple background)
- action: Tool calls or actions taken (orange background)
- result: Results or completions (green background)
- error: Errors or warnings (red background)
"""

import html
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

import threading
import time as time_module

from logger_handler import get_log_handler, reset_log_handler, setup_streamlit_logging
from workflow_monitor import PhaseType, get_monitor, reset_monitor

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
    if "conversation_log" not in st.session_state:
        st.session_state.conversation_log = []
    if "last_update_count" not in st.session_state:
        st.session_state.last_update_count = 0
    if "show_agent_conversations" not in st.session_state:
        st.session_state.show_agent_conversations = False
    if "show_logs" not in st.session_state:
        st.session_state.show_logs = False
    if "log_level_filter" not in st.session_state:
        st.session_state.log_level_filter = "ALL"


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


# ==================== Conversation Display ====================
def display_conversation_log(grouped: bool = False):
    """Display unified conversation log from all agents.

    Args:
        grouped: If True, group messages by agent. If False, show chronological order.
    """
    if not st.session_state.conversation_log:
        st.info("No messages yet. Waiting for agents to start...")
        return

    if grouped:
        # Group messages by agent
        agent_messages = {}
        for entry in st.session_state.conversation_log:
            agent_name = entry.get("agent_name", "System")
            if agent_name not in agent_messages:
                agent_messages[agent_name] = []
            agent_messages[agent_name].append(entry)

        # Display each agent's conversation in an expander
        for agent_name, messages in sorted(agent_messages.items()):
            with st.expander(f"🤖 {agent_name} ({len(messages)} messages)", expanded=False):
                for entry in messages:
                    timestamp = time_module.strftime(
                        "%H:%M:%S", time_module.localtime(entry["timestamp"])
                    )
                    message = entry["message"]
                    message_type = entry.get("message_type", "status")

                    # Choose styling based on message type
                    if message_type == "error":
                        icon = "❌"
                        color = "#ffebee"
                    elif message_type == "result":
                        icon = "✅"
                        color = "#e8f5e9"
                    elif message_type == "action":
                        icon = "⚡"
                        color = "#fff3e0"
                    elif message_type == "thought":
                        icon = "💭"
                        color = "#f3e5f5"
                    else:
                        icon = "ℹ️"
                        color = "#e3f2fd"

                    # Display message
                    st.markdown(
                        f"""
                        <div id="21" style="background-color: {color}; padding: 8px; border-radius: 5px; margin-bottom: 8px; border-left: 3px solid #666;">
                            <div id="20" style="font-size: 0.75em; color: #666; margin-bottom: 3px;">
                                <span style="font-family: monospace;">{timestamp}</span>
                            </div>
                            <div id="19" style="color: #333; font-size: 0.9em;">
                                {icon} {message}
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        # Display in chronological order (original behavior)
        st.markdown("### 💬 Agent Conversation")
        conversation_container = st.container()

        with conversation_container:
            for entry in st.session_state.conversation_log:
                timestamp = time_module.strftime(
                    "%H:%M:%S", time_module.localtime(entry["timestamp"])
                )
                agent_name = entry.get("agent_name", "System")
                message = entry["message"]
                message_type = entry.get("message_type", "status")

                # Choose styling based on message type and agent
                if message_type == "error":
                    icon = "❌"
                    color = "#ffebee"
                elif message_type == "result":
                    icon = "✅"
                    color = "#e8f5e9"
                elif message_type == "action":
                    icon = "⚡"
                    color = "#fff3e0"
                elif message_type == "thought":
                    icon = "💭"
                    color = "#f3e5f5"
                else:
                    icon = "ℹ️"
                    color = "#e3f2fd"

                # Display message
                st.markdown(
                    f"""
                    <div id="17" style="background-color: {color}; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 3px solid #666;">
                        <div id="18" style="font-size: 0.8em; color: #666; margin-bottom: 5px;">
                            <strong>{icon} {agent_name}</strong> · <span style="font-family: monospace;">{timestamp}</span>
                        </div>
                        <div id="16" style="color: #333;">
                            {message}
                    """,
                    unsafe_allow_html=True,
                )


def display_agent_conversations_dialog():
    """Display all agent conversations in a dialog-like expander, grouped by agent and node."""
    if not st.session_state.conversation_log:
        st.info("No agent conversations available yet.")
        return

    # Group messages by agent, then by node
    agent_node_messages = {}
    for entry in st.session_state.conversation_log:
        agent_name = entry.get("agent_name", "System")
        node_name = entry.get("node_name") or "general"  # Use "general" if no node_name

        if agent_name not in agent_node_messages:
            agent_node_messages[agent_name] = {}
        if node_name not in agent_node_messages[agent_name]:
            agent_node_messages[agent_name][node_name] = []

        agent_node_messages[agent_name][node_name].append(entry)

    # Summary
    total_messages = sum(
        sum(len(msgs) for msgs in nodes.values()) for nodes in agent_node_messages.values()
    )
    total_nodes = sum(len(nodes) for nodes in agent_node_messages.values())
    st.info(
        f"📊 {len(agent_node_messages)} agent(s), {total_nodes} node(s), {total_messages} total messages"
    )

    # Display each agent's conversations
    for agent_name, nodes in sorted(agent_node_messages.items()):
        with st.expander(
            f"🤖 {agent_name} ({len(nodes)} nodes, {sum(len(msgs) for msgs in nodes.values())} messages)",
            expanded=False,
        ):
            # Display each node's messages
            for node_name, messages in sorted(nodes.items()):
                node_display_name = node_name if node_name != "general" else "General Messages"
                with st.expander(
                    f"⚙️ {node_display_name} ({len(messages)} messages)", expanded=False
                ):
                    for entry in messages:
                        timestamp = time_module.strftime(
                            "%H:%M:%S", time_module.localtime(entry["timestamp"])
                        )
                        message = entry["message"]
                        message_type = entry.get("message_type", "status")
                        phase = entry.get("phase", "")
                        intermediate_output = entry.get("intermediate_output")

                        # Choose styling based on message type
                        if message_type == "error":
                            icon = "❌"
                            color = "#ffebee"
                        elif message_type == "result":
                            icon = "✅"
                            color = "#e8f5e9"
                        elif message_type == "action":
                            icon = "⚡"
                            color = "#fff3e0"
                        elif message_type == "thought":
                            icon = "💭"
                            color = "#f3e5f5"
                        else:
                            icon = "ℹ️"
                            color = "#e3f2fd"

                        # Display message
                        print(f"message: {message}")
                        safe_message = html.escape(message)
                        print(f"safe_message: {safe_message}")
                        st.markdown(
                            f"""
                            <div id="7" style="background-color: {color}; padding: 8px; border-radius: 5px; margin-bottom: 8px; border-left: 3px solid #666;">
                                <div id="8" style="font-size: 0.75em; color: #666; margin-bottom: 3px;">
                                    <span id="9" style="font-family: monospace;">{timestamp}</span>
                                    {f' · <span id="10" style="font-style: italic;">{phase}</span>' if phase else ''}
                                    {f' · <span id="11" style="font-weight: bold;">{entry.get("status", "")}</span>' if entry.get("status") else ''}
                                </div>
                                <div id="12" style="color: #333; font-size: 0.9em;">
                                    {icon} {safe_message}
                            """,
                            unsafe_allow_html=True,
                        )

                        # Display intermediate output if available
                        if intermediate_output:
                            with st.expander("📋 Intermediate Output", expanded=False):
                                if isinstance(intermediate_output, dict):
                                    # Display state changes
                                    if "state_before" in intermediate_output:
                                        st.markdown("**📥 State Before Execution:**")
                                        st.json(intermediate_output["state_before"])

                                    if "state_after" in intermediate_output:
                                        st.markdown("**📤 State After Execution:**")
                                        st.json(intermediate_output["state_after"])

                                    if (
                                        "messages_added" in intermediate_output
                                        and intermediate_output["messages_added"]
                                    ):
                                        st.markdown("**💬 Messages Added:**")
                                        for msg_info in intermediate_output["messages_added"]:
                                            msg_index = msg_info.get("index", "?")
                                            msg_preview = msg_info.get("preview", "")
                                            st.code(
                                                f"Message {msg_index}: {msg_preview[:200]}...",
                                                language=None,
                                            )

                                    if (
                                        "node_history" in intermediate_output
                                        and intermediate_output["node_history"]
                                    ):
                                        st.markdown("**🔄 Node History:**")
                                        node_history_str = " → ".join(
                                            intermediate_output["node_history"]
                                        )
                                        st.code(node_history_str, language=None)

                                    if "message_count" in intermediate_output:
                                        st.metric(
                                            "Message Count", intermediate_output["message_count"]
                                        )

                                    if "remaining_plans_count" in intermediate_output:
                                        st.metric(
                                            "Remaining Plans",
                                            intermediate_output["remaining_plans_count"],
                                        )

                                    if "workspace" in intermediate_output:
                                        st.markdown("**📁 Workspace:**")
                                        st.code(intermediate_output["workspace"], language=None)

                                    if "error" in intermediate_output:
                                        st.error(f"**Error:** {intermediate_output['error']}")

                                    # Display any other fields
                                    other_fields = {
                                        k: v
                                        for k, v in intermediate_output.items()
                                        if k
                                        not in [
                                            "state_before",
                                            "state_after",
                                            "messages_added",
                                            "node_history",
                                            "message_count",
                                            "remaining_plans_count",
                                            "workspace",
                                            "error",
                                        ]
                                    }
                                    if other_fields:
                                        st.markdown("**📊 Other State Info:**")
                                        st.json(other_fields)
                                else:
                                    # If intermediate_output is not a dict, display as JSON
                                    st.json(intermediate_output)


def _extract_node_name_from_message(message: str, phase: str) -> str | None:
    """Try to extract node name from message or phase."""
    # Common node patterns in messages
    node_patterns = [
        r"node ['\"]([\w_]+)['\"]",
        r"Node ([\w_]+)",
        r"(\w+)_node",
        r"(\w+) node",
    ]

    import re

    for pattern in node_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)

    # Try to infer from phase
    phase_to_node = {
        "llm_chat": "llm_chat",
        "tool_calling": "tool_calling",
        "planner": "planner",
        "replanner": "replanner",
        "mem_extraction": "mem_extraction",
        "history_compression": "history_compression",
        "generate_summary": "generate_summary",
        "gateway": "gateway",
        "init": "init",
        "monitoring": "monitoring",
        "summary": "summary",
    }

    for key, node_name in phase_to_node.items():
        if key in phase.lower():
            return node_name

    return None


def display_logs_dialog():
    """Display logger output in a dialog-like expander."""
    log_handler = get_log_handler()
    logs = log_handler.get_logs(
        level=(
            st.session_state.log_level_filter
            if st.session_state.log_level_filter != "ALL"
            else None
        )
    )

    if not logs:
        st.info("No logs available yet.")
        return

    # Summary
    total_logs = len(logs)
    level_counts = {}
    for log in logs:
        level = log["level"]
        level_counts[level] = level_counts.get(level, 0) + 1

    st.info(f"📊 Found {total_logs} log entries")

    # Level filter
    col1, col2 = st.columns([3, 1])
    with col1:
        level_filter = st.selectbox(
            "Filter by Level",
            options=["ALL", "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
            index=(
                0
                if st.session_state.log_level_filter == "ALL"
                else [
                    "ALL",
                    "TRACE",
                    "DEBUG",
                    "INFO",
                    "SUCCESS",
                    "WARNING",
                    "ERROR",
                    "CRITICAL",
                ].index(st.session_state.log_level_filter)
            ),
            key="log_level_filter_select",
        )
        if level_filter != st.session_state.log_level_filter:
            st.session_state.log_level_filter = level_filter
            st.rerun()

    with col2:
        if st.button("🗑️ Clear Logs"):
            log_handler.clear()
            st.rerun()

    # Display level counts
    if level_counts:
        st.markdown("**Log Level Distribution:**")
        level_cols = st.columns(len(level_counts))
        for i, (level, count) in enumerate(sorted(level_counts.items())):
            with level_cols[i]:
                st.metric(level, count)

    # Display logs in reverse chronological order (newest first)
    st.markdown("---")
    st.markdown("### 📋 Log Entries")

    # Create a scrollable container
    log_container = st.container()

    with log_container:
        for log in reversed(logs[-500:]):  # Limit to last 500 logs for performance
            level = log["level"]
            timestamp = time_module.strftime("%H:%M:%S", time_module.localtime(log["timestamp"]))
            message = log["message"]
            module = log.get("module", "unknown")
            function = log.get("function", "unknown")
            line = log.get("line", "?")

            # Choose styling based on log level
            if level == "ERROR" or level == "CRITICAL":
                icon = "❌"
                color = "#ffebee"
                border_color = "#f44336"
            elif level == "WARNING":
                icon = "⚠️"
                color = "#fff3e0"
                border_color = "#ff9800"
            elif level == "SUCCESS":
                icon = "✅"
                color = "#e8f5e9"
                border_color = "#4caf50"
            elif level == "INFO":
                icon = "ℹ️"
                color = "#e3f2fd"
                border_color = "#2196f3"
            elif level == "DEBUG":
                icon = "🔍"
                color = "#f3e5f5"
                border_color = "#9c27b0"
            else:  # TRACE
                icon = "🔎"
                color = "#fafafa"
                border_color = "#9e9e9e"

            # Display log entry
            with st.expander(
                f"{icon} [{level}] {timestamp} - {module}:{function}:{line}", expanded=False
            ):
                st.markdown(
                    f"""
                    <div id="13" style="background-color: {color}; padding: 8px; border-radius: 5px; border-left: 3px solid {border_color}; margin-bottom: 8px;">
                        <div id="14" style="font-size: 0.75em; color: #666; margin-bottom: 3px;">
                            <strong>Level:</strong> {level} ·
                            <strong>Time:</strong> {timestamp} ·
                            <strong>Module:</strong> {module} ·
                            <strong>Function:</strong> {function} ·
                            <strong>Line:</strong> {line}
                        </div>
                        <div id="15" style="color: #333; font-size: 0.9em; font-family: monospace; white-space: pre-wrap;">
                            {message.replace('<', '&lt;').replace('>', '&gt;')}
                    """,
                    unsafe_allow_html=True,
                )

                # Show file path if available
                if log.get("file"):
                    st.caption(f"File: {log['file']}")


def update_conversation_from_monitor():
    """Update conversation log from monitor updates."""
    monitor = get_monitor()
    updates = monitor.get_updates()

    # Only process new updates
    new_updates = updates[st.session_state.last_update_count :]

    for update in new_updates:
        # Map phase to agent name
        phase_to_agent = {
            "ideation_": "Ideation Agent",
            "data_": "Data Agent",
            "experiment_init": "Experiment Agent",
            "experiment_coding": "Coding Subagent",
            "experiment_exec": "Execution Subagent",
            "experiment_summary": "Summary Subagent",
            "experiment_analysis": "Analysis Subagent",
            "experiment_revision": "Experiment Agent",
        }

        agent_name = update.agent_name
        if not agent_name:
            # Try to infer from phase
            phase_str = str(update.phase.value)
            for prefix, name in phase_to_agent.items():
                if phase_str.startswith(prefix):
                    agent_name = name
                    break
            if not agent_name:
                agent_name = "System"

        # Extract node name if not provided
        node_name = getattr(update, "node_name", None)
        if not node_name:
            node_name = _extract_node_name_from_message(update.message, str(update.phase.value))

        # Extract intermediate output if not provided
        intermediate_output = getattr(update, "intermediate_output", None)
        if not intermediate_output and update.data:
            # Try to extract intermediate output from data field
            intermediate_output = update.data

        # Add to conversation log with node information
        st.session_state.conversation_log.append(
            {
                "timestamp": update.timestamp,
                "agent_name": agent_name,
                "message": update.message,
                "message_type": update.message_type,
                "phase": str(update.phase.value),
                "status": update.status,
                "node_name": node_name,  # Add node name
                "intermediate_output": intermediate_output,  # Add intermediate output
            }
        )

    st.session_state.last_update_count = len(updates)


# ==================== Workflow Runner ====================
class WorkflowRunner:
    """Manages workflow execution with progress tracking."""

    def __init__(self, config: dict):
        self.config = config
        self.workflow = None
        self.monitor = get_monitor()
        self.running = False
        self.result = None
        self.error = None

    def run(self):
        """Run the workflow with progress tracking."""
        try:
            # Log workflow start
            self.monitor.log_update(
                phase=PhaseType.IDEATION_LITERATURE_SEARCH,
                status="started",
                message=f"Starting workflow for: {self.config['user_query']}",
                agent_name="System",
                message_type="status",
            )

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

            # Log completion
            self.monitor.log_update(
                phase=PhaseType.COMPLETE,
                status="completed",
                message="Workflow completed successfully!",
                agent_name="System",
                message_type="result",
            )

            return result

        except Exception as e:
            # Log error to conversation
            self.monitor.log_update(
                phase=PhaseType.ERROR,
                status="error",
                message=f"Error: {str(e)}",
                agent_name="System",
                message_type="error",
            )
            raise

    def run_async(self):
        """Run workflow in background thread."""

        def _run():
            self.running = True
            try:
                self.result = self.run()
            except Exception as e:
                self.error = e
            finally:
                self.running = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread


# ==================== Main App ====================
def main():
    """Main application entry point."""
    init_session_state()

    # Setup logging capture
    if "logging_setup" not in st.session_state:
        try:
            setup_streamlit_logging(min_level="DEBUG")
            st.session_state.logging_setup = True
        except Exception as e:
            st.warning(f"⚠️ Failed to setup logging capture: {e}")
            st.session_state.logging_setup = True  # Mark as setup to avoid retrying

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
                    st.session_state.conversation_log = []
                    st.session_state.last_update_count = 0
                    st.session_state.workflow_data = {}
                    reset_monitor()
                    reset_log_handler()
                    st.rerun()

    elif st.session_state.workflow_running:
        # Display conversation log
        st.markdown("### 🔄 Workflow Running")

        # Create placeholder for conversation updates
        conversation_placeholder = st.empty()

        # Run workflow in async mode to allow UI updates
        if "workflow_thread" not in st.session_state:
            runner = WorkflowRunner(config)
            st.session_state.workflow_thread = runner
            runner.run_async()

        runner = st.session_state.workflow_thread

        # Update conversation from monitor
        update_conversation_from_monitor()

        # Display conversation in placeholder
        with conversation_placeholder.container():
            # Show grouped conversation view
            if st.session_state.show_agent_conversations:
                st.markdown("### 💬 Agent Conversations (Grouped by Agent)")
                display_agent_conversations_dialog()
                if st.button("❌ Close Grouped View", use_container_width=True):
                    st.session_state.show_agent_conversations = False
                    st.rerun()
            else:
                display_conversation_log(grouped=False)
                if st.session_state.conversation_log:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📋 View Grouped by Agent", use_container_width=True):
                            st.session_state.show_agent_conversations = True
                            st.rerun()
                    with col2:
                        if st.button("📋 View Logs", use_container_width=True):
                            st.session_state.show_logs = True
                            st.rerun()

                # Display logs dialog if requested
                if st.session_state.show_logs:
                    st.markdown("---")
                    st.markdown("### 📋 Logger Output")
                    display_logs_dialog()
                    if st.button("❌ Close Logs View", use_container_width=True):
                        st.session_state.show_logs = False
                        st.rerun()

            # Show spinner while running
            if runner.running:
                st.info("⏳ Workflow in progress... (auto-refreshing)")
                time_module.sleep(2)  # Wait before next refresh
                st.rerun()

        # Check if workflow completed or errored
        if not runner.running:
            if runner.error:
                st.error(f"❌ Error: {str(runner.error)}")
                st.exception(runner.error)
                st.session_state.workflow_running = False
                del st.session_state.workflow_thread

                if st.button("🔄 Reset and Try Again"):
                    st.session_state.workflow_result = None
                    st.session_state.conversation_log = []
                    st.rerun()
            elif runner.result:
                # Store result and update state
                st.session_state.workflow_result = runner.result
                st.session_state.workflow_running = False
                del st.session_state.workflow_thread
                st.success("✅ Workflow completed successfully!")
                time_module.sleep(1)
                st.rerun()

    else:
        # Show results
        result = st.session_state.workflow_result

        if result:
            st.markdown("### ✅ Workflow Complete")

            # Display agent conversations dialog
            if st.session_state.show_agent_conversations:
                st.markdown("---")
                st.markdown("### 💬 Agent Conversations")
                display_agent_conversations_dialog()
                if st.button("❌ Close Conversations View", use_container_width=True):
                    st.session_state.show_agent_conversations = False
                    st.rerun()
            else:
                # Show brief conversation summary
                if st.session_state.conversation_log:
                    total_messages = len(st.session_state.conversation_log)
                    agent_count = len(
                        set(
                            entry.get("agent_name", "System")
                            for entry in st.session_state.conversation_log
                        )
                    )
                    st.info(
                        f"💬 {agent_count} agent(s) generated {total_messages} messages. Click below to view detailed conversations."
                    )
                    if st.button("💬 View All Agent Conversations", use_container_width=True):
                        st.session_state.show_agent_conversations = True
                        st.rerun()

            # Show brief summary
            st.markdown("---")
            st.markdown("### 📋 Summary")

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:
                st.markdown("**Status:**")
                st.success(f"✅ {result.final_status}")

                if hasattr(result, "workspace_path"):
                    st.markdown("**Workspace:**")
                    st.code(str(result.workspace_path))

            with summary_col2:
                if hasattr(result, "novelty_score") and result.novelty_score:
                    st.metric("Novelty Score", f"{result.novelty_score:.1f}/10")

                if hasattr(result, "papers"):
                    st.metric("Papers Found", len(result.papers))

            # Show final summary if available
            if hasattr(result, "final_summary") and result.final_summary:
                with st.expander("📄 View Full Summary", expanded=False):
                    st.markdown(result.final_summary)

            # Action buttons
            st.markdown("---")
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                if st.button("💾 Save Summary", use_container_width=True):
                    if hasattr(result, "save_summary"):
                        saved_path = result.save_summary()
                        st.success(f"Saved to: {saved_path}")

            with col2:
                if st.button("💬 View Conversations", use_container_width=True):
                    st.session_state.show_agent_conversations = True
                    st.rerun()

            with col3:
                if st.button("📋 View Logs", use_container_width=True):
                    st.session_state.show_logs = True
                    st.rerun()

            with col4:
                if st.button("🔄 Start New Research", use_container_width=True):
                    st.session_state.workflow_result = None
                    st.session_state.workflow_running = False
                    st.session_state.conversation_log = []
                    st.session_state.last_update_count = 0
                    st.session_state.show_agent_conversations = False
                    st.session_state.show_logs = False
                    reset_log_handler()
                    st.rerun()

            with col5:
                if st.button("📂 Open Workspace", use_container_width=True):
                    if hasattr(result, "workspace_path"):
                        st.info(f"Workspace: {result.workspace_path}")

            # Display logs dialog if requested
            if st.session_state.show_logs:
                st.markdown("---")
                st.markdown("### 📋 Logger Output")
                display_logs_dialog()
                if st.button("❌ Close Logs View", use_container_width=True):
                    st.session_state.show_logs = False
                    st.rerun()

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
