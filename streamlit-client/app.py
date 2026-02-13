import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import streamlit as st

os.environ["CODING_AGENT_VERSION"] = "v3"
os.environ.setdefault("SCIEVO_ENABLE_OPENHANDS", "0")

sys.path.insert(0, str(Path(__file__).parent.parent))

from scievo.agents import ideation_agent
from scievo.agents.ideation_agent.state import IdeationAgentState
from scievo.core.brain import Brain
from scievo.core.llms import ModelRegistry
from scievo.workflows.data_workflow import DataWorkflow
from scievo.workflows.experiment_workflow import ExperimentWorkflow
from scievo.workflows.full_workflow_with_ideation import FullWorkflowWithIdeation

st.set_page_config(page_title="SciEvo Chat", layout="centered")

st.markdown(
    """
    <style>
    /* App headers: title, subheader, form section titles - exclude chat message content */
    h1, h2, h3, h4, h5, h6 {
        color: #384166 !important;
    }
    /* Exclude LLM-generated content inside chat messages */
    [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] h4,
    [data-testid="stChatMessage"] h5, [data-testid="stChatMessage"] h6 {
        color: inherit !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def register_all_models(user_api_key=None, user_model=None):
    api_key = user_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False

    default_model = user_model or os.getenv("SCIEVO_DEFAULT_MODEL", "gemini/gemini-2.5-flash-lite")
    openai_api_key = (
        user_api_key
        if user_api_key and "openai" in default_model.lower()
        else os.getenv("OPENAI_API_KEY")
    )

    models = [
        ("ideation", default_model, api_key),
        ("data", default_model, api_key),
        ("plan", default_model, api_key),
        ("history", default_model, api_key),
        ("experiment_agent", default_model, api_key),
        ("experiment_coding", default_model, api_key),
        ("experiment_execute", default_model, api_key),
        ("experiment_summary", default_model, api_key),
        ("experiment_monitor", default_model, api_key),
        ("paper_search", default_model, api_key),
        ("metric_search", default_model, api_key),
        ("critic", default_model, api_key),
        ("mem", default_model, api_key),
    ]

    embed_model = os.getenv("EMBED_MODEL", "text-embedding-004")
    embed_api_key = os.getenv("EMBED_API_KEY", openai_api_key or api_key)
    models.append(("embed", embed_model, embed_api_key))

    for name, model, key in models:
        ModelRegistry.register(name=name, model=model, api_key=key)

    return True


def stream_markdown(text, delay=0.02):
    buf = ""
    slot = st.empty()
    for line in text.split("\n"):
        buf += line + "\n"
        slot.markdown(buf)
        time.sleep(delay)


def render_intermediate_state(intermediate_state):
    if not intermediate_state:
        return
    by_node = defaultdict(list)
    for item in intermediate_state:
        by_node[item.get("node_name", "unknown")].append(item.get("output", ""))

    st.divider()
    st.subheader("Intermediate States")
    for node, outputs in by_node.items():
        with st.expander(node, expanded=False):
            for i, content in enumerate(outputs, 1):
                st.markdown(f"**Step {i}**")
                st.markdown(content)


def run_ideation(q):
    s = IdeationAgentState(user_query=q)
    r = st.session_state.ideation_graph.invoke(s, {"recursion_limit": 50})
    rs = IdeationAgentState(**r)
    out = []
    if rs.output_summary:
        out.append("## Research Ideas Summary\n\n" + rs.output_summary)
    if rs.novelty_score is not None:
        out.append(
            "## Novelty Evaluation\n```json\n"
            + json.dumps(
                {
                    "novelty_score": rs.novelty_score,
                    "feedback": rs.novelty_feedback,
                },
                indent=2,
            )
            + "\n```"
        )
    if rs.research_ideas:
        out.append("## Generated Research Ideas\n")
        for i, idea in enumerate(rs.research_ideas[:5], 1):
            out.append(f"### {i}. {idea.get('title','')}\n{idea.get('description','')}")
    return ("\n\n".join(out) if out else "No result", rs.intermediate_state)


def run_data(path, q):
    w = DataWorkflow(
        data_path=Path(path),
        workspace_path=st.session_state.workspace_path,
        recursion_limit=100,
    )
    w.run()
    intermediate_state = getattr(w, "data_agent_intermediate_state", [])
    if w.final_status != "success":
        return "Data workflow failed", intermediate_state
    out = ["## Data Analysis Complete"]
    if w.data_summary:
        out.append(w.data_summary)
    return "\n\n".join(out), intermediate_state


def run_experiment(q, path):
    if path:
        w = ExperimentWorkflow.from_data_analysis_file(
            workspace_path=st.session_state.workspace_path,
            user_query=q,
            data_analysis_path=path,
            max_revisions=5,
            recursion_limit=100,
        )
    else:
        return "No data analysis file", []
    w.run()
    return w.final_summary or "Experiment finished", w.experiment_agent_intermediate_state


def run_full(cfg):
    w = FullWorkflowWithIdeation(
        user_query=cfg["query"],
        workspace_path=st.session_state.workspace_path,
        data_path=Path(cfg["data_path"]) if cfg["data_path"] else None,
        run_data_workflow=cfg["run_data"],
        run_experiment_workflow=cfg["run_exp"],
        max_revisions=5,
    )
    w.run()
    return w.final_summary or "Workflow finished", []


def get_upload_temp_dir() -> Path:
    """Return temp directory for uploaded files. Clean old dirs on startup."""
    base = Path(tempfile.gettempdir()) / "scievo_uploads"
    base.mkdir(parents=True, exist_ok=True)
    # Clean dirs older than 1 hour (handles closed sessions)
    now = time.time()
    for d in base.iterdir():
        if d.is_dir() and (now - d.stat().st_mtime) > 3600:
            try:
                shutil.rmtree(d)
            except OSError:
                pass
    return base


def save_and_extract_upload(uploaded_file) -> Path | None:
    """Save uploaded zip to temp dir, extract it, return path to extracted dir."""
    if uploaded_file is None or not uploaded_file.name.lower().endswith(".zip"):
        return None
    base = get_upload_temp_dir()
    dest_dir = Path(tempfile.mkdtemp(dir=base))
    zip_path = dest_dir / uploaded_file.name
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    extract_dir = dest_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()
    return extract_dir


def find_data_analysis_file(extract_dir: Path) -> Path | None:
    """Find data_analysis.md in extracted dir (root or first subdir)."""
    candidates = [extract_dir / "data_analysis.md", extract_dir / "analysis.md"]
    for c in candidates:
        if c.exists():
            return c
    for p in extract_dir.rglob("data_analysis.md"):
        return p
    for p in extract_dir.rglob("analysis.md"):
        return p
    return None


def _rm_upload_root(p: Path):
    """Remove the scievo_uploads session dir (go up to find it)."""
    cur = Path(p).resolve().parent if Path(p).resolve().is_file() else Path(p).resolve()
    while cur != cur.parent:
        parent = cur.parent
        if parent.name == "scievo_uploads":
            try:
                shutil.rmtree(cur)
            except OSError:
                pass
            return
        cur = parent


def cleanup_uploaded_data():
    """Remove temp uploaded data and restore workspace_path to default."""
    for key in ("uploaded_data_path", "uploaded_experiment_path", "uploaded_full_data_path"):
        path = st.session_state.get(key)
        if path and isinstance(path, (str, Path)):
            _rm_upload_root(Path(path))
            if key in st.session_state:
                del st.session_state[key]
    # Restore agent workspace to default
    if "default_workspace_path" in st.session_state:
        st.session_state.workspace_path = st.session_state.default_workspace_path


def get_next_memo_number(memory_dir: Path) -> int:
    if not memory_dir.exists():
        return 1

    existing_memos = [
        d.name for d in memory_dir.iterdir() if d.is_dir() and d.name.startswith("memo_")
    ]

    if not existing_memos:
        return 1

    numbers = []
    for memo in existing_memos:
        try:
            num = int(memo.replace("memo_", ""))
            numbers.append(num)
        except ValueError:
            continue

    return max(numbers) + 1 if numbers else 1


def save_chat_history(messages: list, workflow_type: str, metadata: dict = None):
    base_dir = Path.cwd() / "case-study-memory"
    base_dir.mkdir(parents=True, exist_ok=True)

    memo_number = get_next_memo_number(base_dir)
    memo_dir = base_dir / f"memo_{memo_number}"
    memo_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    chat_data = {
        "timestamp": timestamp,
        "workflow_type": workflow_type,
        "metadata": metadata or {},
        "messages": messages,
    }

    chat_file = memo_dir / "chat_history.json"
    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)

    return memo_dir


if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
if "default_model" not in st.session_state:
    st.session_state.default_model = os.getenv(
        "SCIEVO_DEFAULT_MODEL", "gemini/gemini-2.5-flash-lite"
    )

if not st.session_state.api_key:
    st.title("SciEvo Research Assistant")
    st.warning("API Key Required")
    st.markdown("Please provide an API key to use the SciEvo Research Assistant.")

    col1, col2 = st.columns(2)
    with col1:
        model_option = st.selectbox(
            "Select Model Provider",
            ["Gemini", "OpenAI"],
            index=0 if "gemini" in st.session_state.default_model.lower() else 1,
        )

    with col2:
        api_key_input = st.text_input(
            "API Key", type="password", placeholder="Enter your API key here", value=""
        )

    if st.button("Save API Key", type="primary"):
        if api_key_input:
            st.session_state.api_key = api_key_input
            if model_option == "Gemini":
                st.session_state.default_model = "gemini/gemini-2.5-flash-lite"
            else:
                st.session_state.default_model = "gpt-4o-mini"
            st.rerun()
        else:
            st.error("Please enter a valid API key")
    st.stop()

col_title, col_reset = st.columns([5, 1])
with col_title:
    st.title("SciEvo Research Assistant")
with col_reset:
    if st.button("🔄 Reset", help="Clear all chat history", type="secondary"):
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

if "initialized" not in st.session_state:
    if not os.getenv("BRAIN_DIR"):
        os.environ["BRAIN_DIR"] = str(Path.cwd() / "tmp_brain")
    Brain()
    if register_all_models(st.session_state.api_key, st.session_state.default_model):
        st.session_state.ideation_graph = ideation_agent.build().compile()
        st.session_state.initialized = True
    else:
        st.error("Failed to register models. Please check your API key.")
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
# If workspace_path points to expired temp upload dir, restore to default
_ws = st.session_state.workspace_path
if isinstance(_ws, (str, Path)) and "scievo_uploads" in str(_ws) and not Path(_ws).exists():
    cleanup_uploaded_data()

if "selected_workflow" not in st.session_state:
    st.session_state.selected_workflow = None

# Workflow selection UI - buttons (placed at top for visibility)
st.subheader("Select Workflow Type")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💡 Ideation", use_container_width=True, key="btn_ideation"):
        st.session_state.selected_workflow = "ideation"
        st.rerun()

with col2:
    if st.button("📊 Data Analysis", use_container_width=True, key="btn_data"):
        st.session_state.selected_workflow = "data"
        st.rerun()

with col3:
    if st.button("🧪 Experiment", use_container_width=True, key="btn_experiment"):
        st.session_state.selected_workflow = "experiment"
        st.rerun()

with col4:
    if st.button("🚀 Full Workflow", use_container_width=True, key="btn_full"):
        st.session_state.selected_workflow = "full"
        st.rerun()

st.divider()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Workflow input forms
workflow_config = None

if st.session_state.selected_workflow == "ideation":
    with st.form("ideation_form", clear_on_submit=True):
        st.markdown("### 💡 Ideation Workflow")
        topic = st.text_input("Research Topic", placeholder="Enter your research topic here...")
        submitted = st.form_submit_button("Run Ideation", type="primary")
        if submitted and topic:
            workflow_config = {"type": "ideation", "query": topic}
            st.session_state.selected_workflow = None

elif st.session_state.selected_workflow == "data":
    with st.form("data_form", clear_on_submit=True):
        st.markdown("### 📊 Data Analysis Workflow")
        st.caption("Upload a zip dataset or enter a path to existing data")
        uploaded_zip = st.file_uploader(
            "Upload ZIP dataset (optional)",
            type=["zip"],
            help="Upload a zip file containing your dataset. Extracted temporarily, deleted on reset.",
        )
        if st.session_state.get("uploaded_data_path"):
            st.info(f"📁 Using uploaded data: `{st.session_state.uploaded_data_path}`")
        data_path_manual = st.text_input(
            "Or enter data path manually",
            placeholder="e.g. /path/to/data.csv or /path/to/data_dir",
        )
        query = st.text_input("Query", placeholder="What would you like to analyze?")
        submitted = st.form_submit_button("Run Data Analysis", type="primary")
        if submitted and query:
            path_to_use = None
            if uploaded_zip:
                cleanup_uploaded_data()  # Remove previous upload before saving new one
                extracted = save_and_extract_upload(uploaded_zip)
                if extracted:
                    st.session_state.uploaded_data_path = str(extracted)
                    st.session_state.workspace_path = extracted.parent
                    path_to_use = str(extracted)
                else:
                    st.error("Failed to process uploaded zip file.")
            elif data_path_manual.strip():
                path_to_use = data_path_manual.strip()
            elif st.session_state.get("uploaded_data_path"):
                path = Path(st.session_state.uploaded_data_path)
                if path.exists():
                    path_to_use = st.session_state.uploaded_data_path
                    st.session_state.workspace_path = path.parent
                else:
                    cleanup_uploaded_data()
            if path_to_use:
                workflow_config = {"type": "data", "path": path_to_use, "query": query}
                st.session_state.selected_workflow = None
            else:
                st.error("Please upload a zip file or enter a data path.")

elif st.session_state.selected_workflow == "experiment":
    with st.form("experiment_form", clear_on_submit=True):
        st.markdown("### 🧪 Experiment Workflow")
        st.caption("Upload a zip containing data_analysis.md or enter path manually")
        uploaded_exp_zip = st.file_uploader(
            "Upload ZIP with data analysis (optional)",
            type=["zip"],
            key="exp_upload",
            help="Zip containing data_analysis.md. Extracted temporarily, deleted on reset.",
        )
        if st.session_state.get("uploaded_experiment_path"):
            st.info(f"📁 Using: `{st.session_state.uploaded_experiment_path}`")
        data_path_manual = st.text_input(
            "Or enter data analysis path manually",
            placeholder="Path to data_analysis.md (optional)",
        )
        query = st.text_input("Experiment Query", placeholder="Describe your experiment...")
        submitted = st.form_submit_button("Run Experiment", type="primary")
        if submitted and query:
            path_to_use = None
            if uploaded_exp_zip:
                prev = st.session_state.get("uploaded_experiment_path")
                if prev:
                    _rm_upload_root(Path(prev))
                    if "uploaded_experiment_path" in st.session_state:
                        del st.session_state.uploaded_experiment_path
                extracted = save_and_extract_upload(uploaded_exp_zip)
                if extracted:
                    analysis_file = find_data_analysis_file(extracted)
                    if analysis_file:
                        st.session_state.uploaded_experiment_path = str(analysis_file)
                        st.session_state.workspace_path = analysis_file.parent
                        path_to_use = str(analysis_file)
                    else:
                        st.error("Zip must contain data_analysis.md or analysis.md")
            elif data_path_manual.strip():
                path_to_use = data_path_manual.strip()
            elif st.session_state.get("uploaded_experiment_path"):
                p = Path(st.session_state.uploaded_experiment_path)
                if p.exists():
                    path_to_use = st.session_state.uploaded_experiment_path
                    st.session_state.workspace_path = p.parent
                else:
                    if "uploaded_experiment_path" in st.session_state:
                        del st.session_state.uploaded_experiment_path
            if path_to_use:
                workflow_config = {"type": "experiment", "query": query, "path": path_to_use}
                st.session_state.selected_workflow = None
            else:
                st.error("Please upload a zip with data_analysis.md or enter a data analysis path.")

elif st.session_state.selected_workflow == "full":
    with st.form("full_form", clear_on_submit=True):
        st.markdown("### 🚀 Full Workflow")
        topic = st.text_input("Research Topic", placeholder="Enter your research topic...")
        st.caption("Data (for Data Analysis): upload zip or enter path")
        uploaded_full_zip = st.file_uploader(
            "Upload ZIP dataset (optional)",
            type=["zip"],
            key="full_upload",
            help="Zip dataset for Data Analysis. Extracted temporarily, deleted on reset.",
        )
        if st.session_state.get("uploaded_full_data_path"):
            st.info(f"📁 Using: `{st.session_state.uploaded_full_data_path}`")
        data_path_manual = st.text_input(
            "Or enter data path manually",
            placeholder="Path to data file/dir (optional)",
        )
        run_data = st.checkbox("Run Data Analysis", value=False)
        run_exp = st.checkbox("Run Experiment", value=False)
        submitted = st.form_submit_button("Run Full Workflow", type="primary")
        if submitted and topic:
            data_path_to_use = None
            if run_data:
                if uploaded_full_zip:
                    prev = st.session_state.get("uploaded_full_data_path")
                    if prev:
                        _rm_upload_root(Path(prev))
                        if "uploaded_full_data_path" in st.session_state:
                            del st.session_state.uploaded_full_data_path
                    extracted = save_and_extract_upload(uploaded_full_zip)
                    if extracted:
                        st.session_state.uploaded_full_data_path = str(extracted)
                        st.session_state.workspace_path = extracted.parent
                        data_path_to_use = str(extracted)
                elif data_path_manual.strip():
                    data_path_to_use = data_path_manual.strip()
                elif st.session_state.get("uploaded_full_data_path"):
                    p = Path(st.session_state.uploaded_full_data_path)
                    if p.exists():
                        data_path_to_use = st.session_state.uploaded_full_data_path
                        st.session_state.workspace_path = p.parent
                    else:
                        if "uploaded_full_data_path" in st.session_state:
                            del st.session_state.uploaded_full_data_path
                if not data_path_to_use:
                    st.error("Run Data Analysis requires uploading a zip or entering a data path.")
                    data_path_to_use = None
            if data_path_to_use is not None or not run_data:
                workflow_config = {
                    "type": "full",
                    "query": topic,
                    "data_path": data_path_to_use,
                    "run_data": run_data,
                    "run_exp": run_exp,
                }
                st.session_state.selected_workflow = None

if workflow_config:
    # Add user message to chat
    if workflow_config["type"] == "ideation":
        user_msg = f"Ideation: {workflow_config['query']}"
    elif workflow_config["type"] == "data":
        user_msg = f"Data Analysis: {workflow_config['path']} - {workflow_config['query']}"
    elif workflow_config["type"] == "experiment":
        user_msg = f"Experiment: {workflow_config['query']}"
        if workflow_config.get("path"):
            user_msg += f" (Data: {workflow_config['path']})"
    else:  # full
        user_msg = f"Full Workflow: {workflow_config['query']}"
        if workflow_config.get("data_path"):
            user_msg += f" (Data: {workflow_config['data_path']})"
        if workflow_config.get("run_data"):
            user_msg += " [Data Analysis]"
        if workflow_config.get("run_exp"):
            user_msg += " [Experiment]"

    st.session_state.messages.append({"role": "user", "content": user_msg})

    # Execute workflow
    with st.chat_message("assistant"):
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("Processing your request...")
            with st.spinner(""):
                if workflow_config["type"] == "ideation":
                    resp, intermediate_state = run_ideation(workflow_config.get("query"))
                elif workflow_config["type"] == "data":
                    resp, intermediate_state = run_data(
                        workflow_config["path"], workflow_config["query"]
                    )
                elif workflow_config["type"] == "experiment":
                    resp, intermediate_state = run_experiment(
                        workflow_config["query"], workflow_config.get("path")
                    )
                elif workflow_config["type"] == "full":
                    resp, intermediate_state = run_full(workflow_config)
                else:
                    resp, intermediate_state = "Unknown workflow type", []

        loading_placeholder.empty()
        stream_markdown(resp)
        render_intermediate_state(intermediate_state)
        st.session_state.messages.append({"role": "assistant", "content": resp})

        metadata = {
            "workflow_type": workflow_config["type"],
            "query": workflow_config.get("query"),
            "path": workflow_config.get("path"),
        }
        if workflow_config["type"] == "full":
            metadata.update(
                {
                    "data_path": workflow_config.get("data_path"),
                    "run_data": workflow_config.get("run_data"),
                    "run_exp": workflow_config.get("run_exp"),
                }
            )

        memo_dir = save_chat_history(
            st.session_state.messages, workflow_type=workflow_config["type"], metadata=metadata
        )
        st.session_state.last_saved_memo = str(memo_dir)

    st.rerun()


def parse_command(prompt):
    prompt = prompt.strip()
    if prompt.startswith("/ideation"):
        p = prompt.split(maxsplit=1)
        return {"type": "ideation", "query": p[1] if len(p) > 1 else None}
    if prompt.startswith("/data"):
        p = prompt.split(maxsplit=2)
        if len(p) < 3:
            return {"type": "error", "msg": "Usage: /data <path> <query>"}
        return {"type": "data", "path": p[1], "query": p[2]}
    if prompt.startswith("/experiment"):
        p = prompt.split(maxsplit=1)
        if len(p) < 2:
            return {"type": "error", "msg": "Usage: /experiment <query> [data_path]"}
        r = p[1].split(maxsplit=1)
        return {"type": "experiment", "query": r[0], "path": r[1] if len(r) > 1 else None}
    if prompt.startswith("/full"):
        p = prompt.split()
        cfg = {
            "type": "full",
            "query": p[1] if len(p) > 1 else None,
            "data_path": None,
            "run_data": False,
            "run_exp": False,
        }
        i = 2
        while i < len(p):
            if p[i] == "--data" and i + 1 < len(p):
                cfg["data_path"] = p[i + 1]
                cfg["run_data"] = True
                i += 2
            elif p[i] == "--experiment":
                cfg["run_exp"] = True
                i += 1
            else:
                i += 1
        return cfg
    return {"type": "ideation", "query": prompt}


# Chat input for general questions (use workflow buttons above for structured workflows)
if prompt := st.chat_input("Ask a question or select a workflow above"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("Processing your request...")
            with st.spinner(""):
                resp, intermediate_state = run_ideation(prompt)

        loading_placeholder.empty()
        stream_markdown(resp)
        render_intermediate_state(intermediate_state)
        st.session_state.messages.append({"role": "assistant", "content": resp})

        memo_dir = save_chat_history(
            st.session_state.messages, workflow_type="ideation", metadata={"query": prompt}
        )
        st.session_state.last_saved_memo = str(memo_dir)

    st.rerun()
