import json
import os
import sys
import time
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

st.title("SciEvo Research Assistant")

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
            "content": "Hello. I can run ideation, data analysis, experiments, or a full workflow.\n\nCommands:\n/ideation <topic>\n/data <path> <query>\n/experiment <query> [data_path]\n/full <topic> [--data <path>] [--experiment]",
        }
    ]

if "workspace_path" not in st.session_state:
    st.session_state.workspace_path = Path.cwd() / "workspace"

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


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
        for i, idea in enumerate(rs.research_ideas[:5], 0):
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


if prompt := st.chat_input("Ask or run command"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    cfg = parse_command(prompt)
    with st.chat_message("assistant"):
        if cfg.get("type") == "error":
            stream_markdown(cfg["msg"])
            st.session_state.messages.append({"role": "assistant", "content": cfg["msg"]})
        else:
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                st.markdown("Processing your request...")
                with st.spinner(""):
                    if cfg["type"] == "ideation":
                        resp, intermediate_state = run_ideation(cfg.get("query"))
                    elif cfg["type"] == "data":
                        resp, intermediate_state = run_data(cfg["path"], cfg["query"])
                    elif cfg["type"] == "experiment":
                        resp, intermediate_state = run_experiment(cfg["query"], cfg.get("path"))
                    elif cfg["type"] == "full":
                        resp, intermediate_state = run_full(cfg)
                    else:
                        resp, intermediate_state = "Unknown command", []

            loading_placeholder.empty()
            stream_markdown(resp)
            render_intermediate_state(intermediate_state)
            st.session_state.messages.append({"role": "assistant", "content": resp})

            metadata = {
                "workflow_type": cfg["type"],
                "query": cfg.get("query"),
                "path": cfg.get("path"),
            }
            if cfg["type"] == "full":
                metadata.update(
                    {
                        "data_path": cfg.get("data_path"),
                        "run_data": cfg.get("run_data"),
                        "run_exp": cfg.get("run_exp"),
                    }
                )

            memo_dir = save_chat_history(
                st.session_state.messages, workflow_type=cfg["type"], metadata=metadata
            )
            st.session_state.last_saved_memo = str(memo_dir)
