import json
import os
import sys
import time
from collections import defaultdict
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
st.title("SciEvo Research Assistant")


def register_all_models():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No API key found")
        st.stop()

    default_model = os.getenv("SCIEVO_DEFAULT_MODEL", "gemini/gemini-2.5-flash-lite")
    openai_api_key = os.getenv("OPENAI_API_KEY")

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


if "initialized" not in st.session_state:
    if not os.getenv("BRAIN_DIR"):
        os.environ["BRAIN_DIR"] = str(Path.cwd() / "tmp_brain")
    Brain()
    register_all_models()
    st.session_state.ideation_graph = ideation_agent.build().compile()
    st.session_state.initialized = True

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
        for i, idea in enumerate(rs.research_ideas[:5], 1):
            out.append(f"### {i}. {idea.get('title','')}\n{idea.get('description','')}")
    return ("\n\n".join(out) if out else "No result", rs.intermediate_state)


def run_data(path, q):
    w = DataWorkflow(
        data_path=Path(path),
        workspace_path=st.session_state.workspace_path,
        user_query=q,
        recursion_limit=100,
    )
    w.run()
    if w.final_status != "success":
        return "Data workflow failed", []
    out = ["## Data Analysis Complete"]
    if w.data_summary:
        out.append(w.data_summary)
    return "\n\n".join(out), []


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
    return w.final_summary or "Experiment finished", []


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

            stream_markdown(resp)
            render_intermediate_state(intermediate_state)
            st.session_state.messages.append({"role": "assistant", "content": resp})
