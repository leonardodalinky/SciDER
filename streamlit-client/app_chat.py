import os
import sys
import time
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

st.title("🔬 SciEvo Research Assistant")


def register_all_models():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "No API key found. Please set GEMINI_API_KEY or OPENAI_API_KEY environment variable."
        )
        st.stop()

    default_model = os.getenv("SCIEVO_DEFAULT_MODEL", "gemini/gemini-2.5-flash-lite")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    models_to_register = [
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
    models_to_register.append(("embed", embed_model, embed_api_key))

    for model_name, model, key in models_to_register:
        try:
            ModelRegistry.register(name=model_name, model=model, api_key=key)
        except Exception as e:
            st.warning(f"Failed to register {model_name}: {e}")


if "initialized" not in st.session_state:
    try:
        if not os.getenv("BRAIN_DIR"):
            os.environ["BRAIN_DIR"] = str(Path.cwd() / "tmp_brain")

        Brain()
        register_all_models()

        st.session_state.ideation_graph = ideation_agent.build().compile()
        st.session_state.initialized = True
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        st.exception(e)
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Hello! I'm SciEvo Research Assistant. I can help you with:\n\n- 💡 **Ideation**: Generate research ideas\n- 📊 **Data Analysis**: Analyze datasets\n- 🧪 **Experiment**: Run ML experiments\n- 🔬 **Full Workflow**: Complete research pipeline\n\nYou can use commands like:\n- `/ideation <topic>` - Generate research ideas\n- `/data <data_path> <query>` - Analyze data\n- `/experiment <query> [data_analysis_path]` - Run experiment\n- `/full <topic> [options]` - Run full workflow\n\nOr just ask naturally!",
        }
    )

if "workspace_path" not in st.session_state:
    st.session_state.workspace_path = Path.cwd() / "workspace"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def parse_command(prompt: str):
    prompt = prompt.strip()

    if prompt.startswith("/ideation"):
        parts = prompt.split(maxsplit=1)
        return {
            "type": "ideation",
            "user_query": parts[1] if len(parts) > 1 else None,
            "research_domain": None,
        }

    elif prompt.startswith("/data"):
        parts = prompt.split(maxsplit=2)
        if len(parts) < 3:
            return {"type": "error", "message": "Usage: /data <data_path> <query>"}
        return {"type": "data", "data_path": parts[1], "user_query": parts[2]}

    elif prompt.startswith("/experiment"):
        parts = prompt.split(maxsplit=1)
        if len(parts) < 2:
            return {"type": "error", "message": "Usage: /experiment <query> [data_analysis_path]"}
        rest = parts[1].split(maxsplit=1)
        return {
            "type": "experiment",
            "user_query": rest[0],
            "data_analysis_path": rest[1] if len(rest) > 1 else None,
        }

    elif prompt.startswith("/full"):
        parts = prompt.split(maxsplit=1)
        if len(parts) < 2:
            return {
                "type": "error",
                "message": "Usage: /full <topic> [--data <path>] [--experiment] [--domain <domain>]",
            }

        args = parts[1].split()
        config = {
            "type": "full",
            "user_query": args[0],
            "research_domain": None,
            "data_path": None,
            "run_data_workflow": False,
            "run_experiment_workflow": False,
        }

        i = 1
        while i < len(args):
            if args[i] == "--data" and i + 1 < len(args):
                config["data_path"] = args[i + 1]
                config["run_data_workflow"] = True
                i += 2
            elif args[i] == "--experiment":
                config["run_experiment_workflow"] = True
                i += 1
            elif args[i] == "--domain" and i + 1 < len(args):
                config["research_domain"] = args[i + 1]
                i += 2
            else:
                i += 1

        return config

    else:
        return {"type": "ideation", "user_query": prompt, "research_domain": None}


def run_ideation(user_query: str, research_domain: str | None = None):
    ideation_state = IdeationAgentState(
        user_query=user_query,
        research_domain=research_domain,
    )

    result = st.session_state.ideation_graph.invoke(
        ideation_state,
        {"recursion_limit": 50},
    )

    result_state = IdeationAgentState(**result)

    response_parts = []

    if result_state.output_summary:
        response_parts.append(f"## Research Ideas Summary\n\n{result_state.output_summary}")

    if result_state.papers:
        response_parts.append(f"\n## Papers Found ({len(result_state.papers)})\n")
        for i, paper in enumerate(result_state.papers[:10], 1):
            title = paper.get("title", "Unknown")
            response_parts.append(f"{i}. {title}")

    if result_state.novelty_score is not None:
        response_parts.append(f"\n## Novelty Score: {result_state.novelty_score:.2f}/10")

    if result_state.novelty_feedback:
        response_parts.append(f"\n## Novelty Feedback\n\n{result_state.novelty_feedback}")

    if result_state.research_ideas:
        response_parts.append(
            f"\n## Generated Research Ideas ({len(result_state.research_ideas)})\n"
        )
        for i, idea in enumerate(result_state.research_ideas[:5], 1):
            idea_title = idea.get("title", f"Idea {i}")
            idea_desc = idea.get("description", "")
            response_parts.append(f"### {i}. {idea_title}\n{idea_desc}\n")

    return (
        "\n".join(response_parts)
        if response_parts
        else "No ideas generated. Please try a different query."
    )


def run_data_workflow(data_path: str, user_query: str):
    workflow = DataWorkflow(
        data_path=Path(data_path),
        workspace_path=st.session_state.workspace_path,
        user_query=user_query,
        recursion_limit=100,
    )

    workflow.run()

    response_parts = []

    if workflow.final_status == "success":
        response_parts.append("## Data Analysis Complete\n")

        if workflow.data_summary:
            response_parts.append(f"### Summary\n\n{workflow.data_summary}\n")

        if workflow.papers:
            response_parts.append(f"\n### Papers Found ({len(workflow.papers)})\n")
            for i, paper in enumerate(workflow.papers[:10], 1):
                title = paper.get("title", "Unknown")
                response_parts.append(f"{i}. {title}")

        if workflow.datasets:
            response_parts.append(f"\n### Datasets Found ({len(workflow.datasets)})\n")
            for i, dataset in enumerate(workflow.datasets[:5], 1):
                name = dataset.get("name", "Unknown")
                response_parts.append(f"{i}. {name}")

        if workflow.metrics:
            response_parts.append(f"\n### Metrics Found ({len(workflow.metrics)})\n")
            for i, metric in enumerate(workflow.metrics[:5], 1):
                name = metric.get("name", "Unknown")
                response_parts.append(f"{i}. {name}")

        response_parts.append(f"\n\nWorkspace: {workflow.workspace_path}")
    else:
        response_parts.append(
            f"## Data Analysis Failed\n\n{workflow.error_message or 'Unknown error'}"
        )

    return "\n".join(response_parts)


def run_experiment_workflow(user_query: str, data_analysis_path: str | None = None):
    if data_analysis_path:
        workflow = ExperimentWorkflow.from_data_analysis_file(
            workspace_path=st.session_state.workspace_path,
            user_query=user_query,
            data_analysis_path=data_analysis_path,
            max_revisions=5,
            recursion_limit=100,
        )
    else:
        default_data_path = st.session_state.workspace_path / "data_analysis.md"
        if default_data_path.exists():
            workflow = ExperimentWorkflow.from_data_analysis_file(
                workspace_path=st.session_state.workspace_path,
                user_query=user_query,
                data_analysis_path=default_data_path,
                max_revisions=5,
                recursion_limit=100,
            )
        else:
            return "Error: No data analysis file found. Please run data workflow first or specify path."

    workflow.run()

    response_parts = []

    if workflow.final_status == "success":
        response_parts.append("## Experiment Complete\n")

        if workflow.final_summary:
            response_parts.append(f"### Summary\n\n{workflow.final_summary}\n")

        if workflow.execution_results:
            response_parts.append(f"\n### Execution Results ({len(workflow.execution_results)})\n")
            for i, result in enumerate(workflow.execution_results[:5], 1):
                response_parts.append(f"{i}. {str(result)[:200]}...")

        response_parts.append(f"\n\nWorkspace: {workflow.workspace_path}")
    else:
        response_parts.append(
            f"## Experiment Failed\n\n{workflow.error_message or 'Unknown error'}"
        )

    return "\n".join(response_parts)


def run_full_workflow(config: dict):
    workflow = FullWorkflowWithIdeation(
        user_query=config["user_query"],
        workspace_path=st.session_state.workspace_path,
        research_domain=config.get("research_domain"),
        data_path=Path(config["data_path"]) if config.get("data_path") else None,
        run_data_workflow=config.get("run_data_workflow", False),
        run_experiment_workflow=config.get("run_experiment_workflow", False),
        repo_source=None,
        max_revisions=5,
        ideation_agent_recursion_limit=50,
        data_agent_recursion_limit=100,
        experiment_agent_recursion_limit=100,
    )

    workflow.run()

    response_parts = []

    if workflow.final_status == "success":
        response_parts.append("## Full Workflow Complete\n")

        if workflow.ideation_summary:
            response_parts.append(f"### Ideation Results\n\n{workflow.ideation_summary}\n")

        if workflow.data_summary:
            response_parts.append(f"### Data Analysis Results\n\n{workflow.data_summary}\n")

        if workflow.final_summary:
            response_parts.append(f"### Experiment Results\n\n{workflow.final_summary}\n")

        if workflow.novelty_score is not None:
            response_parts.append(f"\n### Novelty Score: {workflow.novelty_score:.2f}/10")

        response_parts.append(f"\n\nWorkspace: {workflow.workspace_path}")
    else:
        response_parts.append(f"## Workflow Failed\n\n{workflow.error_message or 'Unknown error'}")

    return "\n".join(response_parts)


if prompt := st.chat_input("Ask me anything or use commands..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        config = parse_command(prompt)

        if config.get("type") == "error":
            error_msg = config["message"]
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            try:
                with st.spinner("Processing..."):
                    if config["type"] == "ideation":
                        if not config.get("user_query"):
                            response = "Please provide a research topic. Usage: /ideation <topic>"
                        else:
                            response = run_ideation(
                                config["user_query"], config.get("research_domain")
                            )

                    elif config["type"] == "data":
                        response = run_data_workflow(config["data_path"], config["user_query"])

                    elif config["type"] == "experiment":
                        response = run_experiment_workflow(
                            config["user_query"], config.get("data_analysis_path")
                        )

                    elif config["type"] == "full":
                        response = run_full_workflow(config)

                    else:
                        response = "Unknown command type."

                def response_stream():
                    words = response.split()
                    for i, word in enumerate(words):
                        yield word + (" " if i < len(words) - 1 else "")
                        if i % 5 == 0:
                            time.sleep(0.02)

                full_response = st.write_stream(response_stream())
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.exception(e)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
