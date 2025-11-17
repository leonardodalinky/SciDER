from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from scievo.core import constant
from scievo.core.llms import ModelRegistry
from scievo.core.plan import Plan
from scievo.core.types import Message
from scievo.core.utils import parse_json_from_llm_response
from scievo.prompts import PROMPTS
from scievo.tools import ToolRegistry

from .state import ExperimentAgentState

LLM_NAME = "plan"
AGENT_NAME = "experiment_planner"


import json


def should_replan(agent_state: ExperimentAgentState) -> str:
    """
    Decide whether to continue (go to 'gateway')
    or finish (go to END).
    """

    last_msg = agent_state.patched_history[-1]
    content = (last_msg.content or "").strip()

    # Try JSON parse
    try:
        data = json.loads(content)
    except:
        # If not JSON → treat as continue
        return "gateway"

    # Case: continue
    if isinstance(data, dict) and data.get("continued") is True:
        return "gateway"

    # Case: end (no more work)
    if isinstance(data, dict) and data.get("continued") is False:
        return "end"

    # Case: new plan created (interpreted as finishing replanner stage)
    if isinstance(data, dict) and "modified" in data:
        # modified means replanner finished → next step will recreate plan outside
        return "end"

    # fallback
    return "gateway"


@logger.catch
def planner_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    logger.trace("planner_node of Agent {}", AGENT_NAME)

    # Extract the GitHub repo URL from user_query
    repo_url = agent_state.user_query.strip()
    if not repo_url.startswith("http"):
        raise ValueError(f"User query is not a valid GitHub URL: {repo_url}")

    # Clone the repository using github.clone_repo tool
    tools = ToolRegistry.get_toolset("github")
    clone_func = tools["clone_repo"].func

    # clone into: ~/.experiment_repos/<repo_name>
    local_root = Path.home() / ".experiment_repos"
    local_root.mkdir(parents=True, exist_ok=True)

    clone_result = clone_func(repo_url=repo_url, dest_dir=str(local_root))

    logger.debug(f"[ExperimentAgent] clone_repo result: {clone_result}")

    # Determine cloned repo path
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_dir = local_root / repo_name
    agent_state.repo_dir = repo_dir  # save path to agent state

    # Read README using read_readme tool
    readme_func = tools["read_readme"].func
    readme_text = readme_func(repo_dir=str(repo_dir))
    agent_state.readme_text = readme_text

    logger.debug(f"[ExperimentAgent] README loaded ({len(readme_text)} chars)")

    # Construct a planning prompt for the LLM
    user_prompt = PROMPTS.experiment.planner_system_prompt.render(
        repo_url=repo_url,
        repo_dir=str(repo_dir),
        readme_text=readme_text,
        user_instruction=agent_state.user_instructions,
    )

    system_prompt = (
        Message(
            role="system",
            content=user_prompt,
        )
        .with_log(cond=constant.LOG_SYSTEM_PROMPT)
        .content
    )

    # Call planning model to generate plan
    planning_msg = ModelRegistry.completion(
        LLM_NAME,
        [],  # planner uses no chat history, it's stateless
        system_prompt=system_prompt,
        agent_sender=AGENT_NAME,
    ).with_log()

    # Parse JSON output into Plan model
    try:
        plans = parse_json_from_llm_response(planning_msg, Plan)
    except Exception as e:
        raise RuntimeError(f"Failed to parse plan from LLM: {e}")

    # Update experiment agent internal state
    agent_state.plans = plans
    agent_state.remaining_plans = plans.steps
    agent_state.past_plans = []

    # Add a control message: “Follow the current plan.”
    agent_state.add_message(
        Message(
            role="assistant",
            content="Follow the current plan.",
            agent_sender=AGENT_NAME,
        )
    )

    # Logging next step (not added to history)
    Message(
        role="assistant",
        content=PROMPTS.experiment.replanner_user_response.render(
            next_step=agent_state.remaining_plans[0],
        ),
        agent_sender=AGENT_NAME,
    ).with_log()

    return agent_state


def replanner_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    logger.trace("replanner_node of Agent {}", AGENT_NAME)

    system_prompt = PROMPTS.experiment.replanner_system_prompt.render()
    replanner_user_prompt = PROMPTS.experiment.replanner_user_prompt.render(
        user_query=agent_state.user_query,
        plan=agent_state.plans.steps if agent_state.plans else [],
        past_steps=agent_state.past_plans or [],
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": replanner_user_prompt},
    ]

    raw = ModelRegistry.completion(
        LLM_NAME, [], system_prompt=system_prompt, agent_sender=AGENT_NAME, messages=messages
    )

    llm_msg = (
        raw
        if isinstance(raw, Message)
        else Message(
            role="assistant",
            content=str(raw),
            agent_sender=AGENT_NAME,
        ).with_log()
    )

    agent_state.add_message(llm_msg)
    return agent_state
