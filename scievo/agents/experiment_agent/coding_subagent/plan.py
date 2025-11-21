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
    or finish and generate report (go to 'report').

    Decision logic:
    1. If replanner says {"continued": true} → continue to gateway
    2. If replanner says {"continued": false} → finish and generate report
    3. If replanner says {"modified": [...]} → finish and generate report (plan modified, will be handled separately)
    4. If no remaining plans AND no past plans → finish (nothing to do)
    5. Otherwise → continue (fallback)
    """

    last_msg = agent_state.patched_history[-1]
    content = (last_msg.content or "").strip()

    # Check if there are any plans left
    has_remaining_plans = agent_state.remaining_plans and len(agent_state.remaining_plans) > 0
    has_past_plans = agent_state.past_plans and len(agent_state.past_plans) > 0

    # If no plans at all, finish
    if not has_remaining_plans and not has_past_plans:
        logger.debug("No plans remaining, finishing experiment")
        return "report"

    # Try to extract JSON from response (handles cases where LLM adds extra text)
    try:
        # First try direct JSON parse
        data = json.loads(content)
    except:
        # If that fails, try to extract JSON from markdown code blocks or text
        try:
            import re

            # Try to find JSON in markdown code blocks
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                # Try to find JSON object in the text
                json_match = re.search(
                    r'\{[^{}]*"continued"[^{}]*\}|"modified"[^{}]*\[[^\]]*\]', content
                )
                if json_match:
                    data = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON found in response")
        except Exception as e:
            logger.warning(
                f"Could not parse replanner JSON response: {e}. Content: {content[:200]}"
            )
            # If we can't parse and have remaining plans, continue; otherwise finish
            if has_remaining_plans:
                logger.debug("Could not parse JSON but has remaining plans, continuing")
                return "gateway"
            else:
                logger.debug("Could not parse JSON and no remaining plans, finishing")
                return "report"

    # Case: continue
    if isinstance(data, dict) and data.get("continued") is True:
        logger.debug("Replanner says continue, going to gateway")
        return "gateway"

    # Case: end (no more work)
    if isinstance(data, dict) and data.get("continued") is False:
        logger.debug("Replanner says finished, generating report")
        return "report"

    # Case: new plan created (interpreted as finishing replanner stage)
    if isinstance(data, dict) and "modified" in data:
        # modified means replanner finished → generate report
        logger.debug("Replanner modified plan, generating report")
        return "report"

    # fallback: if we have remaining plans, continue; otherwise finish
    if has_remaining_plans:
        logger.debug("Fallback: has remaining plans, continuing")
        return "gateway"
    else:
        logger.debug("Fallback: no remaining plans, finishing")
        return "report"


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

    # Add a user message to trigger execution of the first step
    agent_state.add_message(
        Message(
            role="user",
            content=PROMPTS.experiment.replanner_user_response.render(
                next_step=agent_state.remaining_plans[0],
            ),
        )
    )

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
        )
    )

    # Try to extract and validate JSON from response
    # If LLM didn't return pure JSON, try to extract it
    content = llm_msg.content or ""
    try:
        # First try direct parse
        json.loads(content)
        # If successful, it's valid JSON
    except:
        # Try to extract JSON from markdown or text
        try:
            import re

            from scievo.core.utils import repair_json

            # Try to find JSON in markdown code blocks
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Try to find JSON object in the text
                json_match = re.search(
                    r'\{[^{}]*(?:"continued"|"modified")[^{}]*\}', content, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Last resort: try to find any JSON-like structure
                    json_match = re.search(
                        r'\{.*"continued".*\}|\{.*"modified".*\}', content, re.DOTALL
                    )
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        raise ValueError("No JSON found in response")

            # Repair and validate JSON
            json_str = repair_json(json_str)
            # Validate it's proper JSON
            json.loads(json_str)
            # Replace content with clean JSON
            llm_msg.content = json_str
            logger.debug(f"Extracted JSON from LLM response: {json_str}")
        except Exception as e:
            logger.warning(f"Could not extract JSON from replanner response: {e}")
            # If we can't extract JSON, default to continue if there are remaining plans
            if agent_state.remaining_plans and len(agent_state.remaining_plans) > 0:
                llm_msg.content = '{"continued": true}'
                logger.info("Defaulting to continue due to remaining plans")
            else:
                llm_msg.content = '{"continued": false}'
                logger.info("Defaulting to finish due to no remaining plans")

    llm_msg.with_log()
    agent_state.add_message(llm_msg)
    return agent_state
