from loguru import logger
from pydantic import BaseModel

from scievo.core import constant
from scievo.core.llms import ModelRegistry
from scievo.core.plan import Plan
from scievo.core.types import Message
from scievo.core.utils import parse_json_from_llm_response
from scievo.prompts import PROMPTS

from .state import CodingAgentState

LLM_NAME = "plan"
AGENT_NAME = "experiment_coding_planner"


@logger.catch
def planner_node(agent_state: CodingAgentState) -> CodingAgentState:
    """Initial planning for the coding task."""
    logger.trace("planner_node of Agent {}", AGENT_NAME)

    user_query_msg = Message(
        role="user",
        content=agent_state.user_query,
        agent_sender=AGENT_NAME,
    )

    agent_state.add_message(user_query_msg)

    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.patched_history,
        system_prompt=(
            Message(
                role="system",
                content=PROMPTS.experiment_coding_v2.planner_system_prompt.render(
                    is_replanner=False
                ),
            )
            .with_log(cond=constant.LOG_SYSTEM_PROMPT)
            .content
        ),
        agent_sender=AGENT_NAME,
    ).with_log()

    agent_state.add_message(msg)

    plans = parse_json_from_llm_response(msg, Plan)

    agent_state.add_message(
        Message(
            role="user",
            content="Follow the current plan.",
            agent_sender=AGENT_NAME,
        )
    )

    agent_state.plans = plans
    agent_state.remaining_plans = plans.steps
    agent_state.past_plans = []

    # Dummy user response, just for logging
    Message(
        role="user",
        content=PROMPTS.experiment_coding_v2.replanner_user_response.render(
            next_step=agent_state.remaining_plans[0],
        ),
        agent_sender=AGENT_NAME,
    ).with_log()

    return agent_state


def replanner_node(agent_state: CodingAgentState) -> CodingAgentState:
    """Replan based on critic feedback and execution results."""
    logger.trace("replanner_node of Agent {}", AGENT_NAME)

    agent_state.past_plans.append(agent_state.remaining_plans.pop(0))

    # Check if all plans are done
    if len(agent_state.remaining_plans) == 0:
        logger.debug("All plans are done, going into talk mode")
        agent_state.talk_mode = True
        return agent_state

    user_query = agent_state.user_query
    critic_feedback = agent_state.critic_feedback

    user_msg = Message(
        role="user",
        content=PROMPTS.experiment_coding_v2.replanner_user_prompt.render(
            user_query=user_query,
            plan=agent_state.plans.steps,
            past_steps=agent_state.past_plans,
            critic_feedback=critic_feedback,
        ),
        agent_sender=AGENT_NAME,
    ).with_log()

    agent_state.add_message(user_msg)

    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.patched_history,
        system_prompt=(
            Message(
                role="system",
                content=PROMPTS.experiment_coding_v2.planner_system_prompt.render(
                    is_replanner=True
                ),
            )
            .with_log(cond=constant.LOG_SYSTEM_PROMPT)
            .content
        ),
        agent_sender=AGENT_NAME,
    ).with_log()

    agent_state.add_message(msg)

    class Replan(BaseModel):
        continued: bool | None = None
        modified: list[str] = []

    plans = parse_json_from_llm_response(msg, Replan)

    if plans.continued is True:
        pass  # No changes to plan
    elif plans.continued is False:
        # plans done
        logger.debug("Replanner indicates all plans are done, going into talk mode")
        agent_state.talk_mode = True
        return agent_state
    else:
        agent_state.plans = Plan(steps=plans.modified)
        agent_state.remaining_plans = plans.modified

    agent_state.add_message(
        Message(
            role="user",
            content=PROMPTS.experiment_coding_v2.replanner_user_response.render(
                next_step=agent_state.remaining_plans[0],
            ),
            agent_sender=AGENT_NAME,
        )
    )

    return agent_state


def should_replan(agent_state: CodingAgentState) -> str:
    if agent_state.talk_mode:
        return "prepare_for_completion"
    else:
        return "gateway"
