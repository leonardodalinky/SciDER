from loguru import logger
from pydantic import BaseModel

from scievo.core import constant
from scievo.core.llms import ModelRegistry
from scievo.core.plan import Plan
from scievo.core.types import Message
from scievo.core.utils import parse_json_from_llm_response
from scievo.prompts import PROMPTS

from .state import DataAgentState

LLM_NAME = "plan"
AGENT_NAME = "data_planner"


@logger.catch
def planner_node(agent_state: DataAgentState) -> DataAgentState:
    logger.trace("planner_node of Agent {}", AGENT_NAME)

    user_query_msg = Message(
        role="user",
        content=agent_state.user_query,
        agent_sender=AGENT_NAME,
    )

    msg = ModelRegistry.completion(
        LLM_NAME,
        [user_query_msg],
        system_prompt=(
            Message(
                role="system", content=PROMPTS.data.planner_system_prompt.render(is_replanner=False)
            )
            .with_log(cond=constant.LOG_SYSTEM_PROMPT)
            .content
        ),
        agent_sender=AGENT_NAME,
    ).with_log()

    # NOTE: we don't add the message to the history
    plans = parse_json_from_llm_response(msg, Plan)

    # NOTE:
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

    # dummy user response, just for logging
    Message(
        role="user",
        content=PROMPTS.data.replanner_user_response.render(
            next_step=agent_state.remaining_plans[0],
        ),
        agent_sender=AGENT_NAME,
    ).with_log()

    return agent_state


def replanner_node(agent_state: DataAgentState) -> DataAgentState:
    logger.trace("replanner_node of Agent {}", AGENT_NAME)

    agent_state.past_plans.append(agent_state.remaining_plans.pop(0))

    # NOTE: when all the plans are done, go into the talk mode
    if len(agent_state.remaining_plans) == 0:
        logger.debug("All plans are done, going into talk mode")
        agent_state.talk_mode = True
        # agent_state.remaining_plans = ["Response to users' query."]
        return agent_state

    user_query = agent_state.user_query

    user_msg = Message(
        role="user",
        content=PROMPTS.data.replanner_user_prompt.render(
            user_query=user_query,
            plan=agent_state.plans.steps,
            past_steps=agent_state.past_plans,
        ),
        agent_sender=AGENT_NAME,
    ).with_log()

    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.patched_history + [user_msg],
        system_prompt=(
            Message(
                role="system", content=PROMPTS.data.planner_system_prompt.render(is_replanner=True)
            )
            .with_log(cond=constant.LOG_SYSTEM_PROMPT)
            .content
        ),
        agent_sender=AGENT_NAME,
    ).with_log()

    class Replan(BaseModel):
        continued: bool = False
        modified: list[str] = []

    # NOTE: we don't add the message to the history
    plans = parse_json_from_llm_response(msg, Replan)

    if plans.continued:
        # no edits to plan
        pass
    else:
        agent_state.plans = Plan(steps=plans.modified)
        agent_state.remaining_plans = plans.modified

    agent_state.add_message(
        Message(
            role="user",
            content=PROMPTS.data.replanner_user_response.render(
                next_step=agent_state.remaining_plans[0],
            ),
            agent_sender=AGENT_NAME,
        )
    )

    return agent_state


def should_replan(agent_state: DataAgentState) -> str:
    if agent_state.talk_mode:
        return "prepare_for_talk_mode"
    else:
        return "gateway"
