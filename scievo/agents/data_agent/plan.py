from functional import seq
from langgraph.graph import END
from loguru import logger

from scievo.core.llms import ModelRegistry
from scievo.core.plan import Plan
from scievo.core.types import Message
from scievo.core.utils import array_to_bullets, parse_json_from_llm_response
from scievo.prompts import PROMPTS

from .state import DataAgentState

LLM_NAME = "plan"
AGENT_NAME = "data"


@logger.catch
def planner_node(agent_state: DataAgentState) -> DataAgentState:
    logger.trace("planner_node of Agent {}", AGENT_NAME)

    system_prompt = PROMPTS.data.planner_system_prompt

    assert len(agent_state.history) == 1, "History should only have one message for planner node"

    msg = ModelRegistry.completion(
        LLM_NAME,
        [agent_state.history[0]],
        system_prompt,
        agent_sender=AGENT_NAME,
    ).with_log()

    # NOTE: we don't add the message to the history
    plans = parse_json_from_llm_response(msg, Plan)

    agent_state.plans = plans
    agent_state.remaining_plans = plans.steps
    agent_state.past_plans = []

    # dummy user response, just for logging
    Message(
        role="user",
        content=PROMPTS.data.replanner_user_response.format(
            next_step=agent_state.remaining_plans[0],
        ),
    ).with_log()

    return agent_state


def replanner_node(agent_state: DataAgentState) -> DataAgentState:
    logger.trace("replanner_node of Agent {}", AGENT_NAME)

    agent_state.past_plans.append(agent_state.remaining_plans.pop(0))

    # TODO: when all the plans are done, go into the talk mode
    if len(agent_state.remaining_plans) == 0:
        logger.debug("All plans are done, going into talk mode")
        agent_state.talk_mode = True
        agent_state.remaining_plans = ["Response to users' query."]
        return agent_state

    system_prompt = PROMPTS.data.replanner_system_prompt

    user_query = (
        seq(agent_state.history).filter(lambda msg: msg.role == "user").head_option(no_wrap=True)
        or "No user query provided."
    )

    user_msg = Message(
        role="user",
        content=PROMPTS.data.replanner_user_prompt.format(
            user_query=user_query,
            plan=array_to_bullets(agent_state.plans.steps),
            past_steps=array_to_bullets(agent_state.past_plans),
        ),
    ).with_log()

    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.history + [user_msg],
        system_prompt,
        agent_sender=AGENT_NAME,
    ).with_log()

    # NOTE: we don't add the message to the history
    plans = parse_json_from_llm_response(msg, Plan)

    if len(plans.steps) == 0:
        # no edits to plan
        pass
    else:
        agent_state.plans = plans
        agent_state.remaining_plans = plans.steps

    agent_state.history.append(
        Message(
            role="user",
            content=PROMPTS.data.replanner_user_response.format(
                next_step=agent_state.remaining_plans[0],
            ),
        )
    )

    return agent_state


def should_replan(agent_state: DataAgentState) -> str:
    if agent_state.talk_mode:
        return END
    else:
        return "gateway"
