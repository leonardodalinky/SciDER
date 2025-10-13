from ..core.registry import register_agent
from ..core.types import Agent
from ..tools.plan_tool import create_plans, set_plan_answer_and_next_step


@register_agent("dummy_agent")
def get_dummy_agent(model: str, **kwargs):
    def instructions(ctx_vars):
        return f"""\
You are a helpful agent that can respond to the users' instruction.

First, break down the user's question into multiple task goals. Then think step by step to give the answer.

You should always use the `create_plans` tool to create the plans before you respond to the user. And use the `set_plan_answer_and_next_step` tool to set the answer for the current plan step and move to the next plan step, until the plan has reached the end.
"""

    # If you want to switch to another agent, use the `push_agent` tool to push the agent to the agent stack, and use the `pop_agent` tool to pop the agent from the agent stack when finished.

    tool_list = [create_plans, set_plan_answer_and_next_step]

    return Agent(
        name="Dummy Agent",
        model=model,
        instructions=instructions,
        functions=tool_list,
        tool_choice="auto",
    )
