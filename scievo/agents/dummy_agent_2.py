from ..core.registry import register_agent
from ..core.types import Agent
from ..tools.plan_tool import create_plans, pop_agent, set_plan_answer_and_next_step


@register_agent("dummy_agent_2")
def get_dummy_agent_2(model: str, **kwargs):
    def instructions(ctx_vars):
        return f"""\
You are a helpful agent that can respond to the users' instruction.

First, break down the user's question into multiple task goals. Then think step by step to give the answer.

Function calls instruction:
- You should always use the `create_plans` tool to create the plans before you respond to the user.
- And use the `set_plan_answer_and_next_step` tool to set the answer for the current plan step and move to the next plan step, until the plan has reached the end.
- Only call one tool in one turn.

The overall procedure is:
1. Saying "hello from dummy agent 2"
2. Saying "This is the 2nd plan of dummy agent 2"
"""

    tool_list = [create_plans, set_plan_answer_and_next_step]

    return Agent(
        name="Dummy Agent 2",
        model=model,
        instructions=instructions,
        functions=tool_list,
        tool_choice="required",
    )
