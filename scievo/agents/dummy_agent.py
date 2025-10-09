from ..core.registry import register_agent
from ..core.types import Agent
from ..tools.dummy_tool import dummy_func
from ..tools.plan_tool import plan_next_step, pop_agent, push_agent, update_plan


@register_agent("dummy_agent")
def get_dummy_agent(model: str, **kwargs):
    def instructions(ctx_vars):
        return f"""\
You are a helpful agent that can respond to the users' instruction.

You should always use the `update_plan` tool to update the plans and plan step before you respond to the user. And use the `plan_next_step` tool to move to the next plan step, until the plan has reached the end.
If you want to switch to another agent, use the `push_agent` tool to push the agent to the agent stack, and use the `pop_agent` tool to pop the agent from the agent stack when finished.
"""

    tool_list = [dummy_func, update_plan, plan_next_step, push_agent, pop_agent]

    return Agent(
        name="Dummy Agent",
        model=model,
        instructions=instructions,
        functions=tool_list,
        tool_choice="required",
    )
