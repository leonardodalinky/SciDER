"""
These tool is used to make plans for agents.
"""

from ..core.registry import register_tool, registry


@register_tool(
    "update_plan",
    json_schema={
        "type": "function",
        "description": "Update the plans and plan step. Return a string indicating the result of the update.",
        "parameters": {
            "type": "object",
            "properties": {
                "plans": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The plans to be updated",
                },
                "plan_step": {
                    "type": "integer",
                    "description": "The plan step to be updated, range: [1, len(plans) + 1]. If set to len(plans) + 1, it means the plan has reached the end. If not set, it will be set to 1.",
                },
            },
            "required": ["plans"],
        },
    },
)
def update_plan(
    ctx_vars: dict,
    plans: list[str],
    plan_step: int = 1,
) -> str:
    assert "plan_step" in ctx_vars
    assert (
        len(plans) + 1 >= plan_step >= 0
    ), "Invalid plan step. The plan step should be in the range [1, len(plans) + 1]"
    plan_step = plan_step or ctx_vars["plan_step"]
    ctx_vars["plans"] = plans
    ctx_vars["plan_step"] = plan_step
    if plan_step == len(plans) + 1:
        return """\
The plan has been updated, and it has reached the end.
"""
    else:
        return f"""\
The plan has been updated. The current step {plan_step} is:
{plans[plan_step - 1]}"""


@register_tool(
    "plan_next_step",
    json_schema={
        "type": "function",
        "description": "Move to the next plan step. Return a string indicating the result of the update.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
)
def plan_next_step(ctx_vars: dict) -> str:
    assert "plan_step" in ctx_vars
    assert "plans" in ctx_vars
    old_step = ctx_vars["plan_step"]
    if old_step > len(ctx_vars["plans"]):
        return """\
The plan has reached the end.
"""
    else:
        new_step = ctx_vars["plan_step"] + 1
        ctx_vars["plan_step"] = new_step
        return f"""\
The plan has been updated. The current step {new_step} is:
{ctx_vars["plans"][new_step - 1]}"""


def push_agent(ctx_vars: dict, name: str) -> str:
    assert "agent_stack" in ctx_vars
    assert "plans_stack" in ctx_vars
    assert "agent" in ctx_vars
    agent_init_func = registry.agents[name]
    agent = agent_init_func(**ctx_vars)

    # store current plans and plan step
    current_plans = ctx_vars["plans"]
    current_plan_step = ctx_vars["plan_step"]
    ctx_vars["plans_stack"].append((current_plan_step, current_plans))

    # push agent to agent stack
    ctx_vars["agent_stack"].append(ctx_vars["agent"])
    return f"""\
The agent {name} has been pushed to agent stack."""


@register_tool(
    "pop_agent",
    json_schema={
        "type": "function",
        "description": "Pop the agent from the agent stack, indicating the end of the current agent. Return a string indicating the result of the pop and hint the next plan step from the previous plan.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
)
def pop_agent(ctx_vars: dict) -> str:
    assert "agent_stack" in ctx_vars
    assert "plans_stack" in ctx_vars
    assert "agent" in ctx_vars
    # restore agent from agent stack
    ctx_vars["agent_stack"].pop()
    prev_agent = ctx_vars["agent"]
    # restore plans from plans stack
    (ctx_vars["plan_step"], ctx_vars["plans"]) = ctx_vars["plans_stack"].pop()
    return f"""\
The agent has been popped from agent stack. The next plan step {ctx_vars["plan_step"]} of the previous agent {prev_agent.name} is:
{ctx_vars["plans"][ctx_vars["plan_step"] - 1]}"""
