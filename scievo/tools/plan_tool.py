"""
These tool is used to make plans for agents.
"""

from ..core.registry import register_tool, registry

NEXT_PLAN_PROMPT = """\
Try to think step by step to give the answer. Note that, before moving to next step, please think and self-relect on the generated answer of the previous step."""


@register_tool(
    "create_plans",
    json_schema={
        "type": "function",
        "function": {
            "name": "create_plans",
            "description": "Create new plans.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plans": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The plans to be created",
                    },
                    # "plan_step": {
                    #     "type": "integer",
                    #     "description": "The plan step to be updated, range: [1, len(plans) + 1]. If set to len(plans) + 1, it means the plan has reached the end. If not set, it will be set to 1.",
                    # },
                },
                "required": ["plans"],
            },
        },
    },
)
def create_plans(
    ctx_vars: dict,
    plans: list[str],
    # plan_step: int = 1,
) -> str:
    assert "plan_step" in ctx_vars
    # assert (
    #     len(plans) + 1 >= plan_step >= 0
    # ), "Invalid plan step. The plan step should be in the range [1, len(plans) + 1]"
    # plan_step = ctx_vars["plan_step"]
    plan_step = 1
    ctx_vars["plans"] = plans
    ctx_vars["plans"].append("Generate a summary of processes and answers of the plan")
    ctx_vars["plan_step"] = plan_step
    if plan_step == len(plans) + 1:
        return """\
The plans have been created, and it has reached the end.
"""
    else:
        return f"""\
The plans have been created. The current step {plan_step} is:
{plans[plan_step - 1]}

{NEXT_PLAN_PROMPT}"""


@register_tool(
    "set_plan_answer_and_next_step",
    json_schema={
        "type": "function",
        "function": {
            "name": "set_plan_answer_and_next_step",
            "description": "Set the answer for the current plan step and move to the next plan step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The answer for the current plan step.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
)
def set_plan_answer_and_next_step(ctx_vars: dict, answer: str) -> str:
    assert "plan_step" in ctx_vars
    assert "plans" in ctx_vars
    ctx_vars["plan_step"] += 1
    if ctx_vars["plan_step"] > len(ctx_vars["plans"]):
        return f"""\
The plan answer for step {ctx_vars["plan_step"]} has been set. And the plan has reached the end.
"""
    else:
        return f"""\
The plan answer for step {ctx_vars["plan_step"]} has been set. Now move to the next plan step {ctx_vars["plan_step"]}:
{ctx_vars["plans"][ctx_vars["plan_step"] - 1]}

{NEXT_PLAN_PROMPT}
"""


@register_tool(
    "push_agent",
    json_schema={
        "type": "function",
        "function": {
            "name": "push_agent",
            "description": "Push the agent to the agent stack. Return a string indicating the result of the push.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the agent to be pushed to the agent stack.",
                    },
                },
                "required": ["name"],
            },
        },
    },
)
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
        "function": {
            "name": "pop_agent",
            "description": "Pop the agent from the agent stack, indicating the end of the current agent. Return a string indicating the result of the pop and hint the next plan step from the previous plan.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
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
{ctx_vars["plans"][ctx_vars["plan_step"] - 1]}

{NEXT_PLAN_PROMPT}"""
