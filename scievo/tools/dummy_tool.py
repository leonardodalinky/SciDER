from ..core.registry import register_tool


@register_tool(
    "dummy_func",
    json_schema={
        "type": "function",
        "description": "Dummy function. Only shows the context variables.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
)
def dummy_func(ctx_vars) -> str:
    return f"The context variables is {ctx_vars}"
