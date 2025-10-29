from . import register_tool


@register_tool(
    "noop",
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "No-op tool: performs no action. You can call this tool to do nothing and move on as an intermediate thinking step.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
)
def noop(ctx_vars) -> str:
    return "No action taken"
