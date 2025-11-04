from .registry import register_tool, register_toolset_desc

register_toolset_desc("noop", "No-op toolset.")


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
def noop() -> str:
    return "No action taken"
