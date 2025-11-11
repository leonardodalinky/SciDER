from .registry import register_tool, register_toolset_desc

register_toolset_desc("todo", "Simple todo echo toolset.")


@register_tool(
    "todo",
    {
        "type": "function",
        "function": {
            "name": "todo",
            "description": "Record a todo item and echo it back.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todo": {
                        "type": "string",
                        "description": "The todo item to record.",
                    }
                },
                "required": ["todo"],
            },
        },
    },
)
def todo(todo: str) -> str:
    return todo
