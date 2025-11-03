from ..core.types import GraphState
from . import register_tool


@register_tool(
    "dummy",
    {
        "type": "function",
        "function": {
            "name": "dummy_func",
            "description": "Dummy function. Only shows the graph state.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
)
def dummy_func(graph_state: GraphState) -> str:
    return f"The graph state is {graph_state}"
