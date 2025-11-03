from ..core.constant import __GRAPH_STATE_NAME__
from ..core.types import GraphState
from . import register_tool


@register_tool(
    "state",
    {
        "type": "function",
        "function": {
            "name": "change_toolset",
            "description": "Change the current toolset for the agent. Available toolsets can be seen in the system prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "toolset": {
                        "type": "string",
                        "description": "The name of the toolset to change to",
                    }
                },
                "required": ["toolset"],
            },
        },
    },
)
def change_toolset(graph_state: GraphState, ctx: dict, toolset: str) -> str:
    """
    Change the toolset for the current agent.

    Args:
        graph_state: The graph state dictionary
        ctx: The context dictionary
        toolset: The name of the toolset to change to

    Returns:
        Confirmation message of the toolset change
    """
    try:
        # Get the agent name from the graph state
        # We need to find which agent is currently active
        from ..tools import ToolRegistry

        # Check if the toolset exists
        available_toolsets = list(ToolRegistry.get_instance()._toolsets_desc.keys())
        if toolset not in available_toolsets:
            return f"Error: Toolset '{toolset}' is not available. Available toolsets: {', '.join(available_toolsets)}"

        current_agent = ctx.get("current_agent", None)
        if current_agent is None:
            return "Error: No current agent found in context"
        elif current_agent not in graph_state.agents:
            return f"Error: Agent '{current_agent}' not found in graph state"

        graph_state.agents[current_agent].toolset = toolset

        return f"Successfully changed toolset to '{toolset}' for agent '{current_agent}'"
    except Exception as e:
        return f"Error changing toolset: {e}"
