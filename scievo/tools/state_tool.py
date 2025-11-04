from ..core.constant import __GRAPH_STATE_NAME__
from ..core.types import GraphState
from .registry import register_tool, register_toolset_desc

register_toolset_desc("state", "State management toolset.")

MAX_ACTIVE_TOOLSETS = 2


@register_tool(
    "state",
    {
        "type": "function",
        "function": {
            "name": "activate_toolset",
            "description": f"Activate a toolset for the agent. Available toolsets can be seen in the system prompt. Only the last {MAX_ACTIVE_TOOLSETS} toolsets can be active at the same time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "toolset": {
                        "type": "string",
                        "description": "The name of the toolset to activate",
                    }
                },
                "required": ["toolset"],
            },
        },
    },
)
def activate_toolset(graph_state: GraphState, ctx: dict, toolset: str) -> str:
    try:
        # Get the agent name from the graph state
        # We need to find which agent is currently active
        from ..tools import ToolRegistry

        # Check if the toolset exists
        available_toolsets = list(ToolRegistry.instance()._toolsets_desc.keys())
        if toolset not in available_toolsets:
            return f"Error: Toolset '{toolset}' is not available. Available toolsets: {', '.join(available_toolsets)}"

        current_agent = ctx.get("current_agent", None)
        if current_agent is None:
            return "Error: No current agent found in context"
        elif current_agent not in graph_state.agents:
            return f"Error: Agent '{current_agent}' not found in graph state"

        toolsets = graph_state.agents[current_agent].toolsets
        if toolset in toolsets:
            toolsets.remove(toolset)
            toolsets.insert(0, toolset)
        else:
            toolsets.insert(0, toolset)
            if len(toolsets) > MAX_ACTIVE_TOOLSETS:
                toolsets = toolsets[:MAX_ACTIVE_TOOLSETS]

        graph_state.agents[current_agent].toolsets = toolsets

        return f"Successfully activated toolset '{toolset}' for agent '{current_agent}'. Active toolsets: {', '.join(toolsets)}"
    except Exception as e:
        return f"Error activating toolset: {e}"
