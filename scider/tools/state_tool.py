from scider.core.constant import __AGENT_STATE_NAME__
from scider.core.types import ToolsetState

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
def activate_toolset(agent_state: ToolsetState, ctx: dict, toolset: str) -> str:
    from scider.tools import ToolRegistry

    try:
        # Check if the toolset exists
        available_toolsets = list(ToolRegistry.instance()._toolsets_desc.keys())
        if toolset not in available_toolsets:
            return f"Error: Toolset '{toolset}' is not available. Available toolsets: {', '.join(available_toolsets)}"

        toolsets = agent_state.toolsets
        if toolset in toolsets:
            toolsets.remove(toolset)
            toolsets.insert(0, toolset)
        else:
            toolsets.insert(0, toolset)
            if len(toolsets) > MAX_ACTIVE_TOOLSETS:
                toolsets = toolsets[:MAX_ACTIVE_TOOLSETS]

        agent_state.toolsets = toolsets

        return f"Successfully activated toolset '{toolset}'. Active toolsets: {', '.join(toolsets)}"
    except Exception as e:
        return f"Error activating toolset: {e}"
