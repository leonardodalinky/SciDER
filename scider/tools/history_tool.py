"""
Tool for restoring and recalling conversation history.
"""

from scider.core.types import HistoryState
from scider.prompts import PROMPTS

from .registry import register_tool, register_toolset_desc

register_toolset_desc(
    "history",
    "History management toolset for recalling and managing conversation history patches (compressed/summarized conversation segments). "
    "NOTE: This toolset should be used cautiously as it may lead to a large context window and previous conversation history being compressed.",
)


@register_tool(
    "history",
    {
        "type": "function",
        "function": {
            "name": "recall_history_patch",
            "description": "Recall a specific history patch (compressed/summarized conversation segment) by its patch ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch_id": {
                        "type": "integer",
                        "description": "The ID of the patch to recall.",
                    },
                },
                "required": ["patch_id"],
            },
        },
    },
)
def recall_history_patch(agent_state: HistoryState, patch_id: int) -> str:
    """
    Recall a specific history patch and show both the original messages
    and the compressed version.
    """
    # TODO
    try:
        patch = agent_state.get_patch_by_id(patch_id)

        if patch is None:
            return f"Error: Patch with ID {patch_id} not found. Available patches: {[p.patch_id for p in agent_state.history_patches]}"

        original_messages = agent_state.partial_history_of_patch(patch_id)

        # Format the result
        history_text = ""

        for i, msg in enumerate(original_messages, start=patch.start_idx):
            history_text += f"--- Message {i} Begin ---\n"
            history_text += msg.to_plain_text(verbose_tool=False)
            history_text += f"--- Message {i} End ---\n"

        return PROMPTS.history.recall_tool_response.render(
            patch_id=patch_id,
            n_messages=len(patch.n_messages),
            history_text=history_text,
        )

    except Exception as e:
        return f"Error recalling patch: {e}"
