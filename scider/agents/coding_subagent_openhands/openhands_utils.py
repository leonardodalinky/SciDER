"""
OpenHands integration tool for coding tasks.

This tool provides access to an external OpenHands coding agent.
The agent instance should be passed via the tool context (ctx).
"""

import os
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from openhands.sdk import Conversation, LocalConversation


def code_subagent(agent_state, instruction: str, bg_info: str) -> str:
    """
    Execute a coding task using the OpenHands agent.

    Args:
        instruction: Natural language instruction for the coding task
        bg_info: Background information for the coding task

    Returns:
        Result message from the coding agent
    """
    logger.debug("Calling OpenHands code_subagent with instruction: {}", instruction)

    if not instruction.strip():
        return "Error: instruction must be a non-empty string."

    enable_openhands = os.getenv("SCIDER_ENABLE_OPENHANDS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if not enable_openhands:
        return (
            "Error: OpenHands toolset is disabled.\n"
            "Hint: set env `SCIDER_ENABLE_OPENHANDS=1` to enable it, or switch to the Claude coding subagent "
            "(set `CODING_AGENT_VERSION=v3`)."
        )

    # Setup openhands paths first (must be before any openhands imports)
    # Keep this import local to avoid mutating sys.path unless OpenHands is explicitly enabled.
    from scider.core import openhands_import  # noqa: F401

    conversation: "Conversation | LocalConversation" = getattr(
        agent_state, "openhands_conversation", None
    )
    if conversation is None:
        return "Error: openhands_conversation not found in agent state."

    try:
        # Send message to the OpenHands agent
        conversation.send_message(
            f"""\
# Requirements:
- At the end of your response, provide a detailed explanation of what you did and why.
- Ensure that all changes are made in a way that maintains the integrity of the codebase.

# Workspace
{os.path.abspath(agent_state.workspace.working_dir)}

# Background information:
{bg_info}

# Task:
{instruction}
"""
        )

        # Run the agent until completion
        conversation.run()

        if (e := conversation.state.events[-1]).source == "agent":
            last_response = "\n".join([c.text for c in e.llm_message.content])
        else:
            last_response = "Error: No response from the coding agent at the end."

        # Return success message
        # Note: The exact response format depends on OpenHands SDK API
        # This may need adjustment based on actual SDK implementation
        return f"""\
Coding task completed. The summary of changes is as follows:
{last_response}
"""

    except Exception as e:
        logger.exception("OpenHands agent error")
        return f"Error executing coding task: {str(e)}"
