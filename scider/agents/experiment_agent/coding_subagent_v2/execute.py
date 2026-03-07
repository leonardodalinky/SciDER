"""
Execution nodes for the Coding Subagent V2

This module provides a minimal execution flow that delegates all coding work
to OpenHands SDK. The flow is: START -> openhands_node -> summary_node -> END
"""

import os

from loguru import logger
from openhands.sdk.event import ActionEvent

from scider.core import constant
from scider.core.llms import ModelRegistry
from scider.core.types import Message
from scider.prompts import PROMPTS

from .state import CodingAgentState

LLM_NAME = "experiment_coding"
AGENT_NAME = "experiment_coding"


def openhands_node(agent_state: CodingAgentState) -> CodingAgentState:
    """
    Execute the coding task using OpenHands sub-agent.

    This node directly invokes the OpenHands conversation to handle
    the entire coding workflow. OpenHands has its own internal planning,
    tool calling, and execution mechanisms.
    """
    logger.debug("openhands_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("openhands")

    conversation = agent_state.openhands_conversation
    if conversation is None:
        logger.error("OpenHands conversation not initialized")
        agent_state.output_summary = "Error: OpenHands conversation not initialized."
        return agent_state

    try:
        # Construct the message for OpenHands
        instruction = agent_state.user_query or "No specific coding task provided."
        bg_info = agent_state.data_summary or "No background information available."
        # prefix with `> ` for markdown blockquote
        instruction = "\n".join([f"> {line}" for line in instruction.splitlines()])
        bg_info = "\n".join([f"> {line}" for line in bg_info.splitlines()])
        workspace_dir = os.path.abspath(agent_state.workspace.working_dir)

        message = f"""\
# Requirements:
- At the end of your response, provide a detailed explanation of what you did and why.
- Ensure that all changes are made in a way that maintains the integrity of the codebase.
- Avoid long-running executions of training or data processing; focus on code changes. If needed for code testing, design some simple test code instead.

# Important Notes:
- DO NOT train the full model. Just train a demo if needed for testing code changes.
- DO NOT run large data processing tasks. Just simulate with small data if needed for testing code
- Always ensure that the code runs without errors after your changes.
- I would run the full experiments later after getting your code changes.

# Workspace
{workspace_dir}

# Task:
{instruction}

# Background information:
```
{bg_info}
```
"""

        logger.info("Sending task to OpenHands sub-agent: {}", instruction[:100])

        # Send message to the OpenHands agent
        conversation.send_message(message)

        # Run the agent until completion
        with agent_state.workspace:
            conversation.run()

        # Extract the last response from OpenHands
        if conversation.state.events:
            for e in reversed(conversation.state.events):
                if isinstance(e, ActionEvent) and e.source == "agent":
                    if hasattr(e, "llm_message") and e.llm_message:
                        content = e.llm_message.content
                    elif (m := getattr(e, "to_llm_message", None)) is not None and callable(m):
                        content = m().content
                    else:
                        # Unable to extract content from this event
                        continue
                    last_response = "\n".join([c.text for c in content])
                    break
            else:
                last_response = "Coding task completed (no detailed response available)."
        else:
            last_response = "Coding task completed (no detailed response available)."

        # Log the result
        logger.info("OpenHands sub-agent completed task")

        # Store the response in history for summary generation
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[OpenHands Sub-Agent Result]\n{last_response}",
                agent_sender="openhands",
            ).with_log()
        )

    except Exception as e:
        logger.exception("OpenHands agent error")
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[OpenHands Error] {str(e)}",
                agent_sender="openhands",
            ).with_log()
        )

    return agent_state


def summary_node(agent_state: CodingAgentState) -> CodingAgentState:
    """Generate summary of the coding workflow and results."""
    logger.debug("summary_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("summary")

    # Add summary generation prompt from PROMPTS
    summary_prompt = Message(
        role="user",
        content=PROMPTS.experiment_coding_v2.summary_prompt.render(),
        agent_sender=AGENT_NAME,
    )
    agent_state.add_message(summary_prompt)

    # Get summary from LLM
    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.patched_history,
        system_prompt=(
            Message(
                role="system",
                content=PROMPTS.experiment_coding_v2.summary_system_prompt.render(),
            )
            .with_log(cond=constant.LOG_SYSTEM_PROMPT)
            .content
        ),
        agent_sender=AGENT_NAME,
        tools=None,  # No tools needed for final summary
    ).with_log()

    # Store the summary text
    agent_state.output_summary = msg.content or ""
    agent_state.add_message(msg)

    logger.info(f"Coding task summary generated: {len(agent_state.output_summary)} characters")

    return agent_state
