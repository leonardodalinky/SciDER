"""
Execution nodes for the Coding Subagent V3 Claude

This module provides a minimal execution flow that delegates all coding work
to Claude Agent SDK. The flow is: START -> claude_node -> summary_node -> END
"""

import os

from loguru import logger

from scievo.core import constant
from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.prompts import PROMPTS
from scievo.tools.claude_agent_sdk_tool import run_claude_agent_sdk

from .state import ClaudeCodingAgentState

LLM_NAME = "experiment_coding"
AGENT_NAME = "experiment_coding"


def claude_node(agent_state: ClaudeCodingAgentState) -> ClaudeCodingAgentState:
    """
    Execute the coding task using Claude Agent SDK.

    This node directly invokes the Claude Agent SDK to handle
    the entire coding workflow. Claude Agent SDK has its own internal planning,
    tool calling, and execution mechanisms.
    """
    logger.debug("claude_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("claude")

    try:
        # Construct the prompt for Claude Agent SDK
        instruction = agent_state.user_query or "No specific coding task provided."
        bg_info = agent_state.data_summary or "No background information available."
        workspace_dir = os.path.abspath(agent_state.workspace.working_dir)

        prompt = f"""\
# Requirements:
- At the end of your response, provide a detailed explanation of what you did and why.
- Ensure that all changes are made in a way that maintains the integrity of the codebase.

# Workspace
{workspace_dir}

# Background information:
{bg_info}

# Task:
{instruction}
"""

        logger.info("Sending task to Claude Agent SDK: {}", instruction[:100])

        # Call Claude Agent SDK tool
        result = run_claude_agent_sdk(
            prompt=prompt,
            cwd=workspace_dir,
            allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            permission_mode="acceptEdits",
            **{constant.__AGENT_STATE_NAME__: agent_state},
        )

        # Minimal implementation: use result directly without TOON parsing
        result_str = str(result)

        has_error = False
        lines = result_str.split("\n")
        for line in lines[:10]:  # Check first 10 lines for error field
            stripped = line.strip()
            # Check for explicit error field (not error=None which is normal)
            if stripped.startswith("error:") and "error=None" not in line:
                has_error = True
                break

        # Success case: task completed successfully
        logger.info("Claude Agent SDK completed task")
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Claude Agent SDK Result]\nClaude Agent SDK has completed the coding task. The changes have been applied to the workspace.",
                agent_sender="claude_agent_sdk",
            ).with_log()
        )

    except Exception as e:
        logger.exception("Claude Agent SDK error")
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Claude Agent SDK Error] {str(e)}",
                agent_sender="claude_agent_sdk",
            ).with_log()
        )

    return agent_state


def summary_node(agent_state: ClaudeCodingAgentState) -> ClaudeCodingAgentState:
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
