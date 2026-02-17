"""
Execution nodes for the Coding Subagent V3 Claude

This module provides a minimal execution flow that delegates all coding work
to Claude Agent SDK. The flow is: START -> claude_node -> summary_node -> END
"""

import json
import os

from jinja2 import Template
from loguru import logger

from scievo.core import constant
from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.prompts import PROMPTS
from scievo.tools.claude_agent_sdk_tool import run_claude_agent_sdk
from scievo.tools.claude_code_tool import run_claude_code

from .state import ClaudeCodingAgentState

LLM_NAME = "experiment_coding"
AGENT_NAME = "experiment_coding"

CLAUDE_PROMPT: Template = Template(
    """\
# Requirements:
- At the end of your response, provide a detailed explanation of what you did and why.
- Avoid directly reading large files; instead, read the specific parts you need or just read the first few lines.
- Ensure that all changes are made in a way that maintains the integrity of the codebase.
- Avoid long-running executions of training or data processing; focus on code changes. If needed for code testing, design some simple test code instead.

# Important Notes:
- DO NOT train the full model. Just train a demo if needed for testing code changes.
- DO NOT run large data processing tasks. Just simulate with small data if needed for testing code
- Always ensure that the code runs without errors after your changes.
- I would run the full experiments later after getting your code changes.

# Workspace
{{ workspace_dir }}
{% if bg_info %}
# Background information:
```
{{ bg_info }}
```
{% endif %}
# Task:
{{ instruction }}
"""
)


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
        # Construct the message for Claude Agent SDK
        instruction = agent_state.user_query or "No specific coding task provided."
        bg_info = agent_state.data_summary or ""
        # prefix with `> ` for markdown blockquote
        instruction = "\n".join([f"> {line}" for line in instruction.splitlines()])
        bg_info = "\n".join([f"> {line}" for line in bg_info.splitlines()])
        workspace_dir = os.path.abspath(agent_state.workspace.working_dir)

        prompt = CLAUDE_PROMPT.render(
            workspace_dir=workspace_dir,
            instruction=instruction,
            bg_info=bg_info,
        )

        logger.info("Sending task to Claude Agent SDK: {}", instruction[:100])

        # Call Claude Agent SDK tool (preferred)
        sdk_result = run_claude_agent_sdk(
            prompt=prompt,
            cwd=workspace_dir,
            allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            permission_mode="acceptEdits",
            **{constant.__AGENT_STATE_NAME__: agent_state},
        )
        try:
            raw_sdk_result: dict = json.loads(sdk_result)
        except Exception:
            raw_sdk_result = None

        sdk_text = str(sdk_result)
        has_error = any(
            (line.strip().startswith("error:") and "error=None" not in line)
            for line in sdk_text.splitlines()[:20]
        )

        if not has_error:
            logger.info("Claude Agent SDK completed task")
            agent_state.add_message(
                Message(
                    role="assistant",
                    content=(
                        "[Claude Agent SDK Result]\n"
                        "Claude Agent SDK has completed the coding task. The changes have been applied to the workspace.\n\n"
                        f"{sdk_text}"
                    ),
                    agent_sender="claude_agent_sdk",
                ).with_log()
            )
        else:
            # Fallback to Claude Code CLI (still Claude-based, but doesn't require SDK install)
            logger.warning("Claude Agent SDK returned an error; falling back to Claude Code CLI")
            cli_result = run_claude_code(
                instruction=prompt,
                cwd=workspace_dir,
                timeout=1800,
                **{constant.__AGENT_STATE_NAME__: agent_state},
            )
            agent_state.add_message(
                Message(
                    role="assistant",
                    content=(
                        "[Claude Agent SDK Error]\n"
                        f"{sdk_text}\n\n"
                        "[Claude Code CLI Fallback Result]\n"
                        f"{str(cli_result)}"
                    ),
                    agent_sender="claude_code",
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

    claude_output = "Claude Agent SDK execution completed"
    if agent_state.history:
        last_msg = agent_state.history[-1]
        if last_msg.role == "assistant" and last_msg.content:
            if agent_state.intermediate_full_output:
                claude_output = last_msg.content
            else:
                claude_output = last_msg.content[:2000]

    agent_state.intermediate_state.append(
        {
            "node_name": "claude",
            "output": claude_output,
            "_raw_claude_result": raw_sdk_result,
        }
    )

    return agent_state


def summary_node(agent_state: ClaudeCodingAgentState) -> ClaudeCodingAgentState:
    """Generate summary of the coding workflow and results."""
    logger.debug("summary_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("summary")

    if agent_state.skip_summary:
        logger.info("Skipping summary generation as per agent state setting")
        agent_state.add_message(
            Message(
                role="assistant",
                content="[Summary generation skipped]",
                agent_sender="summary_node",
            ).with_log()
        )
        agent_state.output_summary = "Summary generation skipped."
        return agent_state

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

    agent_state.intermediate_state.append(
        {
            "node_name": "summary",
            "output": agent_state.output_summary or "No summary generated",
        }
    )

    return agent_state
