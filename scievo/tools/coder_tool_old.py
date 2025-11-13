"""
**Deprecated**

This is the old version of the coder tool, which uses aider as a library in the current Python environment.
"""

import io
from contextlib import redirect_stdout

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from scievo.core import constant

from .registry import register_tool, register_toolset_desc

register_toolset_desc(
    "coder-old",
    "Coder toolset. This toolset allows you to call an AI coding agent using natural language instructions. "
    "The agent can modify, create, or edit code files based on your instructions. "
    "But you have to provide the details, like the file structure, the file names, the file paths, the file content, etc."
    "Note: The coding agent has no memory of previous interactions and operates independently on each call.",
)


@register_tool(
    "coder",
    {
        "type": "function",
        "function": {
            "name": "run_coder",
            "description": (
                "Execute a coding task using an AI coding agent (aider). "
                "This tool calls another agent with natural language instructions to modify, create, or edit code files. "
                "But you have to provide the details, like the file structure, the file names, the file paths, the file content, etc."
                "IMPORTANT: The coding agent has no memory of previous interactions - each call is independent. "
                "Provide clear, complete instructions for each task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fnames": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of file paths or directory paths to add to the coding session. "
                            "The agent will have access to read and modify these files. "
                            "If not provided, the agent will work without pre-loaded files."
                        ),
                    },
                    "instruction": {
                        "type": "string",
                        "description": (
                            "Natural language instruction for the coding agent. "
                            "Be specific and clear about what you want the agent to do. "
                            "Example: 'Create a new function called calculate_sum that adds two numbers'"
                        ),
                    },
                },
                "required": ["instruction"],
            },
        },
    },
)
def run_coder(fnames: list[str] | None = None, instruction: str = "") -> str:
    """
    Execute a coding task using aider.

    Args:
        fnames: Optional list of file paths to include in the coding session
        instruction: Natural language instruction for the coding agent

    Returns:
        Result message from the coding agent
    """
    try:
        # Create the model with configuration from constants
        model = Model(constant.AIDER_MODEL, verbose=constant.AIDER_VERBOSE)
        model.set_reasoning_effort(constant.AIDER_REASONING_EFFORT)

        # Create a coder object with the specified files (or empty list if None)
        coder = Coder.create(
            main_model=model,
            fnames=fnames or [],
            io=InputOutput(yes=True),
            use_git=constant.AIDER_GIT,
            auto_commits=constant.AIDER_AUTO_COMMITS,
            dirty_commits=constant.AIDER_DIRTY_COMMITS,
        )

        # Redirect stdout to capture and discard aider's logging output
        stdout_capture = io.StringIO()
        with redirect_stdout(stdout_capture):
            # Execute the instruction
            result = coder.run(instruction)

        # Discard the captured stdout (aider's logs)
        # Only return the actual result
        return f"Coding task completed. Result: {result}"

    except Exception as e:
        return f"Error executing coding task: {str(e)}"
