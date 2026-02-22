"""
This is the new version of the coder tool, which uses aider as a command line tool.
"""

import re
import shutil
import subprocess
import tempfile

from loguru import logger

from .registry import register_tool, register_toolset_desc

register_toolset_desc(
    "coder",
    "Coder toolset. This toolset allows you to call an AI coding agent using natural language instructions. "
    "The agent can modify, create, or edit code files based on your instructions. "
    "But you have to provide the details, like the file structure, the file names, the file paths, the file content, etc."
    "Note: The coding agent has no memory of previous interactions and operates independently on each call. And it will not chat and exit the session after it's done.",
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
                "And the coding agent will not chat and exit the session after it's done."
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

    # Use OS command line to call `aider` and pass the instruction via stdin
    logger.debug("Running aider with fnames: \n{}\n\ninstruction: \n{}", fnames, instruction)
    try:
        if not instruction.strip():
            return "Error: instruction must be a non-empty string."
        aider_path = shutil.which("aider")
        if not aider_path:
            err_text = "Error: 'aider' executable not found in PATH. Please install aider and ensure it is available."
            logger.error(err_text)
            return err_text

        # create temp file to store the instruction
        with tempfile.NamedTemporaryFile(mode="w") as temp_file:
            temp_file.write(instruction)
            temp_file.flush()

            temp_file_path = temp_file.name

            cmd = [aider_path, "--message-file", temp_file_path, "--yes", "--exit"]

            if fnames:
                cmd.extend(fnames)

            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
            )

            if result.returncode != 0:
                return "Error executing coding task: " + (
                    result.stderr.strip() or f"Non-zero exit status {result.returncode}"
                )

            output_text = result.stdout.strip()
            output_text = _parse_aider_output(output_text)

            return output_text or "Coding task completed."

    except Exception as e:
        return f"Error executing coding task: {str(e)}"


def _parse_aider_output(output_text: str) -> str:
    """Parse aider output to extract the result message."""
    lines = output_text.splitlines()
    result = ""
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Repo-map:"):
            result = "\n".join(lines[i + 1 :])
            break
    else:
        result = "\n".join(lines)

    result = result.strip()

    # remove all the color codes in the results
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    result = ansi_escape.sub("", result)

    return result
