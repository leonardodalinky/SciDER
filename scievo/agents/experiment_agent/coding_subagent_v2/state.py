import os
import uuid
from typing import Optional

# Now import openhands (will use local version if paths were added)
from openhands.sdk import LLM, Agent, AgentContext, Conversation, Tool

# Setup openhands paths first (must be before any openhands imports)
from scievo.core import openhands_import  # noqa: F401

# Try to import LLMSummarizingCondenser if available
try:
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
except ImportError:
    # Fallback: LLMSummarizingCondenser is not available in this version
    LLMSummarizingCondenser = None
from openhands.sdk.context.skills import Skill
from pydantic import PrivateAttr

from scievo.core.code_env import LocalEnv
from scievo.core.types import HistoryState, ToolsetState


class CodingAgentState(ToolsetState, HistoryState):
    """State of the Coding Subagent V2.

    This agent delegates coding tasks to OpenHands SDK which has its own
    internal planning mechanism. No external planning is needed.

    Note: No RBankState - memory extraction is not used in this agent.
    """

    # Summary of the data from data agent, providing background info for the coding task (input)
    data_summary: str

    # User's coding task description (input, optional)
    user_query: str | None = None

    # Local environment for the agent (input)
    workspace: LocalEnv

    # OpenHands Conversation object - persists throughout the execution (private)
    # This maintains the conversation history with the external coding agent
    _openhands_conversation: Optional["Conversation"] = PrivateAttr(default=None)

    # Output summary (output)
    output_summary: str | None = None

    def __init__(self, _openhands_conversation: Optional["Conversation"] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a default empty conversation if not provided
        if _openhands_conversation is None:
            api_key = os.getenv("OPENHANDS_API_KEY") or os.getenv("LLM_API_KEY")
            model = os.getenv("OPENHANDS_MODEL", "anthropic/claude-sonnet-4-5-20250929")

            llm = LLM(
                model=model,
                api_key=api_key,
                usage_id=f"openhands-coding-agent-{uuid.uuid4().hex[:8]}",
            )

            from openhands.tools.file_editor import FileEditorTool
            from openhands.tools.glob import GlobTool
            from openhands.tools.grep import GrepTool
            from openhands.tools.task_tracker import TaskTrackerTool
            from openhands.tools.terminal import TerminalTool

            tools = [
                Tool(name=FileEditorTool.name),
                Tool(name=TaskTrackerTool.name),
                Tool(name=TerminalTool.name),
                Tool(name=GlobTool.name),
                Tool(name=GrepTool.name),
            ]
            agent_context = AgentContext(
                skills=[
                    Skill(
                        name="Python Dependency Management by `uv` instead of `pip`",
                        content="For Python projects: Always prioritize using 'uv' for managing dependencies and virtual environments. "
                        "Avoid using 'pip' or other package managers that directly affect the native system environment. "
                        "Use 'uv sync' to install dependencies from lock files, 'uv venv' to create isolated environments, "
                        "and 'uv add' to add new packages. This approach ensures project isolation and reproducibility. "
                        "This skill applies only to Python projects.",
                    ),
                    Skill(
                        name="Avoid Long Time Operations",
                        content="Avoid using tools or commands that may lead to long wait times or blocking operations, "
                        "such as training the model directly within this environment. ",
                    ),
                    Skill(
                        name="File Operations Should Use Absolute Paths as Much as Possible",
                        content="When using the File Editor tool and other file-related tools, always refer to files using their absolute paths. "
                        "This ensures that file operations are unambiguous and correctly targeted within the workspace. ",
                    ),
                ],
                system_message_suffix="""\
You are operating in CLI mode, so all file paths should be absolute paths as much as possible.
Besides, try to avoid long time operations that may block the process, e.g., training the deep learning model directly.
""",
            )
            # Build agent kwargs - only include condenser if available
            agent_kwargs = {
                "llm": llm,
                "tools": tools,
                "system_prompt_kwargs": {"cli_mode": True},
                "agent_context": agent_context,
            }
            # Add condenser only if LLMSummarizingCondenser is available
            if LLMSummarizingCondenser is not None:
                agent_kwargs["condenser"] = LLMSummarizingCondenser(
                    llm=llm.model_copy(update={"usage_id": "condenser"}),
                    max_size=48,
                    keep_first=4,
                )

            agent = Agent(**agent_kwargs)
            _openhands_conversation = Conversation(
                agent=agent, workspace=self.workspace.working_dir
            )

        self._openhands_conversation = _openhands_conversation

        # Ensure the openhands toolset is included initially
        self.toolsets.append("openhands")

    @property
    def openhands_conversation(self) -> "Conversation":
        """Get the OpenHands Conversation object."""
        return self._openhands_conversation
