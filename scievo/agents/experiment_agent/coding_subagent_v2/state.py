import os
import uuid
from typing import Optional

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from pydantic import PrivateAttr

from scievo.core.code_env import LocalEnv
from scievo.core.plan import PlanState
from scievo.core.types import HistoryState, ToolsetState


class CodingAgentState(ToolsetState, PlanState, HistoryState):
    """State of the Coding Subagent V2.

    This agent follows the plan-and-execute paradigm for coding tasks.
    It integrates with OpenHands SDK for external code manipulation.

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

    # Whether the agent has finished all plans
    talk_mode: bool = False

    # Critic feedback from the last plan step
    critic_feedback: str = ""

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
            agent = Agent(
                llm=llm,
                tools=tools,
                system_prompt_kwargs={"cli_mode": True},
                condenser=LLMSummarizingCondenser(
                    llm=llm.model_copy(update={"usage_id": "condenser"}),
                    max_size=48,
                    keep_first=4,
                ),
            )
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
