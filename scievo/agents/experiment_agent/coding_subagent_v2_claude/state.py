from scievo.core.code_env import LocalEnv
from scievo.core.plan import PlanState
from scievo.core.types import HistoryState, ToolsetState


class CodingAgentState(ToolsetState, PlanState, HistoryState):
    """State of the Coding Subagent V2.

    This agent follows the plan-and-execute paradigm for coding tasks.
    It integrates with Claude Agent SDK for external code manipulation.

    Note: No RBankState - memory extraction is not used in this agent.
    """

    # Summary of the data from data agent, providing background info for the coding task (input)
    data_summary: str

    # User's coding task description (input, optional)
    user_query: str | None = None

    # Local environment for the agent (input)
    workspace: LocalEnv

    # Whether the agent has finished all plans
    talk_mode: bool = False

    # Critic feedback from the last plan step
    critic_feedback: str = ""

    # Output summary (output)
    output_summary: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure the claude_agent_sdk toolset is included initially
        if "claude_agent_sdk" not in self.toolsets:
            self.toolsets.append("claude_agent_sdk")
