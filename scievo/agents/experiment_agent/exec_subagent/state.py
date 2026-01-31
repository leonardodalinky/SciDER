from scievo.core.code_env import LocalEnv
from scievo.core.exec.manager import SessionManager
from scievo.core.exec.pty_session import LocalShellSession
from scievo.core.types import ExecState, HistoryState, ToolsetState


class ExecAgentState(ExecState, ToolsetState, HistoryState):
    """State of the Experiment Execution Agent.

    This agent is responsible for executing experiments in local shell sessions.
    It combines:
    - ToolsetState: for managing available toolsets
    - HistoryState: for managing conversation history
    """

    # The natural language query describing what experiment to run (input)
    user_query: str

    # Current working directory where experiments are executed (input)
    workspace: LocalEnv

    # Coding summaries from previous revisions (input, optional)
    # Used to provide context about code changes made in each revision
    coding_summaries: list[str] | None = None

    # Raw summary of the experiment execution, try to use `execution_summary_dict` instead (output)
    execution_summary: str = ""

    # Structured summary of the experiment execution (output)
    # Should be:
    # ```json
    # {
    #     "status": "Success" or "Failed",
    #     "commands_executed": ["command 1", "command 2", ...],
    #     "key_outputs": "Highlight any important output or results",
    #     "errors_issues": "Note any errors or issues encountered, or 'None' if successful"
    # }
    # ```
    execution_summary_dict: dict = {}

    # Number of monitoring attempts for the current running command (internal use)
    monitoring_attempts: int = 0

    # Whether to force monitoring in the next step (internal use)
    is_monitor_mode: bool = False

    # Intermediate states
    intermediate_state: list[dict] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.session_id is None:
            s = LocalShellSession(cwd=self.workspace.working_dir)
            # Store session ID instead of the session instance
            self.session_id = s.session_id
        # add initial toolset
        self.toolsets.append("exec")
