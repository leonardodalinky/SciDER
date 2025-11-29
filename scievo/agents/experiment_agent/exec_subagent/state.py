from scievo.core.types import ExecState, HistoryState, ToolsetState


class ExecAgentState(ToolsetState, HistoryState, ExecState):
    """State of the Experiment Execution Agent.

    This agent is responsible for executing experiments in local shell sessions.
    It combines:
    - ToolsetState: for managing available toolsets
    - HistoryState: for managing conversation history
    - ExecState: for managing the shell session
    """

    # The natural language query describing what experiment to run
    user_query: str

    # Current working directory where experiments are executed
    working_dir: str

    # Summary of the experiment execution
    execution_summary: str = ""

    # Structured summary of the experiment execution (parsed from JSON)
    execution_summary_dict: dict = {}

    # Number of monitoring attempts for the current running command
    monitoring_attempts: int = 0

    # Whether to force monitoring in the next step
    force_monitoring: bool = False

    @property
    def round(self) -> int:
        return len(self.node_history) - 1
