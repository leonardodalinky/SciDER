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

    # Current working directory where experiments are executed
    working_dir: str

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # add initial toolset
        self.toolsets.append("exec")
