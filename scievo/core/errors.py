"""
Error handling utilities.
"""


def sprint_chained_exception(e: Exception) -> str:
    ret = repr(e)
    while e.__cause__:
        ret += f"\n  <-  {repr(e.__cause__)}"
        e = e.__cause__
    return ret


class AgentError(Exception):
    def __init__(self, *args, agent_name: str = None):
        super().__init__(*args)
        self.agent_name = agent_name

    def __repr__(self):
        if self.agent_name:
            return f"AgentError({self.agent_name}): {self.args}"
        return f"AgentError: {self.args}"

    def sprint(self):
        return sprint_chained_exception(self)
