from typing import Callable

from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message

# Third-party imports
from pydantic import BaseModel

AgentFunction = Callable[[], "str | Agent | dict"]


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: str | Callable[[dict], str] = "You are a helpful agent."
    functions: list[AgentFunction] = []
    tool_choice: str | None = None
    parallel_tool_calls: bool = False


class Response(BaseModel):
    messages: list[Message | dict] = []
    agent: Agent | None = None
    ctx_vars: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Agent | None = None
    ctx_vars: dict = {}
    image: str | None = None  # base64 encoded image
