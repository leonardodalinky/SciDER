from typing import Callable

from litellm.types.utils import Usage

from .constant import __GRAPH_STATE_NAME__
from .tools import ToolRegistry
from .types import Message


def function_to_json_schema(func: Callable | str) -> dict:
    if isinstance(func, str):
        tool = ToolRegistry.get_instance().tools[func]
        return tool.json_schema
    elif callable(func):
        tool = ToolRegistry.get_instance().tools[func.__name__]
        return tool.json_schema
    else:
        raise ValueError("func must be a string or a callable")


class ModelRegistry:
    _instance = None

    def __init__(self):
        self.models = {}

    @staticmethod
    def get_instance() -> "ModelRegistry":
        if ModelRegistry._instance is None:
            ModelRegistry._instance = ModelRegistry()
        return ModelRegistry._instance

    @classmethod
    def register(
        cls,
        name: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        cls.get_instance().models[name] = {
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            **kwargs,
        }

    @classmethod
    def completion(
        cls,
        name: str,
        history: list[Message],
        system_prompt: str,
        tools: list | None = None,
        tool_choice: str | None = None,
        **kwargs,
    ):
        from litellm import completion as ll_completion

        tools_json_schemas = [function_to_json_schema(tool) for tool in tools] if tools else None
        if tools_json_schemas:
            for schema in tools_json_schemas:
                params = schema["function"]["parameters"]
                params["properties"].pop(__GRAPH_STATE_NAME__, None)
                if __GRAPH_STATE_NAME__ in params["required"]:
                    params["required"].remove(__GRAPH_STATE_NAME__)

        messages = [{"role": "system", "content": system_prompt}] + history

        model_params: dict = cls.get_instance().models[name]
        params = model_params.copy()
        params.update(kwargs)
        params.update(
            {
                "messages": messages,
                "tools": tools_json_schemas,
                "tool_choice": tool_choice,
            }
        )

        response = ll_completion(**params)
        msg: Message = Message.from_ll_message(response.choices[0].message)  # type: ignore
        usage: Usage = response.usage  # type: ignore
        msg.sender = name
        msg.completion_tokens = usage.completion_tokens
        msg.prompt_tokens = usage.prompt_tokens
        return msg
