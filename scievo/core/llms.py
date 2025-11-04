from __future__ import annotations

from threading import RLock
from typing import Callable

from litellm.types.utils import Usage

from ..tools import ToolRegistry
from .constant import __GRAPH_STATE_NAME__
from .types import Message


def function_to_json_schema(func_or_name: Callable | str) -> dict:
    if isinstance(func_or_name, str):
        tool = ToolRegistry.instance().tools[func_or_name]
        return tool.json_schema
    elif callable(func_or_name):
        tool = ToolRegistry.instance().tools[func_or_name.__name__]
        return tool.json_schema
    else:
        raise ValueError("func must be a string or a callable")


class ModelRegistry:
    _instance: ModelRegistry | None = None
    _lock: RLock = RLock()

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.models = {}

    @classmethod
    def instance(cls) -> ModelRegistry:
        return cls()

    @classmethod
    def register(
        cls,
        name: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        cls.instance().models[name] = {
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
        agent_sender: str | None = None,
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

        model_params: dict = cls.instance().models[name]
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
        msg.llm_sender = name
        msg.agent_sender = agent_sender
        msg.completion_tokens = usage.completion_tokens
        msg.prompt_tokens = usage.prompt_tokens
        return msg
