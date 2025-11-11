from __future__ import annotations

from threading import RLock
from typing import Callable

import litellm
from functional import seq
from litellm.types.utils import Usage
from loguru import logger

from ..tools import ToolRegistry
from .constant import __AGENT_STATE_NAME__
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
        logger.debug("Registered model: {}", name)

    def get_model_params(self, name: str) -> dict:
        if name not in self.models:
            raise ValueError(f"Model `{name}` not found")
        return self.models[name]

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
    ) -> Message:
        tools_json_schemas = [function_to_json_schema(tool) for tool in tools] if tools else []
        for schema in tools_json_schemas:
            params = schema["function"]["parameters"]
            params["properties"].pop(__AGENT_STATE_NAME__, None)
            if __AGENT_STATE_NAME__ in params["required"]:
                params["required"].remove(__AGENT_STATE_NAME__)

        messages = [Message(role="system", content=system_prompt)] + history

        model_params: dict = cls.instance().get_model_params(name)
        llm_model: str = model_params["model"]

        if llm_model.startswith("gpt-5"):
            logger.trace("Using GPT-5 Response API for model: {}", name)

            from litellm import responses as ll_responses

            # response API has different tool schema
            res_tools = (
                seq(tools_json_schemas)
                .map(lambda schema: {"type": "function", **schema["function"]})
                .to_list()
            )

            input = []
            for msg in messages:
                input.extend(msg.to_ll_response_message())

            params = model_params.copy()
            params.update(kwargs)
            params.update(
                {
                    "input": input,
                    "tools": res_tools,
                    "tool_choice": tool_choice,
                }
            )

            response = ll_responses(**params)
            # print(response.model_dump_json(indent=2))
            logger.trace("GPT-5 Response API response: {}", response.model_dump_json(indent=2))

            ## tool calls
            tool_calls = (
                seq(response.output)
                .filter(lambda c: c.type == "function_call")
                .map(
                    lambda c: litellm.ChatCompletionMessageToolCall(
                        id=c.id,
                        function=litellm.Function(
                            name=c.name,
                            arguments=c.arguments,
                        ),
                    )
                )
                .to_list()
            )
            ## message content
            msg_content = (
                seq(response.output)
                .filter(lambda c: c.type == "message")
                .map(lambda c: c.content)
                .to_list()
            )
            if len(msg_content) == 0:
                msg_content = None
            else:
                msg_content = msg_content[0][0].text
            ## usage
            usage = response.usage  # type: ignore
            ## reasoning content
            reasoning_msg_block = (
                seq(response.output)
                .filter(lambda c: c.type == "reasoning")
                .filter(lambda c: len(c.summary) > 0)
                .head_option(no_wrap=True)
            )
            if reasoning_msg_block:
                reasoning_summaries = (
                    seq(reasoning_msg_block.summary)
                    .filter(lambda s: s.type == "summary_text")
                    .map(lambda s: s.text)
                    .to_list()
                )
                reasoning_text = "\n".join(reasoning_summaries)
            else:
                reasoning_text = None

            ## construct message
            msg: Message = Message(
                role="assistant",
                content=msg_content,
                tool_calls=tool_calls,
                llm_sender=name,
                agent_sender=agent_sender,
                completion_tokens=usage.output_tokens,
                prompt_tokens=usage.input_tokens,
                reasoning_content=reasoning_text,
            )
            return msg
        else:
            from litellm import completion as ll_completion

            params = model_params.copy()
            params.update(kwargs)
            params.update(
                {
                    "messages": (seq(messages).map(lambda msg: msg.to_ll_message()).to_list()),
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

    @classmethod
    def embedding(cls, name: str, texts: list[str], **kwargs) -> list[list[float]]:
        """Returns a list of embeddings for the given texts."""
        from litellm import embedding as ll_embedding

        model_params: dict = cls.instance().models[name]
        params = model_params.copy()
        params.update(kwargs)
        params.update({"input": texts})

        response = ll_embedding(**params)
        return [d["embedding"] for d in response.data]
