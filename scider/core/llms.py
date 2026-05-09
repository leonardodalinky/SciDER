from __future__ import annotations

from threading import RLock
from time import sleep
from typing import Callable

import litellm
from functional import seq
from litellm import RateLimitError, ServiceUnavailableError
from litellm.types.utils import Usage
from loguru import logger

# Anthropic requires `tools=` whenever the message history contains assistant
# tool_calls. Our generate_summary_node calls completion() without tools but
# the history (full agent loop) does contain tool_calls. Setting modify_params
# tells litellm to auto-inject a dummy tool placeholder so the request goes
# through. Safe for other providers (no-op).
litellm.modify_params = True

from ..tools import ToolRegistry
from .constant import __AGENT_STATE_NAME__
from .types import Message


class ZeroChoiceError(Exception):
    """Raised when LLM completion returns zero choices."""

    pass


def _detect_provider(model_id: str) -> str:
    """Infer the LLM provider family from a litellm model id.

    Returns one of: ``"anthropic"``, ``"gemini"``, ``"openai"``, ``"other"``.
    """
    if not model_id:
        return "other"
    mid = model_id.lower()
    if mid.startswith(("gemini/", "google/", "vertex_ai/")):
        return "gemini"
    if mid.startswith(("anthropic/", "claude-")) or "claude" in mid:
        return "anthropic"
    if mid.startswith(("openai/", "gpt-", "o1-", "o3-", "o4-")) or mid in {"o1", "o3", "o4"}:
        return "openai"
    return "other"


def _serialize_messages_for_provider(
    messages: "list[Message]",
    litellm_model_id: str,
) -> list:
    """Serialize Message objects for the LLM Completion API, injecting
    image and document attachments on tool results according to what each
    provider allows.

    - Anthropic / Gemini: image and document blocks go inline on the tool
      message itself as a multi-part ``content`` list. litellm maps
      ``image_url`` / ``file`` blocks to each provider's native format.
    - OpenAI (and unknown): Chat Completions tool messages only accept a
      text string, so the tool result stays text-only and a synthetic user
      message carrying the image is appended right after. PDFs are NOT
      supported on Chat Completions (OpenAI requires Responses API for
      PDF), so they fall back to text metadata only.

    Messages without any attachments go through the existing
    ``to_ll_message()`` path unchanged.
    """
    provider = _detect_provider(litellm_model_id)
    out: list = []
    for msg in messages:
        imgs = getattr(msg, "tool_result_images", None)
        docs = getattr(msg, "tool_result_documents", None)
        if msg.role == "tool" and (imgs or docs):
            if provider in ("anthropic", "gemini"):
                content_blocks: list[dict] = [{"type": "text", "text": msg.content or ""}]
                for img in imgs or []:
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img['media_type']};base64,{img['data']}"},
                        }
                    )
                for doc in docs or []:
                    # litellm maps "file" to Anthropic ``document`` blocks and
                    # Gemini ``inlineData`` parts with application/pdf.
                    content_blocks.append(
                        {
                            "type": "file",
                            "file": {
                                "file_data": f"data:{doc['media_type']};base64,{doc['data']}",
                                **({"filename": doc["filename"]} if doc.get("filename") else {}),
                            },
                        }
                    )
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": content_blocks,
                    }
                )
            else:
                # OpenAI / other: tool messages are text-only. Images go as a
                # synthetic user message; PDFs cannot be attached on Chat
                # Completions so we surface a note in the text.
                out.append(msg.to_ll_message())
                user_blocks: list[dict] = []
                if imgs:
                    user_blocks.append(
                        {
                            "type": "text",
                            "text": f"[Image(s) from Read tool call {msg.tool_call_id}]",
                        }
                    )
                    for img in imgs:
                        user_blocks.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{img['media_type']};base64,{img['data']}"
                                },
                            }
                        )
                if docs:
                    # Chat Completions does not accept PDF file blocks — note
                    # the omission so the model isn't confused by a missing
                    # document it expected to see.
                    names = ", ".join(d.get("filename") or "document" for d in docs)
                    user_blocks.append(
                        {
                            "type": "text",
                            "text": (
                                f"[PDF document(s) ({names}) from Read tool call "
                                f"{msg.tool_call_id} are not viewable on this API "
                                "path — text metadata only.]"
                            ),
                        }
                    )
                if user_blocks:
                    out.append({"role": "user", "content": user_blocks})
        else:
            out.append(msg.to_ll_message())
    return out


def _serialize_messages_for_responses_api(messages: "list[Message]") -> list:
    """Serialize Message objects for OpenAI's Responses API input list.

    Responses API only allows images / files in ``role: user`` message
    blocks, so a tool result with attachments is emitted as the usual
    ``function_call_output`` followed by a synthetic user message carrying
    ``input_image`` (images) and/or ``input_file`` (PDFs) blocks.
    """
    out: list = []
    for msg in messages:
        imgs = getattr(msg, "tool_result_images", None)
        docs = getattr(msg, "tool_result_documents", None)
        out.extend(msg.to_ll_response_message())
        if msg.role == "tool" and (imgs or docs):
            preamble_bits: list[str] = []
            if imgs:
                preamble_bits.append(f"{len(imgs)} image(s)")
            if docs:
                preamble_bits.append(f"{len(docs)} PDF(s)")
            content: list[dict] = [
                {
                    "type": "input_text",
                    "text": (
                        f"[{' and '.join(preamble_bits)} from Read tool call "
                        f"{msg.tool_call_id}]"
                    ),
                }
            ]
            for img in imgs or []:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{img['media_type']};base64,{img['data']}",
                    }
                )
            for doc in docs or []:
                entry: dict = {
                    "type": "input_file",
                    "file_data": f"data:{doc['media_type']};base64,{doc['data']}",
                }
                if doc.get("filename"):
                    entry["filename"] = doc["filename"]
                content.append(entry)
            out.append({"type": "message", "role": "user", "content": content})
    return out


def function_to_json_schema(func_or_name: Callable | str) -> dict:
    if isinstance(func_or_name, str):
        tool = ToolRegistry.instance().tools[func_or_name]
        return tool.get_json_schema()
    elif callable(func_or_name):
        tool = ToolRegistry.instance().tools[func_or_name.__name__]
        return tool.get_json_schema()
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

    # Retry config: exponential backoff with jitter
    _MAX_RETRIES = 10
    _BASE_DELAY_S = 1.0  # 1s base
    _MAX_DELAY_S = 120.0  # 2 min cap

    @staticmethod
    def _backoff_delay(attempt: int, base: float = 1.0, cap: float = 120.0) -> float:
        """Exponential backoff with ±25% jitter, capped at `cap` seconds."""
        import random

        delay = min(base * (2**attempt), cap)
        jitter = delay * 0.25 * (2 * random.random() - 1)  # ±25%
        return max(0.1, delay + jitter)

    @classmethod
    def completion(cls, *args, **kwargs) -> Message:
        """Completion with exponential-backoff retry for transient errors."""
        for attempt in range(cls._MAX_RETRIES):
            try:
                return cls._completion(*args, **kwargs)
            except RateLimitError as e:
                if attempt == cls._MAX_RETRIES - 1:
                    logger.error(f"RateLimitError after {cls._MAX_RETRIES} attempts: {e}")
                    raise
                delay = cls._backoff_delay(attempt, base=cls._BASE_DELAY_S, cap=cls._MAX_DELAY_S)
                logger.warning(
                    f"RateLimitError on attempt {attempt + 1}/{cls._MAX_RETRIES}. "
                    f"Waiting {delay:.1f}s before retry... Error: {e}"
                )
                sleep(delay)
            except ZeroChoiceError as e:
                if attempt == cls._MAX_RETRIES - 1:
                    raise
                delay = cls._backoff_delay(attempt, base=cls._BASE_DELAY_S, cap=cls._MAX_DELAY_S)
                logger.warning(
                    f"ZeroChoiceError on attempt {attempt + 1}/{cls._MAX_RETRIES}. "
                    f"Waiting {delay:.1f}s before retry... Error: {e}"
                )
                sleep(delay)
            except ServiceUnavailableError as e:
                if attempt == cls._MAX_RETRIES - 1:
                    raise
                delay = cls._backoff_delay(attempt, base=cls._BASE_DELAY_S, cap=cls._MAX_DELAY_S)
                logger.warning(
                    f"ServiceUnavailableError on attempt {attempt + 1}/{cls._MAX_RETRIES}. "
                    f"Waiting {delay:.1f}s before retry... Error: {e}"
                )
                sleep(delay)

    @classmethod
    def _completion(
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

            input = _serialize_messages_for_responses_api(messages)

            params = model_params.copy()
            params.update(kwargs)
            params.update({"input": input})
            if res_tools:
                params["tools"] = res_tools
                params["tool_choice"] = tool_choice

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

            # check if both content and tool_calls are empty
            if msg_content is None and len(tool_calls) == 0:
                raise ZeroChoiceError("No content or tool calls returned from response API")

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
            logger.trace("Using completion API for model: {}", name)

            # completion API
            from litellm import completion as ll_completion

            params = model_params.copy()
            params.update(kwargs)
            params.update(
                {
                    "messages": _serialize_messages_for_provider(messages, llm_model),
                }
            )
            # vLLM (and some strict OpenAI-compatible servers) reject an empty
            # tools array — must either provide ≥1 tool or omit the field.
            if tools_json_schemas:
                params["tools"] = tools_json_schemas
                params["tool_choice"] = tool_choice

            response = ll_completion(**params)
            if response.choices is None or len(response.choices) == 0:
                raise ZeroChoiceError("No choices returned from completion API")
            choice = response.choices[0]
            msg: Message = Message.from_ll_message(choice.message)  # type: ignore
            usage: Usage = response.usage  # type: ignore
            msg.llm_sender = name
            msg.agent_sender = agent_sender
            msg.completion_tokens = usage.completion_tokens
            msg.prompt_tokens = usage.prompt_tokens
            msg.finish_reason = getattr(choice, "finish_reason", None)
            return msg
