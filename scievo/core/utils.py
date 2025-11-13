import re
from typing import Type, TypeVar

import toon
from json_repair import repair_json
from pydantic import BaseModel

from .types import Message

T = TypeVar("T", bound=BaseModel)


def wrap_text_with_block(s: str, block_name: str) -> str:
    return f"```{block_name}\n{s}\n```"


def wrap_dict_to_toon(d: dict) -> str:
    s = toon.encode(d)
    if s == "null":
        raise ValueError("Failed to encode dict to TOON")
    return wrap_text_with_block(s, "toon")


def parse_json_from_llm_response(llm_response: str | Message, tgt_type: Type[T]) -> T:
    if isinstance(llm_response, Message):
        text = llm_response.content
    else:
        text = llm_response
    json_match = re.search(
        r"(?:```\s*)?(?:json\s*)?(.*)(?:```)?", text, flags=re.DOTALL | re.IGNORECASE
    )  # must find something, at least return the entire text
    if not json_match:
        raise ValueError("Failed to find JSON in LLM response")
    json_str = json_match.group(1).strip()
    json_str = repair_json(json_str)
    return tgt_type.model_validate_json(json_str)


def parse_markdown_from_llm_response(llm_response: str | Message) -> str:
    if isinstance(llm_response, Message):
        text = llm_response.content
    else:
        text = llm_response
    markdown_match = re.search(
        r"(?:```\s*)?(?:markdown\s*)?(.*)(?:```)?", text, flags=re.DOTALL | re.IGNORECASE
    )  # must find something, at least return the entire text
    if not markdown_match:
        raise ValueError("Failed to find markdown in LLM response")
    markdown_str = markdown_match.group(1).strip()
    return markdown_str


def array_to_bullets(arr: list[str]) -> str:
    return "\n".join([f"- {s}" for s in arr])
