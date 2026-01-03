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


def _normalize_toon_content(toon_content: str) -> str:
    """
    Normalize TOON content to fix common format issues.

    Fixes:
    1. Illegal key syntax like `authors[6]: value1,value2,...` -> `authors: [value1, value2, ...]`
    2. Handles comma-separated values in indexed keys
    3. Preserves proper TOON format for other lines
    """
    # Check if normalization is needed
    if not re.search(r"\w+\[\d+\]\s*:", toon_content):
        # No illegal syntax found, return as-is
        return toon_content

    lines = []

    for line in toon_content.splitlines():
        # Match illegal indexed key syntax: key[number]: value
        # Example: authors[6]: Fengyu She,Nan Wang,Hongfei Wu,...
        indexed_key_match = re.match(r"^(\w+)\[(\d+)\]\s*:\s*(.+)$", line)
        if indexed_key_match:
            key, index, value = indexed_key_match.groups()

            # Split comma-separated values
            # Handle both quoted and unquoted values
            values = []
            for v in value.split(","):
                v = v.strip()
                # Remove quotes if present
                if (v.startswith('"') and v.endswith('"')) or (
                    v.startswith("'") and v.endswith("'")
                ):
                    v = v[1:-1]
                if v:  # Only add non-empty values
                    values.append(v)

            # Convert to proper TOON list format
            if values:
                # Use YAML-style list format for better compatibility
                formatted_values = ", ".join(f'"{v}"' for v in values)
                lines.append(f"{key}: [{formatted_values}]")
            else:
                lines.append(f"{key}: []")
        else:
            # Regular line - preserve as-is
            lines.append(line)

    return "\n".join(lines)


def unwrap_dict_from_toon(toon_str: str) -> dict:
    """Parse a toon-formatted string back to a dictionary."""
    if isinstance(toon_str, dict):
        return toon_str

    if not isinstance(toon_str, str):
        raise TypeError(f"Expected str or dict, got {type(toon_str)}")
    match = re.search(
        r"```toon\s*\n(.*?)\n```",
        toon_str,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:
        toon_content = match.group(1).strip()
    else:
        toon_content = toon_str.strip()

    if ":" not in toon_content:
        raise ValueError(
            "Invalid TOON content: no ':' found. " "Likely code block extraction failed."
        )

    # Normalize TOON content to fix common format issues
    # (e.g., illegal indexed key syntax like authors[6]: value)
    toon_content = _normalize_toon_content(toon_content)

    try:
        if hasattr(toon, "decode"):
            return toon.decode(toon_content)
        if hasattr(toon, "loads"):
            return toon.loads(toon_content)
        if hasattr(toon, "parse"):
            return toon.parse(toon_content)

        raise RuntimeError("toon library has no decode / loads / parse method")

    except Exception as e:
        logger = __import__("loguru", fromlist=["logger"]).logger
        logger.debug(f"Full TOON content: {toon_content}")
        raise ValueError(f"Failed to decode TOON: {e}") from e


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


def hello_world() -> None:
    """Print 'Hello, World!' to the console."""
    print("Hello, World!")
