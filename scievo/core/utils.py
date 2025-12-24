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


def unwrap_dict_from_toon(toon_str: str) -> dict:
    """Parse a toon-formatted string back to a dictionary."""
    from loguru import logger

    # Handle case where input is already a dict (shouldn't happen, but be defensive)
    if isinstance(toon_str, dict):
        return toon_str

    # Extract toon content from code block if present
    toon_match = re.search(
        r"(?:```\s*)?(?:toon\s*)?(.*)(?:```)?", toon_str, flags=re.DOTALL | re.IGNORECASE
    )
    if not toon_match:
        # If no code block found, try treating the whole string as toon content
        toon_content = toon_str.strip()
    else:
        toon_content = toon_match.group(1).strip()

    # Log the toon content for debugging (first 500 chars)
    logger.debug("Attempting to decode TOON content (preview): {}", toon_content[:500])

    # Try toon.decode() - this is the standard method
    try:
        result = toon.decode(toon_content)
        logger.debug("Successfully decoded TOON")
        return result
    except AttributeError:
        # If decode doesn't exist, try other common method names
        logger.debug("toon.decode() not found, trying alternative methods")
        if hasattr(toon, "loads"):
            return toon.loads(toon_content)
        elif hasattr(toon, "parse"):
            return toon.parse(toon_content)
        else:
            raise ValueError(
                f"toon library does not have decode, loads, or parse methods. "
                f"Available: {[attr for attr in dir(toon) if not attr.startswith('_')]}"
            )
    except Exception as e:
        # Log the full error and content for debugging
        logger.error("TOON decode error: {}. Content length: {}", e, len(toon_content))
        logger.debug("Full TOON content: {}", toon_content)
        # Re-raise with more context
        raise ValueError(
            f"Failed to decode TOON: {e}. Content preview: {toon_content[:200]}"
        ) from e


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
