import logging
import os
import sys

from scider.default.models.yaml_loader import register_from_yaml

logger = logging.getLogger(__name__)


def _resolve_key(openai_key: str | None = None) -> dict[str, str | None]:
    """Resolve OpenAI API key from argument or environment variable."""
    if openai_key is None:
        openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        logger.error("OPENAI_API_KEY is required but not provided.")
        sys.exit(1)
    return {"openai": openai_key}


def register_gpt_low_medium_models(
    reasoning: str = "low",  # noqa: ARG001
    openai_key: str | None = None,
) -> None:
    """Register GPT low and medium cost models from YAML config."""
    keys = _resolve_key(openai_key)
    register_from_yaml("gpt_low_medium.yaml", api_keys=keys)


def register_gpt_medium_high_models(
    reasoning: str = "low",  # noqa: ARG001
    openai_key: str | None = None,
) -> None:
    """Register GPT medium and high cost models from YAML config."""
    keys = _resolve_key(openai_key)
    register_from_yaml("gpt_medium_high.yaml", api_keys=keys)
