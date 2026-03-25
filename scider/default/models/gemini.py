import logging
import os
import sys

from scider.default.models.yaml_loader import register_from_yaml

logger = logging.getLogger(__name__)


def _resolve_keys(
    gemini_key: str | None = None,
    openai_key: str | None = None,
) -> dict[str, str | None]:
    """Resolve API keys from arguments or environment variables."""
    if gemini_key is None:
        gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key is None:
        logger.error("GEMINI_API_KEY is required but not provided.")
        sys.exit(1)

    if openai_key is None:
        openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        logger.warning("OPENAI_API_KEY not found. The 'embed' model will not be registered.")

    return {"gemini": gemini_key, "openai": openai_key}


def register_gemini_low_medium_models(
    reasoning: str = "low",
    gemini_key: str | None = None,
    openai_key: str | None = None,
) -> None:
    """Register Gemini low and medium cost models from YAML config."""
    keys = _resolve_keys(gemini_key, openai_key)
    register_from_yaml("gemini_low_medium.yaml", api_keys=keys)


def register_gemini_medium_high_models(
    reasoning: str = "low",
    gemini_key: str | None = None,
    openai_key: str | None = None,
) -> None:
    """Register Gemini medium and high cost models from YAML config."""
    keys = _resolve_keys(gemini_key, openai_key)
    register_from_yaml("gemini_medium_high.yaml", api_keys=keys)


def register_gemini3_medium_high_models(
    reasoning: str = "low",
    gemini_key: str | None = None,
    openai_key: str | None = None,
) -> None:
    """Register Gemini 3 medium and high cost models from YAML config."""
    keys = _resolve_keys(gemini_key, openai_key)
    # Reuse gemini_medium_high config — create a separate yaml if needed
    register_from_yaml("gemini_medium_high.yaml", api_keys=keys)
