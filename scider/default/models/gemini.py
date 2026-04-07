import logging
import os
import sys

from scider.default.models.yaml_loader import register_from_yaml

logger = logging.getLogger(__name__)


def _resolve_keys(
    gemini_key: str | None = None,
) -> dict[str, str | None]:
    """Resolve API keys from arguments or environment variables."""
    if gemini_key is None:
        gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key is None:
        logger.error("GEMINI_API_KEY is required but not provided.")
        sys.exit(1)

    return {"gemini": gemini_key}


def register_gemini_low_medium_models(
    reasoning: str = "low",
    gemini_key: str | None = None,
) -> None:
    """Register Gemini low and medium cost models from YAML config."""
    keys = _resolve_keys(gemini_key)
    register_from_yaml("gemini_low_medium.yaml", api_keys=keys)


def register_gemini_medium_high_models(
    reasoning: str = "low",
    gemini_key: str | None = None,
) -> None:
    """Register Gemini medium and high cost models from YAML config."""
    keys = _resolve_keys(gemini_key)
    register_from_yaml("gemini_medium_high.yaml", api_keys=keys)


def register_gemini3_medium_high_models(
    reasoning: str = "low",
    gemini_key: str | None = None,
) -> None:
    """Register Gemini 3 medium and high cost models from YAML config."""
    keys = _resolve_keys(gemini_key)
    register_from_yaml("gemini_medium_high.yaml", api_keys=keys)
