"""[Deprecated] Thin shims — use ``register_preset("gemini/medium_high")`` instead."""

import logging
import os
import sys

from scider.default.models.catalog import ModelCatalog, register_preset

logger = logging.getLogger(__name__)


def _ensure_gemini_key(gemini_key: str | None) -> None:
    if gemini_key is not None:
        os.environ.setdefault("GEMINI_API_KEY", gemini_key)
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY is required but not provided.")
        sys.exit(1)


def register_gemini_low_medium_models(*, gemini_key: str | None = None) -> None:
    _ensure_gemini_key(gemini_key)
    ModelCatalog.load()
    register_preset("gemini/low_medium")


def register_gemini_medium_high_models(*, gemini_key: str | None = None) -> None:
    _ensure_gemini_key(gemini_key)
    ModelCatalog.load()
    register_preset("gemini/medium_high")


def register_gemini3_medium_high_models(*, gemini_key: str | None = None) -> None:
    _ensure_gemini_key(gemini_key)
    ModelCatalog.load()
    register_preset("gemini/gemini3_medium_high")
