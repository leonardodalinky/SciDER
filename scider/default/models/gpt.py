"""[Deprecated] Thin shims — use ``register_preset("gpt/medium_high")`` instead."""

import logging
import os
import sys

from scider.default.models.catalog import ModelCatalog, register_preset

logger = logging.getLogger(__name__)


def _ensure_openai_key(openai_key: str | None) -> None:
    if openai_key is not None:
        os.environ.setdefault("OPENAI_API_KEY", openai_key)
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is required but not provided.")
        sys.exit(1)


def register_gpt_low_medium_models(*, openai_key: str | None = None) -> None:
    _ensure_openai_key(openai_key)
    ModelCatalog.load()
    register_preset("gpt/low_medium")


def register_gpt_medium_high_models(*, openai_key: str | None = None) -> None:
    _ensure_openai_key(openai_key)
    ModelCatalog.load()
    register_preset("gpt/medium_high")
