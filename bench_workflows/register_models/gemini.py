"""Gemini model registration for benchmarks — delegates to unified catalog."""

from scider.default.models import ModelCatalog, register_preset


def register_gemini_low_medium_models(reasoning: str = "low"):  # noqa: ARG001
    ModelCatalog.load()
    register_preset("gemini/low_medium")


def register_gemini_medium_high_models(reasoning: str = "low"):  # noqa: ARG001
    ModelCatalog.load()
    register_preset("gemini/medium_high")


def register_gemini3_medium_high_models(reasoning: str = "low"):  # noqa: ARG001
    ModelCatalog.load()
    register_preset("gemini/gemini3_medium_high")
