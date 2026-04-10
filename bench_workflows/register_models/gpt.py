"""GPT model registration for benchmarks — delegates to unified catalog."""

from scider.default.models import ModelCatalog, register_preset


def register_gpt_low_medium_models(reasoning: str = "low"):  # noqa: ARG001
    ModelCatalog.load()
    register_preset("gpt/low_medium")


def register_gpt_medium_high_models(reasoning: str = "low"):  # noqa: ARG001
    ModelCatalog.load()
    register_preset("gpt/medium_high")
