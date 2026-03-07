import logging
import os
import sys

from scider.core.llms import ModelRegistry

logger = logging.getLogger(__name__)

LOW_COST_MODEL = "gemini/gemini-2.5-flash-lite"
MEDIUM_COST_MODEL = "gemini/gemini-2.5-flash"
HIGH_COST_MODEL = "gemini/gemini-2.5-pro"

MEDIUM_COST_MODEL_2 = "gemini/gemini-3-flash-preview"
HIGH_COST_MODEL_2 = "gemini/gemini-3-pro-preview"


def _resolve_keys(
    gemini_key: str | None = None,
    openai_key: str | None = None,
) -> tuple[str, str | None]:
    """Resolve API keys from arguments or environment variables.

    Returns:
        A tuple of (gemini_key, openai_key). openai_key may be None if not available
        (only a warning is emitted since it's only needed for the embed model).

    Raises:
        SystemExit: If gemini_key is not provided and not found in environment.
    """
    if gemini_key is None:
        gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key is None:
        logger.error(
            "GEMINI_API_KEY is required but not provided via argument or environment variable."
        )
        sys.exit(1)

    if openai_key is None:
        openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        logger.warning(
            "OPENAI_API_KEY not found. The 'embed' model (text-embedding-3-small) "
            "will not be registered. Other models will continue to register normally."
        )

    return gemini_key, openai_key


def register_gemini_low_medium_models(
    reasoning: str = "low",
    gemini_key: str | None = None,
    openai_key: str | None = None,
) -> None:
    """Register Gemini low and medium cost models in the ModelRegistry.

    Args:
        reasoning: Reasoning effort level.
        gemini_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        openai_key: OpenAI API key (for embed model). Falls back to OPENAI_API_KEY env var.
    """
    gk, ok = _resolve_keys(gemini_key, openai_key)

    ModelRegistry.register(
        name="ideation",
        model=MEDIUM_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="paper_search",
        model=LOW_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="metric_search",
        model=LOW_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="data",
        model=LOW_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="plan",
        model=MEDIUM_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="critic",
        model=LOW_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="mem",
        model=LOW_COST_MODEL,
        api_key=gk,
    )

    # NOTE: Use OpenAI embeddings for better performance
    if ok is not None:
        ModelRegistry.register(
            name="embed",
            model="text-embedding-3-small",
            api_key=ok,
        )

    ModelRegistry.register(
        name="history",
        model=LOW_COST_MODEL,
        api_key=gk,
    )

    ModelRegistry.register(
        name="experiment_agent",
        model=MEDIUM_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_coding",
        model=MEDIUM_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_execute",
        model=MEDIUM_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_monitor",
        model=LOW_COST_MODEL,
        api_key=gk,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="experiment_summary",
        model=LOW_COST_MODEL,
        api_key=gk,
        reasoning_effort="low",
    )


def register_gemini_medium_high_models(
    reasoning: str = "low",
    gemini_key: str | None = None,
    openai_key: str | None = None,
) -> None:
    """Register Gemini medium and high cost models in the ModelRegistry.

    Args:
        reasoning: Reasoning effort level.
        gemini_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        openai_key: OpenAI API key (for embed model). Falls back to OPENAI_API_KEY env var.
    """
    gk, ok = _resolve_keys(gemini_key, openai_key)

    ModelRegistry.register(
        name="ideation",
        model=HIGH_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="paper_search",
        model=MEDIUM_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="metric_search",
        model=MEDIUM_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="data",
        model=MEDIUM_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="plan",
        model=HIGH_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="critic",
        model=HIGH_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="mem",
        model=MEDIUM_COST_MODEL,
        api_key=gk,
    )

    if ok is not None:
        ModelRegistry.register(
            name="embed",
            model="text-embedding-3-small",
            api_key=ok,
        )

    ModelRegistry.register(
        name="history",
        model=MEDIUM_COST_MODEL,
        api_key=gk,
    )

    ModelRegistry.register(
        name="experiment_agent",
        model=HIGH_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_coding",
        model=HIGH_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_execute",
        model=HIGH_COST_MODEL,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_monitor",
        model=MEDIUM_COST_MODEL,
        api_key=gk,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="experiment_summary",
        model=HIGH_COST_MODEL,
        api_key=gk,
        reasoning_effort="low",
    )


def register_gemini3_medium_high_models(
    reasoning: str = "low",
    gemini_key: str | None = None,
    openai_key: str | None = None,
) -> None:
    """Register Gemini 3 medium and high cost models in the ModelRegistry.

    Args:
        reasoning: Reasoning effort level.
        gemini_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        openai_key: OpenAI API key (for embed model). Falls back to OPENAI_API_KEY env var.
    """
    gk, ok = _resolve_keys(gemini_key, openai_key)

    ModelRegistry.register(
        name="ideation",
        model=HIGH_COST_MODEL_2,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="paper_search",
        model=MEDIUM_COST_MODEL_2,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="metric_search",
        model=MEDIUM_COST_MODEL_2,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="data",
        model=MEDIUM_COST_MODEL_2,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="plan",
        model=HIGH_COST_MODEL_2,
        api_key=gk,
        reasoning_effort=reasoning,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="critic",
        model=HIGH_COST_MODEL_2,
        api_key=gk,
        reasoning_effort=reasoning,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="mem",
        model=MEDIUM_COST_MODEL_2,
        api_key=gk,
    )

    if ok is not None:
        ModelRegistry.register(
            name="embed",
            model="text-embedding-3-small",
            api_key=ok,
        )

    ModelRegistry.register(
        name="history",
        model=MEDIUM_COST_MODEL_2,
        api_key=gk,
    )

    ModelRegistry.register(
        name="experiment_agent",
        model=HIGH_COST_MODEL_2,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_coding",
        model=HIGH_COST_MODEL_2,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_execute",
        model=HIGH_COST_MODEL_2,
        api_key=gk,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_monitor",
        model=MEDIUM_COST_MODEL_2,
        api_key=gk,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="experiment_summary",
        model=HIGH_COST_MODEL_2,
        api_key=gk,
        reasoning_effort="low",
    )
