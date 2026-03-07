import logging
import os
import sys

from scider.core.llms import ModelRegistry

logger = logging.getLogger(__name__)

LOW_COST_MODEL = "gpt-5-nano"
MEDIUM_COST_MODEL = "gpt-5-mini"
HIGH_COST_MODEL = "gpt-5.2"


def _resolve_key(openai_key: str | None = None) -> str:
    """Resolve OpenAI API key from argument or environment variable.

    Returns:
        The resolved OpenAI API key.

    Raises:
        SystemExit: If openai_key is not provided and not found in environment.
    """
    if openai_key is None:
        openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        logger.error(
            "OPENAI_API_KEY is required but not provided via argument or environment variable."
        )
        sys.exit(1)
    return openai_key


def register_gpt_low_medium_models(
    reasoning: str = "low",
    openai_key: str | None = None,
) -> None:
    """Register GPT low and medium cost models in the ModelRegistry.

    Args:
        reasoning: Reasoning effort level.
        openai_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
    """
    key = _resolve_key(openai_key)

    ModelRegistry.register(
        name="ideation",
        model=MEDIUM_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="paper_search",
        model=LOW_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="metric_search",
        model=LOW_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="data",
        model=LOW_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
            "summary": "detailed",
        },
    )

    ModelRegistry.register(
        name="plan",
        model=MEDIUM_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="critic",
        model=LOW_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="mem",
        model=LOW_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": "minimal",
        },
    )

    # NOTE: Use OpenAI embeddings for better performance
    ModelRegistry.register(
        name="embed",
        model="text-embedding-3-small",
        api_key=key,
    )

    ModelRegistry.register(
        name="history",
        model=LOW_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": "minimal",
        },
    )

    ModelRegistry.register(
        name="experiment_agent",
        model=MEDIUM_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="experiment_coding",
        model=MEDIUM_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="experiment_execute",
        model=MEDIUM_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="experiment_monitor",
        model=LOW_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": "minimal",
        },
    )

    ModelRegistry.register(
        name="experiment_summary",
        model=LOW_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": "low",
        },
    )


def register_gpt_medium_high_models(
    reasoning: str = "low",
    openai_key: str | None = None,
) -> None:
    """Register GPT medium and high cost models in the ModelRegistry.

    Args:
        reasoning: Reasoning effort level.
        openai_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
    """
    key = _resolve_key(openai_key)

    ModelRegistry.register(
        name="ideation",
        model=HIGH_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="paper_search",
        model=MEDIUM_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="metric_search",
        model=MEDIUM_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="data",
        model=MEDIUM_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
            "summary": "detailed",
        },
    )

    ModelRegistry.register(
        name="plan",
        model=HIGH_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="critic",
        model=MEDIUM_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="mem",
        model=MEDIUM_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": "minimal",
        },
    )

    ModelRegistry.register(
        name="embed",
        model="text-embedding-3-small",
        api_key=key,
    )

    ModelRegistry.register(
        name="history",
        model=MEDIUM_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": "minimal",
        },
    )

    ModelRegistry.register(
        name="experiment_agent",
        model=HIGH_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="experiment_coding",
        model=HIGH_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="experiment_execute",
        model=HIGH_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="experiment_monitor",
        model=MEDIUM_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": "minimal",
        },
    )

    ModelRegistry.register(
        name="experiment_summary",
        model=MEDIUM_COST_MODEL,
        api_key=key,
        reasoning={
            "effort": "low",
        },
    )
