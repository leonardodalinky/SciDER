import os

from dotenv import load_dotenv

load_dotenv()
from pydantic import BaseModel

from scievo.core.llms import ModelRegistry

LOW_COST_MODEL = "gpt-5-nano"
MEDIUM_COST_MODEL = "gpt-5-mini"
HIGH_COST_MODEL = "gpt-5.2"

OPENAI_KEY = os.getenv("OPENAI_API_KEY")


def register_gpt_low_medium_models(reasoning: str = "low"):
    """Register GPT low and medium cost models in the ModelRegistry."""
    ModelRegistry.register(
        name="ideation",
        model=MEDIUM_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="paper_search",
        model=LOW_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="metric_search",
        model=LOW_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="data",
        model=LOW_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
            "summary": "detailed",
        },
    )

    ModelRegistry.register(
        name="plan",
        model=MEDIUM_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="critic",
        model=LOW_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="mem",
        model=LOW_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": "minimal",
        },
    )

    # NOTE: Use OpenAI embeddings for better performance
    ModelRegistry.register(
        name="embed",
        model="text-embedding-3-small",
        api_key=OPENAI_KEY,
    )

    ModelRegistry.register(
        name="history",
        model=LOW_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": "minimal",
        },
    )

    ModelRegistry.register(
        name="experiment_agent",
        model=MEDIUM_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="experiment_coding",
        model=MEDIUM_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="experiment_execute",
        model=MEDIUM_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="experiment_monitor",
        model=LOW_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": "minimal",
        },
    )

    ModelRegistry.register(
        name="experiment_summary",
        model=LOW_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": "low",
        },
    )


def register_gpt_medium_high_models(reasoning: str = "low"):
    """Register GPT medium and high cost models in the ModelRegistry."""
    ModelRegistry.register(
        name="ideation",
        model=HIGH_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="paper_search",
        model=MEDIUM_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="metric_search",
        model=MEDIUM_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="data",
        model=MEDIUM_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
            "summary": "detailed",
        },
    )

    ModelRegistry.register(
        name="plan",
        model=HIGH_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="critic",
        model=MEDIUM_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="mem",
        model=MEDIUM_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": "minimal",
        },
    )

    ModelRegistry.register(
        name="embed",
        model="text-embedding-3-small",
        api_key=OPENAI_KEY,
    )

    ModelRegistry.register(
        name="history",
        model=MEDIUM_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": "minimal",
        },
    )

    ModelRegistry.register(
        name="experiment_agent",
        model=HIGH_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="experiment_coding",
        model=HIGH_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="experiment_execute",
        model=HIGH_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": reasoning,
        },
    )

    ModelRegistry.register(
        name="experiment_monitor",
        model=MEDIUM_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": "minimal",
        },
    )

    ModelRegistry.register(
        name="experiment_summary",
        model=MEDIUM_COST_MODEL,
        api_key=OPENAI_KEY,
        reasoning={
            "effort": "low",
        },
    )
