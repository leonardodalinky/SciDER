import os

from dotenv import load_dotenv

load_dotenv()
from pydantic import BaseModel

from scievo.core.llms import ModelRegistry

LOW_COST_MODEL = "gemini/gemini-2.5-flash-lite"
MEDIUM_COST_MODEL = "gemini/gemini-2.5-flash"
HIGH_COST_MODEL = "gemini/gemini-2.5-pro"

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")


def register_gemini_low_medium_models(reasoning: str = "low"):
    """Register Gemini low and medium cost models in the ModelRegistry."""
    ModelRegistry.register(
        name="data",
        model=LOW_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="plan",
        model=MEDIUM_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort=reasoning,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="critic",
        model=LOW_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort=reasoning,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="mem",
        model=LOW_COST_MODEL,
        api_key=GEMINI_KEY,
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
        api_key=GEMINI_KEY,
    )

    ModelRegistry.register(
        name="experiment_agent",
        model=MEDIUM_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_coding",
        model=MEDIUM_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_execute",
        model=MEDIUM_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_monitor",
        model=LOW_COST_MODEL,
        api_key=GEMINI_KEY,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="experiment_summary",
        model=LOW_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort="low",
    )


def register_gemini_medium_high_models(reasoning: str = "low"):
    """Register Gemini medium and high cost models in the ModelRegistry."""
    ModelRegistry.register(
        name="data",
        model=MEDIUM_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="plan",
        model=HIGH_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort=reasoning,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="critic",
        model=HIGH_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort=reasoning,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="mem",
        model=MEDIUM_COST_MODEL,
        api_key=GEMINI_KEY,
    )

    ModelRegistry.register(
        name="embed",
        model="text-embedding-3-small",
        api_key=OPENAI_KEY,
    )

    ModelRegistry.register(
        name="history",
        model=MEDIUM_COST_MODEL,
        api_key=GEMINI_KEY,
    )

    ModelRegistry.register(
        name="experiment_agent",
        model=HIGH_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_coding",
        model=HIGH_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_execute",
        model=HIGH_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort=reasoning,
    )

    ModelRegistry.register(
        name="experiment_monitor",
        model=MEDIUM_COST_MODEL,
        api_key=GEMINI_KEY,
        temperature=0.3,
        top_p=0.9,
    )

    ModelRegistry.register(
        name="experiment_summary",
        model=HIGH_COST_MODEL,
        api_key=GEMINI_KEY,
        reasoning_effort="low",
    )
