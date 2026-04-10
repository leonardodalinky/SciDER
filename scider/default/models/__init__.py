from scider.default.models.catalog import (
    CAP_COMPLETION,
    CAP_EMBEDDING,
    DEFAULT_CATALOG_PATH,
    DEFAULT_ROLE_DEFAULTS_PATH,
    PRESETS_DIR,
    ModelCatalog,
    ModelEntry,
    parse_model_spec,
    register_defaults_from_yaml,
    register_preset,
    register_role,
)
from scider.default.models.gemini import (
    register_gemini3_medium_high_models,
    register_gemini_low_medium_models,
    register_gemini_medium_high_models,
)
from scider.default.models.gpt import (
    register_gpt_low_medium_models,
    register_gpt_medium_high_models,
)

__all__ = [
    # Unified catalog API (preferred)
    "ModelCatalog",
    "ModelEntry",
    "parse_model_spec",
    "register_role",
    "register_defaults_from_yaml",
    "register_preset",
    "PRESETS_DIR",
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_ROLE_DEFAULTS_PATH",
    "CAP_COMPLETION",
    "CAP_EMBEDDING",
    # Deprecated shims
    "register_gemini_low_medium_models",
    "register_gemini_medium_high_models",
    "register_gemini3_medium_high_models",
    "register_gpt_low_medium_models",
    "register_gpt_medium_high_models",
]
