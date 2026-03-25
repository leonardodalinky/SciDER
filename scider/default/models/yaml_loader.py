"""Load model configurations from YAML files and register them."""

import logging
from pathlib import Path

import yaml

from scider.core.llms import ModelRegistry

logger = logging.getLogger(__name__)

# Default path to model_settings/ at project root
_DEFAULT_SETTINGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "model_settings"

RESERVED_KEYS = {"model", "key_source"}


def register_from_yaml(
    yaml_path: str | Path,
    api_keys: dict[str, str | None] | None = None,
) -> None:
    """Register all models defined in a YAML config file.

    Args:
        yaml_path: Path to the YAML file (absolute or relative to model_settings/).
        api_keys: Mapping of key_source names to API keys, e.g.
            {"gemini": "...", "openai": "..."}.
            If None, keys are resolved from environment variables.
    """
    path = Path(yaml_path)
    if not path.is_absolute():
        path = _DEFAULT_SETTINGS_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Model settings file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models = config.get("models", {})
    if not models:
        logger.warning(f"No models defined in {path}")
        return

    api_keys = api_keys or {}

    for role, params in models.items():
        model = params.get("model")
        key_source = params.get("key_source", "gemini")

        api_key = api_keys.get(key_source)
        if api_key is None:
            # Try environment variable
            import os

            env_var = f"{key_source.upper()}_API_KEY"
            api_key = os.getenv(env_var)

        if api_key is None:
            if role == "embed":
                logger.warning(
                    f"Skipping '{role}' registration: no API key for key_source='{key_source}'"
                )
                continue
            else:
                logger.warning(
                    f"No API key for key_source='{key_source}' (role='{role}'). "
                    f"Registration may fail at runtime."
                )

        # Extract extra kwargs (everything except model and key_source)
        kwargs = {k: v for k, v in params.items() if k not in RESERVED_KEYS}

        ModelRegistry.register(name=role, model=model, api_key=api_key, **kwargs)

    logger.info(f"Registered {len(models)} models from {path.name}")


def get_available_configs() -> list[str]:
    """List available YAML config file names in model_settings/."""
    if not _DEFAULT_SETTINGS_DIR.exists():
        return []
    return sorted(p.stem for p in _DEFAULT_SETTINGS_DIR.glob("*.yaml"))
