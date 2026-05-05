"""Unified model catalog.

Single source of truth for all models SciDER knows about, regardless of
provider. Lets the frontend mix-and-match across providers per role.

Loaded from `model_settings/catalog.yaml`. Roles' default model picks live in
`model_settings/role_defaults.yaml`. The frontend may override per-role
choices via `~/.scider/settings.json`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import yaml
from loguru import logger

from scider.core.llms import ModelRegistry

_SETTINGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "model_settings"
DEFAULT_CATALOG_PATH = _SETTINGS_DIR / "catalog.yaml"
DEFAULT_ROLE_DEFAULTS_PATH = _SETTINGS_DIR / "role_defaults.yaml"
PRESETS_DIR = _SETTINGS_DIR / "presets"

CAP_COMPLETION = "completion"
CAP_EMBEDDING = "embedding"


@dataclass
class ModelEntry:
    id: str
    provider: str
    litellm_id: str
    capabilities: list[str]
    requires_env: list[str] = field(default_factory=list)
    default_params: dict[str, Any] = field(default_factory=dict)
    # ``True`` if the model accepts image inputs (multimodal vision).
    # Used by agent prompts to conditionally advertise vision features.
    supports_vision: bool = False

    def is_available(self) -> bool:
        return all(os.getenv(k) for k in self.requires_env)

    def missing_keys(self) -> list[str]:
        return [k for k in self.requires_env if not os.getenv(k)]


class ModelCatalog:
    _lock: RLock = RLock()
    _entries: dict[str, ModelEntry] = {}
    _loaded_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path | None = None, *, force: bool = False) -> None:
        """Load the catalog from YAML. Idempotent unless force=True."""
        with cls._lock:
            target = Path(path) if path else DEFAULT_CATALOG_PATH
            if not force and cls._loaded_path == target and cls._entries:
                return
            if not target.exists():
                raise FileNotFoundError(f"Model catalog not found: {target}")
            data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
            models = data.get("models", {}) or {}
            entries: dict[str, ModelEntry] = {}
            for mid, params in models.items():
                entries[mid] = ModelEntry(
                    id=mid,
                    provider=params.get("provider", "unknown"),
                    litellm_id=params.get("litellm_id", mid),
                    capabilities=list(params.get("capabilities", [CAP_COMPLETION])),
                    requires_env=list(params.get("requires_env", [])),
                    default_params=dict(params.get("default_params", {}) or {}),
                    supports_vision=bool(params.get("supports_vision", False)),
                )
            cls._entries = entries
            cls._loaded_path = target
            logger.debug("Loaded {} models from catalog: {}", len(entries), target)

    @classmethod
    def _ensure_loaded(cls) -> None:
        if not cls._entries:
            cls.load()

    @classmethod
    def all(cls) -> list[ModelEntry]:
        cls._ensure_loaded()
        return list(cls._entries.values())

    @classmethod
    def get(cls, model_id: str) -> ModelEntry | None:
        cls._ensure_loaded()
        return cls._entries.get(model_id)

    @classmethod
    def is_available(cls, model_id: str) -> bool:
        entry = cls.get(model_id)
        return bool(entry and entry.is_available())

    @classmethod
    def missing_keys(cls, model_id: str) -> list[str]:
        entry = cls.get(model_id)
        return entry.missing_keys() if entry else []

    @classmethod
    def by_capability(cls, cap: str) -> list[ModelEntry]:
        cls._ensure_loaded()
        return [e for e in cls._entries.values() if cap in e.capabilities]

    @classmethod
    def by_litellm_id(cls, litellm_id: str) -> ModelEntry | None:
        """Reverse lookup: find the catalog entry whose ``litellm_id`` matches.

        Needed because ``ModelRegistry`` stores the litellm id rather than the
        catalog id, so we can't go directly from a registered role back to the
        original entry without this mapping.
        """
        cls._ensure_loaded()
        for entry in cls._entries.values():
            if entry.litellm_id == litellm_id:
                return entry
        return None


def is_vision_model(role_name: str) -> bool:
    """Return True iff the model currently registered for ``role_name``
    supports multimodal image input.

    Looks up ``role_name`` in the process-global ``ModelRegistry``, reads the
    bound ``litellm_id``, and checks the catalog entry's ``supports_vision``
    flag. Returns ``False`` if the role isn't registered yet or the model is
    unknown to the catalog.
    """
    try:
        params = ModelRegistry.instance().get_model_params(role_name)
    except ValueError:
        return False
    litellm_id = params.get("model")
    if not litellm_id:
        return False
    entry = ModelCatalog.by_litellm_id(litellm_id)
    return bool(entry and entry.supports_vision)


def parse_model_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse a model specifier with optional inline overrides.

    Syntax::

        model_id                              -> ("model_id", {})
        model_id[key=val]                     -> ("model_id", {"key": "val"})
        model_id[reasoning_effort=high,temperature=0.7]
        model_id[reasoning.effort=low]        -> nested: {"reasoning": {"effort": "low"}}

    Values are auto-cast: ``0.7`` -> float, ``42`` -> int, ``true/false`` -> bool,
    everything else stays a string.
    """
    spec = spec.strip()
    if "[" not in spec:
        return spec, {}

    bracket_start = spec.index("[")
    if not spec.endswith("]"):
        raise ValueError(f"Malformed model spec (missing closing ']'): {spec}")

    model_id = spec[:bracket_start].strip()
    inner = spec[bracket_start + 1 : -1].strip()
    if not inner:
        return model_id, {}

    overrides: dict[str, Any] = {}
    for pair in inner.split(","):
        pair = pair.strip()
        if "=" not in pair:
            raise ValueError(f"Invalid override (missing '='): {pair!r} in {spec!r}")
        key, raw_val = pair.split("=", 1)
        key = key.strip()
        val = _auto_cast(raw_val.strip())

        # Support dot notation: reasoning.effort=low -> {"reasoning": {"effort": "low"}}
        parts = key.split(".")
        if len(parts) == 1:
            overrides[key] = val
        else:
            d = overrides
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val

    return model_id, overrides


def _auto_cast(v: str) -> Any:
    """Cast a string to int/float/bool if it looks like one."""
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def register_role(
    role: str,
    model_id: str,
    *,
    api_key: str | None = None,
    **overrides: Any,
) -> None:
    """Register a role -> model binding via the unified catalog.

    ``model_id`` may include inline overrides:
    ``gemini-2.5-flash[reasoning_effort=high]``.

    If ``api_key`` is not provided, falls back to the env var declared in
    the catalog entry's ``requires_env``.

    Explicit ``**overrides`` are merged last (highest priority).
    """
    ModelCatalog._ensure_loaded()
    base_id, spec_overrides = parse_model_spec(model_id)
    entry = ModelCatalog.get(base_id)
    if entry is None:
        raise ValueError(
            f"Unknown model id '{base_id}'. Available: {sorted(ModelCatalog._entries)}"
        )
    params = {**entry.default_params, **spec_overrides, **overrides}
    # default_params may carry kwargs that collide with ModelRegistry.register's
    # explicit signature (api_key/base_url) — common for self-hosted vLLM
    # endpoints. Pop them out so **params doesn't double-pass.
    # Precedence: explicit register_role kwarg > default_params > env var.
    inline_key = params.pop("api_key", None)
    inline_base_url = params.pop("base_url", None)
    resolved_key = api_key or inline_key
    if resolved_key is None and entry.requires_env:
        resolved_key = os.getenv(entry.requires_env[0])
    if resolved_key is None and entry.requires_env:
        logger.warning(
            "Registering role '{}' -> '{}' but no API key provided and env var {} not set",
            role,
            base_id,
            entry.requires_env[0],
        )
    ModelRegistry.register(
        name=role,
        model=entry.litellm_id,
        base_url=inline_base_url,
        api_key=resolved_key,
        **params,
    )


def register_defaults_from_yaml(path: str | Path | None = None) -> dict[str, str]:
    """Register every role from role_defaults.yaml whose model is_available.

    Returns the {role: model_id} map that was actually registered.
    """
    ModelCatalog._ensure_loaded()
    target = Path(path) if path else DEFAULT_ROLE_DEFAULTS_PATH
    if not target.exists():
        raise FileNotFoundError(f"Role defaults file not found: {target}")
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    defaults = data.get("defaults", {}) or {}

    registered: dict[str, str] = {}
    for role, model_spec in defaults.items():
        base_id, _ = parse_model_spec(str(model_spec))
        if not ModelCatalog.is_available(base_id):
            logger.info(
                "Skipping default role '{}' -> '{}': missing {}",
                role,
                base_id,
                ModelCatalog.missing_keys(base_id),
            )
            continue
        register_role(role, str(model_spec))
        registered[role] = base_id
    logger.info("Registered {} default roles from {}", len(registered), target.name)
    return registered


def register_preset(name: str) -> dict[str, str]:
    """Register roles from a named preset under model_settings/presets/.

    ``name`` uses ``/`` as separator, e.g. ``"gemini/medium_high"`` loads
    ``model_settings/presets/gemini/medium_high.yaml``.

    Returns the {role: model_id} map that was actually registered.
    """
    path = PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(
            str(p.relative_to(PRESETS_DIR).with_suffix("")) for p in PRESETS_DIR.rglob("*.yaml")
        )
        raise FileNotFoundError(f"Preset '{name}' not found at {path}. Available: {available}")
    return register_defaults_from_yaml(path)
