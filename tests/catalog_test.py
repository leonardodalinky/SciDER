"""Tests for the unified ModelCatalog."""

from __future__ import annotations

import os

import pytest

from scider.core.llms import ModelRegistry
from scider.default.models import (
    CAP_COMPLETION,
    CAP_EMBEDDING,
    ModelCatalog,
    is_vision_model,
    parse_model_spec,
    register_defaults_from_yaml,
    register_role,
)


@pytest.fixture(autouse=True)
def reload_catalog():
    ModelCatalog.load(force=True)
    yield


def _set_env(monkeypatch, **vars):
    for k, v in vars.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)


def test_catalog_loads_known_models():
    ids = {e.id for e in ModelCatalog.all()}
    assert "gemini-2.5-pro" in ids
    assert "gpt-5-mini" in ids
    assert "text-embedding-3-small" in ids


def test_by_capability_split():
    completion_ids = {e.id for e in ModelCatalog.by_capability(CAP_COMPLETION)}
    embedding_ids = {e.id for e in ModelCatalog.by_capability(CAP_EMBEDDING)}
    assert "gemini-2.5-pro" in completion_ids
    assert "text-embedding-3-small" in embedding_ids
    # No overlap by current catalog
    assert completion_ids.isdisjoint(embedding_ids)


def test_is_available_reflects_env(monkeypatch):
    _set_env(monkeypatch, GEMINI_API_KEY=None, OPENAI_API_KEY=None)
    assert ModelCatalog.is_available("gemini-2.5-pro") is False
    assert ModelCatalog.missing_keys("gemini-2.5-pro") == ["GEMINI_API_KEY"]

    _set_env(monkeypatch, GEMINI_API_KEY="fake")
    assert ModelCatalog.is_available("gemini-2.5-pro") is True
    assert ModelCatalog.missing_keys("gemini-2.5-pro") == []


def test_register_role_uses_litellm_id_and_default_params(monkeypatch):
    _set_env(monkeypatch, GEMINI_API_KEY="fake-gemini")
    register_role("ideation", "gemini-2.5-pro")
    params = ModelRegistry.instance().get_model_params("ideation")
    assert params["model"] == "gemini/gemini-2.5-pro"
    assert params["api_key"] == "fake-gemini"
    assert params["reasoning_effort"] == "low"


def test_register_role_overrides_merge(monkeypatch):
    _set_env(monkeypatch, GEMINI_API_KEY="fake")
    register_role("ideation", "gemini-2.5-pro", reasoning_effort="high", temperature=0.7)
    params = ModelRegistry.instance().get_model_params("ideation")
    assert params["reasoning_effort"] == "high"
    assert params["temperature"] == 0.7


def test_register_role_unknown_model():
    with pytest.raises(ValueError):
        register_role("ideation", "no-such-model-id")


def test_register_role_missing_key_warns_but_registers(monkeypatch):
    _set_env(monkeypatch, GEMINI_API_KEY=None)
    register_role("ideation", "gemini-2.5-pro")
    params = ModelRegistry.instance().get_model_params("ideation")
    assert params["api_key"] is None


def test_cross_provider_mixing(monkeypatch):
    _set_env(monkeypatch, GEMINI_API_KEY="g", OPENAI_API_KEY="o")
    register_role("ideation", "gemini-2.5-pro")
    register_role("experiment_coding", "gpt-5-mini")
    ideation = ModelRegistry.instance().get_model_params("ideation")
    coding = ModelRegistry.instance().get_model_params("experiment_coding")
    assert ideation["model"].startswith("gemini/")
    assert coding["model"] == "gpt-5-mini"
    assert ideation["api_key"] == "g"
    assert coding["api_key"] == "o"


def test_register_defaults_skips_unavailable(monkeypatch):
    _set_env(monkeypatch, GEMINI_API_KEY="g", OPENAI_API_KEY=None)
    registered = register_defaults_from_yaml()
    # Gemini-backed roles should land; openai-only ones (e.g. embedding default
    # text-embedding-3-small) should be skipped.
    assert "ideation" in registered
    assert "embedding" not in registered


# --- parse_model_spec tests ---


def test_parse_plain_id():
    mid, overrides = parse_model_spec("gemini-2.5-pro")
    assert mid == "gemini-2.5-pro"
    assert overrides == {}


def test_parse_single_override():
    mid, overrides = parse_model_spec("gemini-2.5-flash[reasoning_effort=high]")
    assert mid == "gemini-2.5-flash"
    assert overrides == {"reasoning_effort": "high"}


def test_parse_multiple_overrides():
    mid, overrides = parse_model_spec("gpt-5[temperature=0.7,top_p=0.9]")
    assert mid == "gpt-5"
    assert overrides == {"temperature": 0.7, "top_p": 0.9}


def test_parse_dot_notation():
    mid, overrides = parse_model_spec("gpt-5-mini[reasoning.effort=low,reasoning.summary=detailed]")
    assert mid == "gpt-5-mini"
    assert overrides == {"reasoning": {"effort": "low", "summary": "detailed"}}


def test_parse_bool_and_int():
    _, overrides = parse_model_spec("x[flag=true,count=42]")
    assert overrides == {"flag": True, "count": 42}


def test_parse_empty_brackets():
    mid, overrides = parse_model_spec("gemini-2.5-pro[]")
    assert mid == "gemini-2.5-pro"
    assert overrides == {}


def test_register_role_with_inline_overrides(monkeypatch):
    _set_env(monkeypatch, GEMINI_API_KEY="g")
    register_role("critic", "gemini-2.5-pro[reasoning_effort=high,temperature=0.3]")
    params = ModelRegistry.instance().get_model_params("critic")
    assert params["model"] == "gemini/gemini-2.5-pro"
    assert params["reasoning_effort"] == "high"
    assert params["temperature"] == 0.3


# --- vision capability tests ---


class TestVisionCapability:
    """``supports_vision`` is loaded from catalog.yaml and surfaced via
    ``ModelEntry.supports_vision`` + ``is_vision_model(role_name)``.
    """

    def test_vision_flag_loaded_from_yaml(self):
        # Representative models from each provider known to have vision.
        for mid in (
            "gemini-2.5-pro",
            "gemini-3.1-pro-preview",
            "gpt-5",
            "gpt-5-mini",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "o3",
            "o4-mini",
        ):
            entry = ModelCatalog.get(mid)
            assert entry is not None, f"{mid} missing from catalog"
            assert entry.supports_vision is True, f"{mid} expected vision=True"

    def test_text_only_models_flagged_false(self):
        for mid in ("o1-mini", "o3-mini"):
            entry = ModelCatalog.get(mid)
            assert entry is not None
            assert entry.supports_vision is False, f"{mid} expected vision=False"

    def test_embedding_models_have_no_vision_flag(self):
        entry = ModelCatalog.get("text-embedding-3-small")
        assert entry is not None
        assert entry.supports_vision is False

    def test_by_litellm_id_reverse_lookup(self):
        entry = ModelCatalog.by_litellm_id("gemini/gemini-2.5-pro")
        assert entry is not None
        assert entry.id == "gemini-2.5-pro"

        # Unknown litellm id returns None
        assert ModelCatalog.by_litellm_id("no/such-model") is None

    def test_is_vision_model_for_registered_role(self, monkeypatch):
        _set_env(monkeypatch, GEMINI_API_KEY="g", OPENAI_API_KEY="o")
        register_role("experiment", "gemini-2.5-pro")
        register_role("experiment_coding", "o3-mini")

        assert is_vision_model("experiment") is True
        assert is_vision_model("experiment_coding") is False

    def test_is_vision_model_for_unregistered_role(self):
        assert is_vision_model("definitely_not_a_registered_role") is False
