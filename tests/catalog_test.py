"""Tests for the unified ModelCatalog."""

from __future__ import annotations

import os

import pytest

from scider.core.llms import ModelRegistry
from scider.default.models import (
    CAP_COMPLETION,
    CAP_EMBEDDING,
    ModelCatalog,
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
