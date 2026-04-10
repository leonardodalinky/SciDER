"""Persistent settings storage for the Streamlit app.

All settings (API keys, model roles) are stored in the browser's localStorage
via raw JS eval. Nothing is saved on the server.
"""

from __future__ import annotations

import json

import streamlit as st

_LS_KEY = "scider_settings"

# All provider key field names
_KEY_FIELDS = ("gemini_api_key", "openai_api_key", "anthropic_api_key", "s2_api_key")


def _ls_get() -> dict | None:
    """Read settings from browser localStorage via JS eval."""
    try:
        from streamlit_js_eval import streamlit_js_eval

        raw = streamlit_js_eval(
            js_expressions=f"JSON.parse(localStorage.getItem('{_LS_KEY}') || 'null')",
            key="_ls_get",
        )
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            return json.loads(raw)
    except Exception:
        pass
    return None


def _ls_set(settings: dict) -> None:
    """Write settings to browser localStorage via JS eval."""
    try:
        from streamlit_js_eval import streamlit_js_eval

        payload = json.dumps(settings, ensure_ascii=True)
        # Escape for JS string literal
        escaped = payload.replace("\\", "\\\\").replace("'", "\\'")
        streamlit_js_eval(
            js_expressions=f"localStorage.setItem('{_LS_KEY}', '{escaped}')",
            key="_ls_set",
        )
    except Exception:
        pass


def _ls_clear() -> None:
    """Remove settings from browser localStorage via JS eval."""
    try:
        from streamlit_js_eval import streamlit_js_eval

        streamlit_js_eval(
            js_expressions=f"localStorage.removeItem('{_LS_KEY}')",
            key="_ls_clear",
        )
    except Exception:
        pass


def load_settings() -> dict:
    """Load settings from browser localStorage.

    Cached in session_state so we don't keep hitting JS eval on every rerun.
    """
    if "_settings_cache" in st.session_state:
        return st.session_state["_settings_cache"]

    settings = _ls_get()
    if not settings or not isinstance(settings, dict):
        settings = {}

    st.session_state["_settings_cache"] = settings
    return settings


def save_settings(settings: dict) -> None:
    """Save settings to browser localStorage + session_state cache."""
    settings = {k: v for k, v in settings.items() if k not in {"model_provider", "api_key"}}
    st.session_state["_settings_cache"] = settings
    _ls_set(settings)


def has_settings() -> bool:
    """Check if at least one provider API key has been configured."""
    settings = load_settings()
    return any(settings.get(k) for k in _KEY_FIELDS)


def clear_settings() -> None:
    """Clear all settings from localStorage and session_state."""
    _ls_clear()
    st.session_state.pop("_settings_cache", None)
