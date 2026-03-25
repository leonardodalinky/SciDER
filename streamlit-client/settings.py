"""Persistent settings storage for the Streamlit app.

Settings are stored in ~/.scider/settings.json and persist across browser
sessions / page refreshes.
"""

import json
from pathlib import Path

SETTINGS_DIR = Path.home() / ".scider"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

REQUIRED_KEYS = ["api_key", "model_provider"]


def load_settings() -> dict:
    """Load settings from disk. Returns empty dict if file not found."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_settings(settings: dict) -> None:
    """Save settings to disk."""
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def has_settings() -> bool:
    """Check if valid settings exist (all required keys present and non-empty)."""
    settings = load_settings()
    return all(settings.get(k) for k in REQUIRED_KEYS)


def clear_settings() -> None:
    """Delete the settings file."""
    if SETTINGS_FILE.exists():
        SETTINGS_FILE.unlink()
