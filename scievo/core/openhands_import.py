"""
OpenHands import helper - sets up paths for local openhands builds.

This module should be imported before any openhands imports to ensure
local builds are found first.

Usage:
    from scievo.core import openhands_import  # Must be before openhands imports
    from openhands.sdk import ...
"""

import sys
from pathlib import Path

from loguru import logger

# OpenHands is intentionally optional in SciEvo. If it's not enabled, this module
# becomes a no-op so importing it cannot mutate sys.path unexpectedly.
_ENABLE_OPENHANDS = __import__("os").getenv("SCIEVO_ENABLE_OPENHANDS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
if not _ENABLE_OPENHANDS:
    __all__ = [
        "_OPENHANDS_SDK_PATH",
        "_OPENHANDS_TOOLS_PATH",
        "_OPENHANDS_SDK_ADDED",
        "_OPENHANDS_TOOLS_ADDED",
    ]
    _SCIEVO_ROOT = Path(__file__).parent.parent.parent
    _OPENHANDS_SDK_PATH = _SCIEVO_ROOT / "software-agent-sdk" / "openhands-sdk"
    _OPENHANDS_TOOLS_PATH = _SCIEVO_ROOT / "software-agent-sdk" / "openhands-tools"
    _OPENHANDS_SDK_ADDED = False
    _OPENHANDS_TOOLS_ADDED = False
    logger.debug("OpenHands disabled; skipping sys.path setup for local OpenHands builds.")
else:
    # Calculate project root (this file is in scievo/core/)
    _SCIEVO_ROOT = Path(__file__).parent.parent.parent
    _OPENHANDS_SDK_PATH = _SCIEVO_ROOT / "software-agent-sdk" / "openhands-sdk"
    _OPENHANDS_TOOLS_PATH = _SCIEVO_ROOT / "software-agent-sdk" / "openhands-tools"

    # Clear any cached openhands modules to force re-import with new paths
    # This is important for notebook environments where modules may be cached
    _openhands_modules_to_clear = [
        m for m in list(sys.modules.keys()) if m.startswith("openhands.") or m == "openhands"
    ]
    if _openhands_modules_to_clear:
        for module_name in _openhands_modules_to_clear:
            del sys.modules[module_name]
        logger.debug(f"Cleared {len(_openhands_modules_to_clear)} cached openhands modules")

    # Add to path if they exist (only once)
    if _OPENHANDS_SDK_PATH.exists() and str(_OPENHANDS_SDK_PATH) not in sys.path:
        sys.path.insert(0, str(_OPENHANDS_SDK_PATH))
        _OPENHANDS_SDK_ADDED = True
        logger.debug(f"Added local openhands-sdk to path: {_OPENHANDS_SDK_PATH}")
    else:
        _OPENHANDS_SDK_ADDED = False
        if not _OPENHANDS_SDK_PATH.exists():
            logger.debug(f"Local openhands-sdk not found at: {_OPENHANDS_SDK_PATH}")

    if _OPENHANDS_TOOLS_PATH.exists() and str(_OPENHANDS_TOOLS_PATH) not in sys.path:
        sys.path.insert(0, str(_OPENHANDS_TOOLS_PATH))
        _OPENHANDS_TOOLS_ADDED = True
        logger.debug(f"Added local openhands-tools to path: {_OPENHANDS_TOOLS_PATH}")
    else:
        _OPENHANDS_TOOLS_ADDED = False
        if not _OPENHANDS_TOOLS_PATH.exists():
            logger.debug(f"Local openhands-tools not found at: {_OPENHANDS_TOOLS_PATH}")

__all__ = [
    "_OPENHANDS_SDK_PATH",
    "_OPENHANDS_TOOLS_PATH",
    "_OPENHANDS_SDK_ADDED",
    "_OPENHANDS_TOOLS_ADDED",
]
