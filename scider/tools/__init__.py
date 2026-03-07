import os

# import tools to register them
from . import (  # noqa: F401
    arxiv_tool,
    claude_agent_sdk_tool,
    claude_code_tool,
    dataset_search_tool,
    env_tool,
    exec_tool,
    fs_tool,
    github_tool,
    history_tool,
    metric_search_tool,
    shell_tool,
    state_tool,
    todo_tool,
    web_tool,
)

# OpenHands is intentionally optional. Avoid importing/registering it unless explicitly enabled,
# since importing it may mutate sys.path and/or require extra dependencies.
_ENABLE_OPENHANDS = os.getenv("SCIDER_ENABLE_OPENHANDS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
if _ENABLE_OPENHANDS:
    from . import openhands_tool  # noqa: F401

from .registry import Tool, ToolRegistry
