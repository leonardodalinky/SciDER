import os

# Import tools that are actually used via LLM tool_calling
from . import (  # noqa: F401
    env_tool,
    exec_tool,
    fs_tool,
    github_tool,
    history_tool,
    shell_tool,
    state_tool,
    todo_tool,
    web_tool,
)
from .registry import Tool, ToolRegistry

# The following were moved to their respective agent modules as internal functions:
# - ideation_tool        -> scider.agents.ideation_agent.ideation_utils
# - paper_search_tool    -> scider.agents.data_agent.paper_subagent.paper_search
# - dataset_search_tool  -> scider.agents.data_agent.paper_subagent.dataset_search
# - metric_search_tool   -> scider.agents.data_agent.paper_subagent.metric_search
# - claude_agent_sdk_tool -> scider.agents.experiment_agent.coding_subagent_v3_claude.claude_sdk
# - claude_code_tool     -> scider.agents.experiment_agent.coding_subagent_v3_claude.claude_code
# - openhands_tool       -> scider.agents.experiment_agent.coding_subagent_v2.openhands_utils
