import os


def str_to_bool(s: str | bool) -> bool:
    if isinstance(s, bool):
        return s
    return s.lower() in ("true", "1", "t")


__AGENT_STATE_NAME__ = "agent_state"
__CTX_NAME__ = "ctx"

LOG_MEM_SUBGRAPH = str_to_bool(os.getenv("LOG_MEM_SUBGRAPH", False))

# Aider
AIDER_GIT = str_to_bool(os.getenv("AIDER_GIT", False))
AIDER_VERBOSE = str_to_bool(os.getenv("AIDER_VERBOSE", False))
AIDER_MODEL = os.getenv("AIDER_MODEL", "gpt-5-nano")
AIDER_REASONING_EFFORT = os.getenv("AIDER_REASONING_EFFORT", "low")
AIDER_COMMIT = str_to_bool(os.getenv("AIDER_COMMIT", False))
AIDER_DIRTY_COMMITS = str_to_bool(os.getenv("AIDER_DIRTY_COMMITS", False))
AIDER_AUTO_COMMITS = str_to_bool(os.getenv("AIDER_AUTO_COMMITS", False))
