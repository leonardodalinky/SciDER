import os


def str_to_bool(s: str | bool) -> bool:
    if isinstance(s, bool):
        return s
    return s.lower() in ("true", "1", "t")


__AGENT_STATE_NAME__ = "agent_state"
__CTX_NAME__ = "ctx"

# ReasoningBank
REASONING_BANK_ENABLED = str_to_bool(os.getenv("REASONING_BANK_ENABLED", True))

LOG_MEM_SUBGRAPH = str_to_bool(os.getenv("LOG_MEM_SUBGRAPH", False))

# Aider
AIDER_GIT = str_to_bool(os.getenv("AIDER_GIT", False))
AIDER_VERBOSE = str_to_bool(os.getenv("AIDER_VERBOSE", False))
AIDER_MODEL = os.getenv("AIDER_MODEL", "gpt-5-nano")
AIDER_REASONING_EFFORT = os.getenv("AIDER_REASONING_EFFORT", "low")
AIDER_COMMIT = str_to_bool(os.getenv("AIDER_COMMIT", False))
AIDER_DIRTY_COMMITS = str_to_bool(os.getenv("AIDER_DIRTY_COMMITS", False))
AIDER_AUTO_COMMITS = str_to_bool(os.getenv("AIDER_AUTO_COMMITS", False))

# history auto compression
HISTORY_AUTO_COMPRESSION = str_to_bool(os.getenv("HISTORY_AUTO_COMPRESSION", True))
HISTORY_AUTO_COMPRESSION_TOKEN_THRESHOLD = int(
    os.getenv("HISTORY_AUTO_COMPRESSION_TOKEN_THRESHOLD", 32000)
)
HISTORY_AUTO_COMPRESSION_KEEP_RATIO = float(os.getenv("HISTORY_AUTO_COMPRESSION_KEEP_RATIO", 0.4))
