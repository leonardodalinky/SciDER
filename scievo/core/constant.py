import os


def str_to_bool(s: str | bool) -> bool:
    if isinstance(s, bool):
        return s
    return s.lower() in ("true", "1", "t")


__AGENT_STATE_NAME__ = "agent_state"
__CTX_NAME__ = "ctx"

# ReasoningBank
REASONING_BANK_ENABLED = str_to_bool(os.getenv("REASONING_BANK_ENABLED", True))
MEM_EXTRACTION_ROUND_FREQ = int(os.getenv("MEM_EXTRACTION_ROUND_FREQ", 99))
MEM_EXTRACTION_CONTEXT_WINDOW = int(os.getenv("MEM_EXTRACTION_CONTEXT_WINDOW", 16))
MEM_EXTRACTION_MAX_NUM_MEMOS = int(os.getenv("MEM_EXTRACTION_MAX_NUM_MEMOS", 3))
MEM_LONG_TERM_DIR = os.getenv("MEM_LONG_TERM_DIR", "tmp_brain/mem_long_term")
MEM_PROJECT_DIR = os.getenv("MEM_PROJECT_DIR", "tmp_brain/mem_project")


# Logging
LOG_MEM_SUBGRAPH = str_to_bool(os.getenv("LOG_MEM_SUBGRAPH", False))
LOG_SYSTEM_PROMPT = str_to_bool(os.getenv("LOG_SYSTEM_PROMPT", False))

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
