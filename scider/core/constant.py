import contextlib
import os


def str_to_bool(s: str | bool) -> bool:
    if isinstance(s, bool):
        return s
    return s.lower() in ("true", "1", "t")


__AGENT_STATE_NAME__ = "agent_state"
__CTX_NAME__ = "ctx"

# Logging
LOG_SYSTEM_PROMPT = str_to_bool(os.getenv("LOG_SYSTEM_PROMPT", False))

# Semantic Scholar
S2_API_KEY = os.getenv("S2_API_KEY", "")

# HuggingFace Dataset Download
HF_DATASET_DOWNLOAD_ENABLED = str_to_bool(os.getenv("HF_DATASET_DOWNLOAD_ENABLED", False))
HF_DATASET_CACHE_DIR = os.getenv("HF_DATASET_CACHE_DIR", "tmp_hf_datasets")
HF_DATASET_MAX_SIZE_MB = int(os.getenv("HF_DATASET_MAX_SIZE_MB", 100))

# User Approval
USER_APPROVAL_ENABLED = str_to_bool(os.getenv("USER_APPROVAL_ENABLED", True))
# When auto-approval is on (USER_APPROVAL_ENABLED=false), route critic's
# user_review through an LLM subagent that judges result quality
# (not process errors). Set to false to fall back to blind auto-approve.
APPROVAL_SUBAGENT_ENABLED = str_to_bool(os.getenv("APPROVAL_SUBAGENT_ENABLED", True))


@contextlib.contextmanager
def override_user_approval(enabled: bool):
    """Temporarily override USER_APPROVAL_ENABLED and reset the approval handler cache."""
    import scider.core.approval as _approval
    import scider.core.constant as _self

    old_val = _self.USER_APPROVAL_ENABLED
    old_handler = _approval._default_handler
    _self.USER_APPROVAL_ENABLED = enabled
    _approval._default_handler = None  # force re-detection
    try:
        yield
    finally:
        _self.USER_APPROVAL_ENABLED = old_val
        _approval._default_handler = old_handler
