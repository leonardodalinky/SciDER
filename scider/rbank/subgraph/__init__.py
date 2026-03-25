"""RBank subgraph utilities."""

from loguru import logger

from scider.core import constant


def rbank_guard_node(state):
    """Guard node that checks REASONING_BANK_ENABLED. Returns state unchanged."""
    return state


def rbank_guard_conditional(_state) -> str:
    """Route to 'continue' if memory is enabled, 'skip' if disabled."""
    if constant.REASONING_BANK_ENABLED:
        return "continue"
    logger.debug("Memory mechanism disabled, skipping rbank subgraph")
    return "skip"
