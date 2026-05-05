"""Shared helpers for agent execute.py modules.

Every agent prepends a ``<system-reminder>`` meta message to its history
with environment context (date, runtime, workspace, etc.). The exact
``parts`` differ per agent, but the envelope and the idempotency guard are
identical, so they live here.
"""

from datetime import datetime

from .types import Message


def build_system_reminder(parts: list[str], *, preamble: str | None = None) -> str:
    """Render the standard ``<system-reminder>`` envelope around ``parts``.

    ``preamble`` is an optional intro line placed before the parts list.
    """
    body = "\n".join(parts)
    if preamble:
        body = preamble + "\n" + body
    return "<system-reminder>\n" + body + "\n</system-reminder>"


def inject_system_reminder(
    history: list[Message],
    *,
    agent_name: str,
    parts: list[str],
    preamble: str | None = None,
) -> None:
    """Idempotently prepend the system-reminder meta message to ``history``.

    No-op if the first message is already a meta message (i.e. this agent's
    context was already injected on a previous loop entry).
    """
    if history and history[0].is_meta:
        return
    history.insert(
        0,
        Message(
            role="user",
            content=build_system_reminder(parts, preamble=preamble),
            agent_sender=agent_name,
            is_meta=True,
        ),
    )


def today_part() -> str:
    """Standard ``date: YYYY-MM-DD`` part used by every agent."""
    return f"date: {datetime.now().strftime('%Y-%m-%d')}"
