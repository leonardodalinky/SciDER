"""Approval subagent — LLM judge that decides whether a result gates proceed.

Invoked by ``SubagentApprovalHandler`` when auto-approval is on and a
result-style approval (currently only ``user_review``) needs judgment.
Focuses on RESULT quality (artifacts, conclusiveness), not process errors.
"""

from .execute import run_approval_subagent

__all__ = ["run_approval_subagent"]
