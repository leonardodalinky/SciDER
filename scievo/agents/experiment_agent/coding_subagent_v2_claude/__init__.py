"""
Coding Subagent V2 with Claude Agent SDK

This agent follows the plan-and-execute paradigm for coding tasks.
It integrates with Claude Agent SDK for external code manipulation.
"""

from .build import build
from .state import CodingAgentState

__all__ = ["build", "CodingAgentState"]
