"""
Experiment Execution Agent

This agent is responsible for executing experiments in local shell sessions.
It parses natural language queries to determine commands to execute and manages
the execution using LocalShellSession.
"""

from .build import build
from .state import ExecAgentState

__all__ = ["build", "ExecAgentState"]
