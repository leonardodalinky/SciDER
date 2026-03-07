"""
Experiment Agent - High-level orchestrator for code modification experiments.

This agent coordinates three sub-agents:
1. Coding Subagent V2 - Plans and executes code modifications
2. Exec Subagent - Runs experiments/commands in a local shell
3. Summary Subagent - Generates comprehensive experiment summaries

The agent runs in a revision loop until the experiment succeeds or max revisions is reached.
"""

from .build import build
from .state import ExperimentAgentState

__all__ = ["build", "ExperimentAgentState"]
