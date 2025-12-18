"""
SciEvo Workflows Package

This package contains different workflow implementations:
- full_workflow: Complete workflow from data analysis to experiment execution (0 to 完全体)
- data_workflow: Partial workflow for data analysis only (DataAgent)
- experiment_workflow: Partial workflow for experiment execution only (ExperimentAgent)
"""

from .data_workflow import DataWorkflow, run_data_workflow
from .experiment_workflow import ExperimentWorkflow, run_experiment_workflow
from .full_workflow import FullWorkflow, run_full_workflow

__all__ = [
    # Full workflow
    "FullWorkflow",
    "run_full_workflow",
    # Partial workflows
    "DataWorkflow",
    "run_data_workflow",
    "ExperimentWorkflow",
    "run_experiment_workflow",
]
