"""
SciDER Workflows Package

This package contains different workflow implementations:
- full_workflow: Complete workflow from data analysis to experiment execution
- full_workflow_with_ideation: Ideation -> (optional) data -> (optional) experiment
- data_workflow: Partial workflow for data analysis only (DataAgent)
- experiment_workflow: Partial workflow for experiment execution only (ExperimentAgent)
- ideation_workflow: Partial workflow for research ideation only (IdeationAgent)
- hypo_data_workflow: Generates synthetic data, then analyzes it
"""

from .data_workflow import DataWorkflow, run_data_workflow
from .experiment_workflow import ExperimentWorkflow, run_experiment_workflow
from .full_workflow import FullWorkflow, run_full_workflow
from .full_workflow_with_ideation import FullWorkflowWithIdeation, run_full_workflow_with_ideation
from .hypo_data_workflow import HypoDataWorkflow, run_hypo_data_workflow
from .ideation_workflow import IdeationWorkflow, run_ideation_workflow

__all__ = [
    # Full workflows
    "FullWorkflow",
    "run_full_workflow",
    "FullWorkflowWithIdeation",
    "run_full_workflow_with_ideation",
    # Partial workflows
    "IdeationWorkflow",
    "run_ideation_workflow",
    "DataWorkflow",
    "run_data_workflow",
    "HypoDataWorkflow",
    "run_hypo_data_workflow",
    "ExperimentWorkflow",
    "run_experiment_workflow",
]
