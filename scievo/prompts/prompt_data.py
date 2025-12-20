import os
from dataclasses import dataclass
from typing import Type, TypeVar

import yaml
from jinja2 import Template

T = TypeVar("T")

PROMPTS: "Prompts" = None  # type: ignore


@dataclass
class Prompts:
    dummy: "DummyPrompts"
    data: "DataPrompts"
    rbank: "RBankPrompts"
    history: "HistoryPrompts"
    experiment_coding: "ExperimentPrompts"
    experiment_exec: "ExperimentExecPrompts"
    experiment_summary: "ExperimentSummaryPrompts"
    critic: "CriticPrompts"
    experiment_coding_v2: "ExperimentCodingV2Prompts"
    experiment_claude_coding_v2: "ExperimentClaudeCodingV2Prompts"
    experiment_agent: "ExperimentAgentPrompts"


@dataclass
class DummyPrompts:
    dummy_prompt: Template


@dataclass
class DataPrompts:
    system_prompt: Template
    user_prompt: Template
    planner_system_prompt: Template
    replanner_user_prompt: Template
    replanner_user_response: Template
    summary_system_prompt: Template
    summary_user_prompt: Template


@dataclass
class RBankPrompts:
    mem_extraction_long_term_system_prompt: Template
    mem_extraction_project_system_prompt: Template
    mem_extraction_user_prompt: Template


@dataclass
class HistoryPrompts:
    compression_system_prompt: Template
    compression_user_prompt: Template
    compressed_patch_template: Template
    recall_tool_response: Template


@dataclass
class ExperimentPrompts:
    planner_system_prompt: Template
    planner_user_prompt: Template
    replanner_system_prompt: Template
    replanner_user_prompt: Template
    replanner_user_response: Template
    experiment_chat_system_prompt: Template
    experiment_chat_user_prompt: Template
    experiment_summary_prompt: Template


@dataclass
class ExperimentExecPrompts:
    exec_system_prompt: Template
    exec_user_prompt: Template
    summary_system_prompt: Template
    summary_user_prompt: Template
    monitoring_system_prompt: Template
    monitoring_user_prompt: Template
    monitoring_end_user_prompt: Template
    monitoring_ctrlc_user_prompt: Template


@dataclass
class ExperimentSummaryPrompts:
    system_prompt: Template
    user_prompt: Template
    summary_system_prompt: Template
    summary_prompt: Template


@dataclass
class CriticPrompts:
    system_prompt: Template
    user_prompt: Template
    user_prompt_summary: Template


@dataclass
class ExperimentCodingV2Prompts:
    system_prompt: Template
    planner_system_prompt: Template
    replanner_user_prompt: Template
    replanner_user_response: Template
    user_prompt: Template
    summary_system_prompt: Template
    summary_prompt: Template


@dataclass
class ExperimentAgentPrompts:
    analysis_system_prompt: Template
    judge_system_prompt: Template
    init_prompt: Template
    coding_subagent_query_prompt: Template
    analysis_prompt: Template
    judge_prompt: Template
    revision_feedback_prompt: Template


@dataclass
class ExperimentClaudeCodingV2Prompts:
    system_prompt: Template
    planner_system_prompt: Template
    replanner_user_prompt: Template
    replanner_user_response: Template
    user_prompt: Template


def parse_yaml_as_templates(model_type: Type[T], path: str) -> T:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    data2 = {}
    for field in model_type.__dataclass_fields__.keys():  # type: ignore
        if field.startswith("_"):
            continue
        data2[field] = Template(data[field])
    return model_type(**data2)


def init():
    DIR = os.path.dirname(__file__)

    global PROMPTS

    PROMPTS = Prompts(
        dummy=parse_yaml_as_templates(DummyPrompts, os.path.join(DIR, "dummy_prompt.yaml")),
        data=parse_yaml_as_templates(DataPrompts, os.path.join(DIR, "data_prompt.yaml")),
        rbank=parse_yaml_as_templates(RBankPrompts, os.path.join(DIR, "rbank_prompt.yaml")),
        history=parse_yaml_as_templates(HistoryPrompts, os.path.join(DIR, "history_prompt.yaml")),
        experiment_coding=parse_yaml_as_templates(
            ExperimentPrompts, os.path.join(DIR, "experiment_coding_prompt.yaml")
        ),
        experiment_coding_v2=parse_yaml_as_templates(
            ExperimentCodingV2Prompts,
            os.path.join(DIR, "experiment_coding_prompt_v2.yaml"),
        ),
        experiment_claude_coding_v2=parse_yaml_as_templates(
            ExperimentClaudeCodingV2Prompts,
            os.path.join(DIR, "experiment_claude_coding_prompt_v2.yaml"),
        ),
        experiment_exec=parse_yaml_as_templates(
            ExperimentExecPrompts, os.path.join(DIR, "experiment_exec_prompt.yaml")
        ),
        experiment_summary=parse_yaml_as_templates(
            ExperimentSummaryPrompts,
            os.path.join(DIR, "experiment_summary_prompt.yaml"),
        ),
        critic=parse_yaml_as_templates(CriticPrompts, os.path.join(DIR, "critic_prompt.yaml")),
        experiment_agent=parse_yaml_as_templates(
            ExperimentAgentPrompts, os.path.join(DIR, "experiment_agent_prompt.yaml")
        ),
    )


init()
