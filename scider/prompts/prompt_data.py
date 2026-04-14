import os
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeVar

import yaml
from jinja2 import Template

T = TypeVar("T")

PROMPTS: "Prompts" = None  # type: ignore


@dataclass
class Prompts:
    data_agent: "DataAgentPrompts"
    critic_agent: "CriticAgentPrompts"
    ideation_agent: "IdeationAgentPrompts"
    experiment_agent: "ExperimentAgentPrompts"
    writing_agent: "WritingAgentPrompts"
    coding_subagent_native: "NativeCodingSubagentPrompts"
    history: "HistoryPrompts"
    paper_subagent: "PaperSubagentPrompts"
    hypo_data_gen: "HypoDataGenPrompts"


@dataclass
class DataAgentPrompts:
    system_prompt: Template
    summary_system_prompt: Template
    summary_user_prompt: Template


@dataclass
class CriticAgentPrompts:
    system_prompt: Template
    summary_system_prompt: Template
    summary_user_prompt: Template


@dataclass
class IdeationAgentPrompts:
    system_prompt: Template
    summary_system_prompt: Template
    summary_user_prompt: Template


@dataclass
class ExperimentAgentPrompts:
    system_prompt: Template
    summary_system_prompt: Template
    summary_user_prompt: Template


@dataclass
class WritingAgentPrompts:
    system_prompt: Template
    summary_system_prompt: Template
    summary_user_prompt: Template


@dataclass
class HistoryPrompts:
    compression_system_prompt: Template
    compression_user_prompt: Template
    compressed_patch_template: Template
    recall_tool_response: Template


@dataclass
class PaperSubagentPrompts:
    summary_system_prompt: Template
    summary_prompt: Template


@dataclass
class NativeCodingSubagentPrompts:
    system_prompt: Template
    summary_system_prompt: Template
    summary_user_prompt: Template


@dataclass
class HypoDataGenPrompts:
    system_prompt: Template
    user_prompt: Template


def parse_yaml_as_templates(model_type: Type[T], path: str | Path) -> T:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    data2 = {}
    for field in model_type.__dataclass_fields__.keys():  # type: ignore
        if field.startswith("_"):
            continue
        data2[field] = Template(data[field])
    return model_type(**data2)


def init():
    DIR = Path(__file__).parent

    global PROMPTS

    PROMPTS = Prompts(
        data_agent=parse_yaml_as_templates(DataAgentPrompts, DIR / "data_agent_prompt.yaml"),
        critic_agent=parse_yaml_as_templates(CriticAgentPrompts, DIR / "critic_agent_prompt.yaml"),
        ideation_agent=parse_yaml_as_templates(
            IdeationAgentPrompts, DIR / "ideation_agent_prompt.yaml"
        ),
        experiment_agent=parse_yaml_as_templates(
            ExperimentAgentPrompts, DIR / "experiment_agent_prompt.yaml"
        ),
        writing_agent=parse_yaml_as_templates(
            WritingAgentPrompts, DIR / "writing_agent_prompt.yaml"
        ),
        coding_subagent_native=parse_yaml_as_templates(
            NativeCodingSubagentPrompts, DIR / "coding_subagent_native_prompt.yaml"
        ),
        history=parse_yaml_as_templates(HistoryPrompts, DIR / "history_prompt.yaml"),
        paper_subagent=parse_yaml_as_templates(
            PaperSubagentPrompts, DIR / "paper_subagent_prompt.yaml"
        ),
        hypo_data_gen=parse_yaml_as_templates(
            HypoDataGenPrompts, DIR / "hypo_data_gen_prompt.yaml"
        ),
    )


init()
