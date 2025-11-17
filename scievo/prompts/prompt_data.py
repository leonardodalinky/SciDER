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
    experiment: "ExperimentPrompts"


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


def parse_yaml_as_templates(model_type: Type[T], path: str) -> T:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    data2 = {}
    for field in model_type.__dataclass_fields__.keys():
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
        experiment=parse_yaml_as_templates(
            ExperimentPrompts, os.path.join(DIR, "experiment_prompt.yaml")
        ),
    )


init()
