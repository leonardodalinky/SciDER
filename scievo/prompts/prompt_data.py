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
    mem_extraction_system_prompt: Template
    mem_extraction_user_prompt: Template


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
    )


init()
