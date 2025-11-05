import os

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as

PROMPTS: "Prompts" = None  # type: ignore


class Prompts(BaseModel):
    dummy: "DummyPrompts"
    data: "DataPrompts"
    rbank: "RBankPrompts"


class DummyPrompts(BaseModel):
    dummy_prompt: str


class DataPrompts(BaseModel):
    system_prompt: str


class RBankPrompts(BaseModel):
    mem_extraction_system_prompt: str
    mem_extraction_user_prompt: str


def init():
    DIR = os.path.dirname(__file__)
    global PROMPTS
    PROMPTS = Prompts(
        dummy=parse_yaml_file_as(DummyPrompts, os.path.join(DIR, "dummy_prompt.yaml")),
        data=parse_yaml_file_as(DataPrompts, os.path.join(DIR, "data_prompt.yaml")),
        rbank=parse_yaml_file_as(RBankPrompts, os.path.join(DIR, "rbank_prompt.yaml")),
    )


init()
