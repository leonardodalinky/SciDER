import os

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as

PROMPTS: "Prompts" = None  # type: ignore


class Prompts(BaseModel):
    dummy: "DummyPrompts"


class DummyPrompts(BaseModel):
    dummy_prompt: str


def init():
    DIR = os.path.dirname(__file__)
    global PROMPTS
    PROMPTS = Prompts(
        dummy=parse_yaml_file_as(DummyPrompts, os.path.join(DIR, "dummy_prompt.yaml"))
    )


init()
