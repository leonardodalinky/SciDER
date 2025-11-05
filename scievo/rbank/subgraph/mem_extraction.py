from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from scievo.core.types import Message
from scievo.prompts import PROMPTS


class MemEntry(BaseModel):
    # Unique id of the memory, a 8 character random string
    id: str
    # Time string of the memory, format: YYYY-MM-DD_HH:MM:SS
    time_str: str
    # Name of the llm used to generate the memory
    llm: str
    # Memory text
    text: str
    # Memory embedding
    embedding: list[float]
    # Name of the agent that calling this subgraph to generate the memory
    # This field should be set by the agent that calling this subgraph
    agent: str | None = None


class MemExtractionState(BaseModel):
    input_msgs: list[Message]
    output_mems: list[MemEntry] = []
    output_error: str | None = None


def mem_extraction_node(state: MemExtractionState) -> MemExtractionState:
    input_msgs_texts = []
    for i, msg in enumerate(state.input_msgs):
        plain = msg.to_plain_text()
        input_msgs_texts.append(f"--- Message {i} Begin ---\n{plain}\n--- Message {i} End ---")
    input_msgs_text: str = "\n".join(input_msgs_texts)

    system_prompt = PROMPTS.rbank.mem_extraction_prompt.format(
        trajectory=input_msgs_text,
    )
    # TODO
    raise NotImplementedError


def build():
    g = StateGraph(MemExtractionState)

    g.add_node("mem_extraction", mem_extraction_node)

    g.add_edge(START, "mem_extraction")
    g.add_edge("mem_extraction", END)
    return g
