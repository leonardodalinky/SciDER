import re
import secrets
from datetime import datetime
from pathlib import Path

from jinja2 import Template
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel

from scievo.core import constant
from scievo.core.constant import LOG_MEM_SUBGRAPH
from scievo.core.errors import AgentError
from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.core.utils import parse_markdown_from_llm_response
from scievo.prompts import PROMPTS
from scievo.rbank.memo import MemEntry, Memo
from scievo.rbank.subgraph.mem_persistence import MemPersistenceState
from scievo.rbank.subgraph.mem_persistence import build as persistence_subgraph

LLM_NAME = "mem"
AGENT_NAME = "mem_extraction"


class MemExtractionState(BaseModel):
    # session mem save dirs (output)
    mem_dir: str | Path
    # agent name (input)
    input_agent_name: str

    input_msgs: list[Message]
    output_mems: list[MemEntry] = []


def mem_extraction_node(state: MemExtractionState) -> MemExtractionState:
    logger.debug("Memory Extraction begin")
    input_msgs_texts = []
    for i, msg in enumerate(state.input_msgs):
        plain = msg.to_plain_text()
        input_msgs_texts.append(f"--- Message {i} Begin ---\n{plain}\n--- Message {i} End ---")
    input_msgs_text: str = "\n".join(input_msgs_texts)

    long_term_system_prompt = PROMPTS.rbank.mem_extraction_long_term_system_prompt
    project_system_prompt = PROMPTS.rbank.mem_extraction_project_system_prompt
    user_prompt = PROMPTS.rbank.mem_extraction_user_prompt.render(
        trajectory=input_msgs_text,
    )
    user_msg = Message(
        role="user",
        content=user_prompt,
    )

    def extract_mems(system_prompt: Template) -> tuple[list[Memo], bool]:
        mem_msg = ModelRegistry.completion(
            LLM_NAME,
            [user_msg],
            system_prompt=(
                Message(role="system", content=system_prompt.render())
                .with_log(cond=constant.LOG_SYSTEM_PROMPT)
                .content
            ),
            agent_sender=AGENT_NAME,
        ).with_log(LOG_MEM_SUBGRAPH)

        # Look for markdown or generic fenced blocks, preferring markdown
        extracted_md = parse_markdown_from_llm_response(mem_msg)

        # Split into individual memory items by '# Memory Item ...' headings
        item_blocks = list(
            re.finditer(
                r"^\s*#\s*Memory Item[^\n]*\n(.*?)(?=^\s*#\s*Memory Item|\Z)",
                extracted_md,
                flags=re.DOTALL | re.MULTILINE,
            )
        )

        memos: list[Memo] = []
        if item_blocks:
            for m in item_blocks:
                block = m.group(1).strip()
                if not block:
                    continue
                try:
                    memos.append(Memo.from_markdown(block))
                except Exception:
                    logger.debug("Failed to parse memo block: {}", block)
                    continue
        else:
            # Fallback: parse the whole markdown as a single memo
            try:
                memos.append(Memo.from_markdown(extracted_md))
            except Exception as e:
                logger.debug("Markdown parse error: {}", e)
                raise AgentError(f"markdown_parse_error: {e}", agent_name=AGENT_NAME) from e

        if len(memos) == 0:
            raise AgentError("no valid memos", agent_name=AGENT_NAME)

        return memos, True

    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # long_term mems
    long_term_mems, long_term_success = extract_mems(long_term_system_prompt)
    if not long_term_success:
        raise AgentError("long term mem extraction failed", agent_name=AGENT_NAME)
    ## Build MemEntry list
    entries: list[MemEntry] = []
    id = secrets.token_hex(4)
    for i, memo in zip(range(len(long_term_mems)), long_term_mems):
        entries.append(
            MemEntry(
                id=f"L_{id}_{i}",
                time_str=now,
                memo=memo,
                llm=LLM_NAME,
                agent=state.input_agent_name,
            )
        )
    state.output_mems.extend(entries)

    # project mems
    project_mems, project_success = extract_mems(project_system_prompt)
    if not project_success:
        raise AgentError("project mem extraction failed", agent_name=AGENT_NAME)
    ## Build MemEntry list
    entries: list[MemEntry] = []
    id = secrets.token_hex(4)
    for i, memo in zip(range(len(project_mems)), project_mems):
        entries.append(
            MemEntry(
                id=f"P_{id}_{i}",
                time_str=now,
                memo=memo,
                llm=LLM_NAME,
                agent=state.input_agent_name,
            )
        )
    state.output_mems.extend(entries)

    return state


# persistence subgraph
persist_subgraph = persistence_subgraph()
persist_subgraph_compiled = persist_subgraph.compile()


def persistence_node(state: MemExtractionState) -> MemExtractionState:
    # Call the persistence subgraph to persist extracted mem entries
    try:
        # persist all mems
        persist_subgraph_compiled.invoke(
            MemPersistenceState(
                input_mems=state.output_mems,
                save_dir=state.mem_dir,
            )
        )
    except Exception as e:
        logger.debug("Persistence subgraph invoke error: {}", e)
        raise AgentError(f"persistence_subgraph_invoke_error: {e}", agent_name=AGENT_NAME) from e

    return state


@logger.catch
def build():
    g = StateGraph(MemExtractionState)

    g.add_node("mem_extraction", mem_extraction_node)
    g.add_node("persistence", persistence_node)

    g.add_edge(START, "mem_extraction")
    g.add_edge("mem_extraction", "persistence")
    g.add_edge("persistence", END)
    return g
