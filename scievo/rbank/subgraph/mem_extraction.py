import re
import secrets
from datetime import datetime
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel

from scievo.core.constant import LOG_MEM_SUBGRAPH
from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.prompts import PROMPTS
from scievo.rbank.memo import MemEntry, Memo
from scievo.rbank.subgraph.mem_persistence import MemPersistenceState
from scievo.rbank.subgraph.mem_persistence import build as persistence_subgraph

LLM_NAME = "mem"
AGENT_NAME = "mem_extraction"


class MemExtractionState(BaseModel):
    save_dir: str | Path
    input_msgs: list[Message]
    output_mems: list[MemEntry] = []
    output_error: str | None = None


def mem_extraction_node(state: MemExtractionState) -> MemExtractionState:
    logger.debug("Memory Extraction begin")
    input_msgs_texts = []
    for i, msg in enumerate(state.input_msgs):
        plain = msg.to_plain_text()
        input_msgs_texts.append(f"--- Message {i} Begin ---\n{plain}\n--- Message {i} End ---")
    input_msgs_text: str = "\n".join(input_msgs_texts)

    system_prompt = PROMPTS.rbank.mem_extraction_system_prompt
    user_prompt = PROMPTS.rbank.mem_extraction_user_prompt.format(
        trajectory=input_msgs_text,
    )
    user_msg = Message(
        role="user",
        content=user_prompt,
    )
    # use llm to extract memories
    mem_msg = ModelRegistry.completion(
        LLM_NAME,
        [user_msg],
        system_prompt,
        agent_sender=AGENT_NAME,
    ).with_log(LOG_MEM_SUBGRAPH)
    mem_text = mem_msg.content

    # Look for markdown or generic fenced blocks, preferring markdown
    md_match = re.search(
        r"(?:```\s*)?(?:markdown\s*)?(.*)(?:```)?", mem_text, flags=re.DOTALL | re.IGNORECASE
    )
    if not md_match:
        raise ValueError("Markdown not found in the memory extraction response")
    extracted_md = md_match.group(1).strip()

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
            state.output_error = f"markdown_parse_error: {e}"
            state.output_mems = []
            return state

    if len(memos) == 0:
        state.output_error = "no_valid_memos"
        state.output_mems = []
        return state

    # Build MemEntry list
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    entries: list[MemEntry] = []
    id = secrets.token_hex(4)
    for i, memo in zip(range(len(memos)), memos):
        entries.append(
            MemEntry(
                id=f"{id}_{i}",
                time_str=now,
                memo=memo,
                llm=LLM_NAME,
                agent=AGENT_NAME,
            )
        )

    state.output_mems = entries
    return state


# persistence subgraph
persist_subgraph = persistence_subgraph()
persist_subgraph_compiled = persist_subgraph.compile()


def persistence_node(state: MemExtractionState) -> MemExtractionState:
    # Call the persistence subgraph to persist extracted mem entries
    if state.output_error:
        return state

    try:
        res = persist_subgraph_compiled.invoke(
            MemPersistenceState(
                input_mems=state.output_mems,
                save_dir=state.save_dir,
            )
        )
    except Exception as e:
        logger.debug("Persistence subgraph invoke error: {}", e)
        state.output_error = f"persistence_subgraph_invoke_error: {e}"
        return state

    err = res.get("output_error", None)
    if err:
        state.output_error = f"persist_error in persistence subgraph of mem extraction: {err}"
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
