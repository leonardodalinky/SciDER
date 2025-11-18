from pathlib import Path

from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel

from scievo.core.errors import AgentError
from scievo.core.llms import ModelRegistry
from scievo.rbank.memo import MemEntry, MemoEmbeddings

LLM_NAME = "embed"
AGENT_NAME = "mem_persistence"


class MemPersistenceState(BaseModel):
    input_mems: list[MemEntry]
    save_dir: str | Path


def mem_persistence_node(state: MemPersistenceState) -> MemPersistenceState:
    logger.debug("Memory Persistence begin: {} mems -> {}", len(state.input_mems), state.save_dir)
    if len(state.input_mems) == 0:
        logger.debug("Persistence error: no mem entries provided")
        raise AgentError("no mem entries provided", agent_name=AGENT_NAME)

    first_llm = state.input_mems[0].llm
    if not all(entry.llm == first_llm for entry in state.input_mems):
        logger.debug("Persistence error: All mem entries must have the same llm")
        raise AgentError("All mem entries must have the same llm", agent_name=AGENT_NAME)

    # ensure save directory exists
    save_dir = Path(state.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    entries_markdown: dict[str, str] = {}
    for entry in state.input_mems:
        base_name = f"{entry.agent}_{entry.time_str}_{entry.id}"

        # persist memo markdown
        md_path = save_dir / f"{base_name}.md"
        try:
            entry.memo.to_markdown_file(md_path)
            entries_markdown[base_name] = entry.memo.to_markdown()
            logger.debug("Persisted memo markdown: {}", md_path)
        except Exception as e:
            logger.debug("Persistence error: persist_markdown_error({}): {}", base_name, e)
            raise AgentError(f"persist_markdown_error: {base_name}", agent_name=AGENT_NAME) from e

    # compute and persist embeddings as json
    try:
        vecs = ModelRegistry.embedding(LLM_NAME, list(entries_markdown.values()))
        logger.debug("Computed embeddings for {} mems", len(vecs))
    except Exception as e:
        logger.debug("Persistence error: embedding_error: {}", e)
        raise AgentError("embedding_error", agent_name=AGENT_NAME) from e

    for base_name, vec in zip(entries_markdown.keys(), vecs):
        mem_emb = MemoEmbeddings(
            embeddings=[MemoEmbeddings._Embedding(llm=LLM_NAME, embedding=vec)]
        )
        emb_json_path = save_dir / f"{base_name}.json"

        try:
            with open(emb_json_path, "w") as f:
                f.write(mem_emb.model_dump_json(indent=2))
            logger.debug("Persisted memo embedding: {}", emb_json_path)
        except Exception as e:
            logger.debug("Persistence error: persist_embedding_error({}): {}", base_name, e)
            raise AgentError(f"persist_embedding_error: {base_name}", agent_name=AGENT_NAME) from e

    logger.debug("Memory Persistence end: {} items persisted", len(entries_markdown))
    return state


@logger.catch
def build():
    g = StateGraph(MemPersistenceState)

    g.add_node("mem_persistence", mem_persistence_node)

    g.add_edge(START, "mem_persistence")
    g.add_edge("mem_persistence", END)
    return g
