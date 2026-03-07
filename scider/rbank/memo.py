from pathlib import Path

import numpy as np
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import BaseModel


class MemEntry(BaseModel):
    """
    Memory entry used in runtime. Only used in subgraph, as intermediate result.
    """

    # Unique id of the memory, random string
    id: str
    # Time string of the memory, format: YYYY-MM-DD_HH:MM:SS
    time_str: str
    # Name of the llm used to generate the memory
    llm: str
    # Memory text
    memo: "Memo"
    # Name of the agent that calling this subgraph to generate the memory
    # This field should be set by the agent that calling this subgraph
    agent: str | None = None


class Memo(BaseModel):
    """
    Persistent memory.
    """

    title: str
    description: str
    content: str

    # TODO: cached
    @classmethod
    def from_markdown_file(cls, path: str | Path) -> "Memo":
        with open(path, "r") as f:
            return cls.from_markdown(f.read())

    def to_markdown_file(self, path: str | Path) -> None:
        with open(path, "w") as f:
            f.write(self.to_markdown())

    @classmethod
    def from_markdown(cls, md: str) -> "Memo":
        headers_to_split_on = [
            ("##", "section"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            return_each_line=False,
            strip_headers=True,
        )

        docs = markdown_splitter.split_text(md)
        if not docs:
            raise ValueError("No markdown content to parse")

        # Capture only the first occurrence of each section
        title = ""
        description = ""
        content = ""
        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            sec = (meta.get("section") or "").strip().lower()
            text = (getattr(doc, "page_content", "") or "").strip()
            if not text:
                continue
            if sec == "title" and not title:
                title = text
            elif sec == "description" and not description:
                description = text
            elif sec == "content" and not content:
                content = text

        if not (title or content or description):
            raise ValueError("Invalid memory item parsed from markdown")

        return cls(title=title, description=description, content=content)

    def to_markdown(self) -> str:
        return f"""\
## Title

{self.title}

## Description

{self.description}

## Content

{self.content}
"""


class MemoEmbeddings(BaseModel):
    """
    Persistent memory embeddings.
    """

    class _Embedding(BaseModel):
        # Name of the llm used to generate the embedding
        llm: str
        # Embedding vector
        embedding: list[float]

    embeddings: list[_Embedding]

    def get_embedding(self, llm: str) -> np.ndarray | None:
        for e in self.embeddings:
            if e.llm == llm:
                return np.array(e.embedding, dtype=np.float32)
        return None

    # TODO: cached
    @classmethod
    def from_json_file(cls, path: str | Path) -> "MemoEmbeddings":
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())

    def to_json_file(self, path: str | Path) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))
