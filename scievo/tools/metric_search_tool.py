"""
Toolset for extracting evaluation metrics from academic papers using RAG.
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from ..core.types import Message
from ..core.utils import wrap_dict_to_toon
from ..rbank.utils import cosine_similarity
from .registry import register_tool, register_toolset_desc

register_toolset_desc("metric_search", "Extract evaluation metrics from academic papers.")


@dataclass
class Metric:
    name: str  # e.g., "Accuracy", "F1 Score", "mAP"
    description: str  # Description of the metric
    domain: str  # e.g., "computer vision", "NLP", "speech"
    paper_title: str  # Title of the paper where this metric was found
    paper_url: Optional[str] = None
    value: Optional[str] = None  # Reported value if mentioned
    formula: Optional[str] = None  # Formula if available
    similarity_score: Optional[float] = None  # RAG relevance score (0-1)


class RAGMetricExtractor:
    """Extract metrics using RAG (Retrieval-Augmented Generation)."""

    def __init__(self, llm_name: str = "metric_search", embedding_llm: Optional[str] = None):
        self.llm_name = llm_name
        self.embedding_llm = embedding_llm or llm_name
        self.embeddings_cache: Dict[str, np.ndarray] = {}  # Cache paper embeddings

    def _get_paper_embedding(self, paper: dict) -> np.ndarray:
        """Get or compute embedding for a paper."""
        paper_id = paper.get("url", paper.get("title", ""))

        if paper_id in self.embeddings_cache:
            return self.embeddings_cache[paper_id]

        # Create text for embedding (title + summary)
        text = f"{paper.get('title', '')}\n{paper.get('summary', '')[:1000]}"

        try:
            # Lazy import to avoid circular dependency
            from ..core.llms import ModelRegistry

            embeddings = ModelRegistry.embedding(self.embedding_llm, [text])
            if embeddings and len(embeddings) > 0:
                embedding = np.array(embeddings[0], dtype=np.float32)
                self.embeddings_cache[paper_id] = embedding
                return embedding
        except Exception as e:
            logger.warning(f"Failed to get embedding for paper: {e}")

        return np.array([])

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for the task query."""
        try:
            # Lazy import to avoid circular dependency
            from ..core.llms import ModelRegistry

            embeddings = ModelRegistry.embedding(self.embedding_llm, [query])
            if embeddings and len(embeddings) > 0:
                return np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to get query embedding: {e}")

        return np.array([])

    def _retrieve_relevant_papers(
        self, papers: List[dict], task_query: str, top_k: int = 5
    ) -> List[Tuple[dict, float]]:
        """Retrieve top-k most relevant papers using vector similarity."""
        if not papers:
            return []

        # Get query embedding
        query_emb = self._get_query_embedding(task_query)
        if len(query_emb) == 0:
            logger.warning("Failed to get query embedding, returning all papers")
            return [(p, 1.0) for p in papers[:top_k]]

        # Compute similarities (cosine_similarity handles normalization internally)
        paper_scores = []
        for paper in papers:
            paper_emb = self._get_paper_embedding(paper)
            if len(paper_emb) == 0:
                continue

            # Compute cosine similarity (function handles normalization)
            similarity = cosine_similarity(query_emb, paper_emb)
            paper_scores.append((paper, similarity))

        # Sort by similarity (descending)
        paper_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        retrieved = paper_scores[:top_k]
        logger.info(
            f"Retrieved {len(retrieved)} papers with similarity scores: "
            f"{[f'{s:.3f}' for _, s in retrieved]}"
        )

        return retrieved

    def extract_from_papers(
        self, papers: List[dict], task_query: str, max_results: int = 20
    ) -> List[Metric]:
        """
        Extract evaluation metrics using RAG approach.

        1. Retrieve most relevant papers using vector similarity
        2. Use retrieved papers as context for LLM extraction

        Args:
            papers: List of paper dictionaries with 'title', 'summary', 'url' fields
            task_query: The original task/query to filter relevant metrics
            max_results: Maximum number of metrics to return

        Returns:
            List of Metric objects
        """
        # If no papers, use fallback to suggest common metrics based on task query
        if not papers:
            logger.info("No papers provided, using fallback to suggest common metrics")
            return self._get_common_metrics(task_query)

        # Step 1: Retrieve relevant papers using RAG
        retrieved_papers = self._retrieve_relevant_papers(
            papers, task_query, top_k=min(10, len(papers))
        )

        if not retrieved_papers:
            logger.warning("No papers retrieved, using fallback")
            return self._get_common_metrics(task_query)

        # Step 2: Prepare context from retrieved papers (use full summary, not truncated)
        papers_text = "\n\n".join(
            [
                f"Paper {i+1} (Relevance: {score:.3f}):\n"
                f"Title: {p.get('title', 'N/A')}\n"
                f"Summary: {p.get('summary', 'N/A')[:800]}\n"  # Use more characters (800 vs 500)
                f"URL: {p.get('url', 'N/A')}"
                for i, (p, score) in enumerate(retrieved_papers)
            ]
        )

        # Use LLM to extract metrics
        system_prompt = """You are an expert at extracting evaluation metrics from academic papers.
Your task is to identify evaluation metrics mentioned in the papers and extract relevant information.

For each metric, provide:
- name: The metric name (e.g., "Accuracy", "F1 Score", "mAP", "BLEU", "ROUGE")
- description: Brief description of what the metric measures
- domain: The research domain (e.g., "computer vision", "NLP", "speech recognition")
- value: If a specific value is mentioned, include it
- formula: If a formula is mentioned, include it

Return a JSON array of metrics. Focus on metrics that are relevant to the task query."""

        user_prompt = f"""Extract evaluation metrics from the following papers that are relevant to the task: "{task_query}"

The papers are ranked by relevance to your query (higher score = more relevant):

{papers_text}

Extract all relevant evaluation metrics mentioned in these papers. Return a JSON array with the following structure:
[
  {{
    "name": "metric name",
    "description": "what it measures",
    "domain": "research domain",
    "paper_title": "title of the paper",
    "paper_url": "URL if available",
    "value": "reported value if mentioned",
    "formula": "formula if available"
  }}
]

Focus on metrics that are commonly used in the research area and relevant to the task query.
Pay special attention to metrics from papers with higher relevance scores."""

        try:
            # Lazy import to avoid circular dependency
            from ..core.llms import ModelRegistry

            # Call LLM to extract metrics
            msg = ModelRegistry.completion(
                self.llm_name,
                [Message(role="user", content=user_prompt)],
                system_prompt=system_prompt,
                agent_sender="metric_extractor",
                tools=None,
            )

            # Parse JSON response
            content = msg.content or "[]"

            # Try to extract JSON from the response
            try:
                # First try direct JSON parse
                metrics_data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
                if json_match:
                    metrics_data = json.loads(json_match.group(1))
                else:
                    # Try to find JSON array in the text
                    json_match = re.search(r"\[.*?\]", content, re.DOTALL)
                    if json_match:
                        metrics_data = json.loads(json_match.group(0))
                    else:
                        metrics_data = []

            # Convert to Metric objects with similarity scores
            metrics = []
            seen_metrics = set()  # Deduplicate by name

            # Create a mapping from paper title to similarity score
            paper_scores_map = {p.get("title", ""): score for p, score in retrieved_papers}

            for item in metrics_data[:max_results]:
                if not isinstance(item, dict):
                    continue

                metric_name = item.get("name", "").strip().lower()
                if not metric_name or metric_name in seen_metrics:
                    continue

                seen_metrics.add(metric_name)

                paper_title = item.get("paper_title", "")
                similarity_score = paper_scores_map.get(paper_title, None)

                metric = Metric(
                    name=item.get("name", "Unknown"),
                    description=item.get("description", ""),
                    domain=item.get("domain", "general"),
                    paper_title=paper_title,
                    paper_url=item.get("paper_url"),
                    value=item.get("value"),
                    formula=item.get("formula"),
                    similarity_score=similarity_score,
                )
                metrics.append(metric)

            # Sort by similarity score (if available)
            metrics.sort(key=lambda m: m.similarity_score or 0.0, reverse=True)

            logger.info(f"Extracted {len(metrics)} metrics using RAG approach")
            return metrics

        except Exception as e:
            # Fallback: return common metrics based on domain
            return self._get_common_metrics(task_query)

    def _get_common_metrics(self, task_query: str) -> List[Metric]:
        """Fallback: return common metrics based on task domain."""
        query_lower = task_query.lower()

        common_metrics = []

        # Computer Vision metrics
        if any(
            term in query_lower for term in ["vision", "image", "object detection", "segmentation"]
        ):
            common_metrics.extend(
                [
                    Metric(
                        name="mAP",
                        description="Mean Average Precision for object detection",
                        domain="computer vision",
                        paper_title="Common metric",
                    ),
                    Metric(
                        name="IoU",
                        description="Intersection over Union for segmentation",
                        domain="computer vision",
                        paper_title="Common metric",
                    ),
                    Metric(
                        name="Accuracy",
                        description="Classification accuracy",
                        domain="computer vision",
                        paper_title="Common metric",
                    ),
                ]
            )

        # NLP metrics
        if any(
            term in query_lower
            for term in ["nlp", "language", "translation", "text", "bert", "transformer"]
        ):
            common_metrics.extend(
                [
                    Metric(
                        name="BLEU",
                        description="Bilingual Evaluation Understudy for translation quality",
                        domain="NLP",
                        paper_title="Common metric",
                    ),
                    Metric(
                        name="ROUGE",
                        description="Recall-Oriented Understudy for Gisting Evaluation",
                        domain="NLP",
                        paper_title="Common metric",
                    ),
                    Metric(
                        name="F1 Score",
                        description="Harmonic mean of precision and recall",
                        domain="NLP",
                        paper_title="Common metric",
                    ),
                ]
            )

        # General metrics
        if not common_metrics:
            common_metrics.append(
                Metric(
                    name="Accuracy",
                    description="Overall classification accuracy",
                    domain="general",
                    paper_title="Common metric",
                )
            )
            common_metrics.append(
                Metric(
                    name="F1 Score",
                    description="Harmonic mean of precision and recall",
                    domain="general",
                    paper_title="Common metric",
                )
            )

        return common_metrics


# Register the tool with the framework
@register_tool(
    "metric_search",
    {
        "type": "function",
        "function": {
            "name": "extract_metrics_from_papers",
            "description": "Extract evaluation metrics from academic papers using RAG (Retrieval-Augmented Generation)",
            "parameters": {
                "type": "object",
                "properties": {
                    "papers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "summary": {"type": "string"},
                                "url": {"type": "string"},
                            },
                        },
                        "description": "List of paper dictionaries with title, summary, and url",
                    },
                    "task_query": {
                        "type": "string",
                        "description": "The original task/query to filter relevant metrics",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of metrics to return",
                        "default": 20,
                    },
                },
                "required": ["papers", "task_query"],
            },
        },
    },
)
def extract_metrics_from_papers(papers: List[dict], task_query: str, max_results: int = 20) -> str:
    """
    Extract evaluation metrics from academic papers using RAG.

    This function uses vector embeddings to retrieve the most relevant papers,
    then uses LLM to extract metrics from the retrieved context.

    Args:
        papers: List of paper dictionaries with 'title', 'summary', 'url' fields
        task_query: The original task/query to filter relevant metrics
        max_results: Maximum number of metrics to return

    Returns:
        str: TOON-formatted string containing the extracted metrics
    """
    try:
        extractor = RAGMetricExtractor()
        metrics = extractor.extract_from_papers(papers, task_query, max_results)

        # Convert Metric objects to dictionaries
        result = [
            {
                "name": metric.name,
                "description": metric.description,
                "domain": metric.domain,
                "paper_title": metric.paper_title,
                "paper_url": metric.paper_url,
                "value": metric.value,
                "formula": metric.formula,
                "similarity_score": metric.similarity_score,  # Include relevance score
            }
            for metric in metrics
        ]

        return wrap_dict_to_toon(result)
    except Exception as e:
        logger.error(f"Error extracting metrics with RAG: {e}")
        return wrap_dict_to_toon({"error": f"Error extracting metrics: {e}"})
