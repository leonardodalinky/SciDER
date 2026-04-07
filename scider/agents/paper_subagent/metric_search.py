"""
Toolset for extracting evaluation metrics from academic papers using RAG.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from scider.core.types import Message


@dataclass
class Metric:
    name: str  # e.g., "Accuracy", "F1 Score", "mAP"
    description: str  # Description of the metric
    domain: str  # e.g., "computer vision", "NLP", "speech"
    paper_title: str  # Title of the paper where this metric was found
    paper_url: Optional[str] = None
    value: Optional[str] = None  # Reported value if mentioned
    formula: Optional[str] = None  # Formula if available


class RAGMetricExtractor:
    """Extract metrics from academic papers using LLM."""

    def __init__(self, llm_name: str = "metric_search"):
        self.llm_name = llm_name

    def extract_from_papers(
        self, papers: List[dict], task_query: str, max_results: int = 20
    ) -> List[Metric]:
        """
        Extract evaluation metrics from papers using LLM.

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

        # Use all papers directly (typically <=10)
        selected_papers = papers[:10]

        # Prepare context from papers
        papers_text = "\n\n".join(
            [
                f"Paper {i+1}:\n"
                f"Title: {p.get('title', 'N/A')}\n"
                f"Summary: {p.get('summary', 'N/A')[:800]}\n"
                f"URL: {p.get('url', 'N/A')}"
                for i, p in enumerate(selected_papers)
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
            from scider.core.llms import ModelRegistry

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

            # Convert to Metric objects
            metrics = []
            seen_metrics = set()  # Deduplicate by name

            for item in metrics_data[:max_results]:
                if not isinstance(item, dict):
                    continue

                metric_name = item.get("name", "").strip().lower()
                if not metric_name or metric_name in seen_metrics:
                    continue

                seen_metrics.add(metric_name)

                metric = Metric(
                    name=item.get("name", "Unknown"),
                    description=item.get("description", ""),
                    domain=item.get("domain", "general"),
                    paper_title=item.get("paper_title", ""),
                    paper_url=item.get("paper_url"),
                    value=item.get("value"),
                    formula=item.get("formula"),
                )
                metrics.append(metric)

            logger.info(f"Extracted {len(metrics)} metrics")
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


def extract_metrics_from_papers(papers: List[dict], task_query: str, max_results: int = 20) -> str:
    """
    Extract evaluation metrics from academic papers using LLM.

    Args:
        papers: List of paper dictionaries with 'title', 'summary', 'url' fields
        task_query: The original task/query to filter relevant metrics
        max_results: Maximum number of metrics to return

    Returns:
        str: JSON-formatted string containing the extracted metrics
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
            }
            for metric in metrics
        ]

        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error extracting metrics with RAG: {e}")
        return json.dumps({"error": f"Error extracting metrics: {e}"})
