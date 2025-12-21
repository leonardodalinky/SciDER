"""
Execution nodes for the Paper Search Agent

This module provides a minimal execution flow that searches for papers, datasets,
extracts metrics, and generates a summary.
Flow: START -> search_node -> dataset_node -> metric_node -> summary_node -> END
"""

from loguru import logger

from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.core.utils import unwrap_dict_from_toon
from scievo.tools.arxiv_tool import search_papers
from scievo.tools.dataset_search_tool import search_datasets
from scievo.tools.metric_search_tool import extract_metrics_from_papers

from .state import PaperSearchAgentState

LLM_NAME = "paper_search"
AGENT_NAME = "paper_search"


def search_node(agent_state: PaperSearchAgentState) -> PaperSearchAgentState:
    """Execute paper search using the search_papers tool."""
    logger.debug("search_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("search")

    try:
        # Call the search_papers tool directly
        # Use only arxiv by default to avoid rate limiting issues with Semantic Scholar
        # Semantic Scholar has strict rate limits (429 errors)
        result = search_papers(
            query=agent_state.user_query,
            sources=["arxiv"],  # Use arxiv only to avoid rate limiting
            max_results=10,
        )

        # Parse the result (tool returns TOON format)
        try:
            papers = unwrap_dict_from_toon(result)
            if isinstance(papers, list):
                agent_state.papers = papers
            else:
                logger.warning("Unexpected result format from search_papers")
                agent_state.papers = []
        except Exception as parse_error:
            logger.warning("Failed to parse search results: {}", parse_error)
            agent_state.papers = []

        logger.info("Found {} papers", len(agent_state.papers))

        # Add search results to history
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Search Results] Found {len(agent_state.papers)} papers for query: '{agent_state.user_query}'",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    except Exception as e:
        logger.exception("Paper search error")
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Search Error] {str(e)}",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    return agent_state


def dataset_node(agent_state: PaperSearchAgentState) -> PaperSearchAgentState:
    """Execute dataset search using the search_datasets tool."""
    logger.debug("dataset_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("dataset")

    try:
        # Call the search_datasets tool directly
        # Use the same query as paper search
        result = search_datasets(
            query=agent_state.user_query,
            sources=["paperswithcode", "huggingface"],  # Default sources
            max_results=10,
        )

        # Parse the result (tool returns TOON format)
        try:
            datasets = unwrap_dict_from_toon(result)
            if isinstance(datasets, list):
                agent_state.datasets = datasets
            else:
                logger.warning("Unexpected result format from search_datasets")
                agent_state.datasets = []
        except Exception as parse_error:
            logger.warning("Failed to parse dataset search results: {}", parse_error)
            agent_state.datasets = []

        logger.info("Found {} datasets", len(agent_state.datasets))

        # Add search results to history
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Dataset Search Results] Found {len(agent_state.datasets)} datasets for query: '{agent_state.user_query}'",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    except Exception as e:
        logger.exception("Dataset search error")
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Dataset Search Error] {str(e)}",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    return agent_state


def metric_node(agent_state: PaperSearchAgentState) -> PaperSearchAgentState:
    """Extract evaluation metrics from the searched papers."""
    logger.debug("metric_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("metric")

    try:
        # Only extract metrics if we have papers
        if not agent_state.papers:
            logger.info("No papers available for metric extraction")
            agent_state.metrics = []
            agent_state.add_message(
                Message(
                    role="assistant",
                    content="[Metric Extraction] No papers available for metric extraction.",
                    agent_sender=AGENT_NAME,
                ).with_log()
            )
            return agent_state

        # Call the extract_metrics_from_papers tool
        result = extract_metrics_from_papers(
            papers=agent_state.papers,
            task_query=agent_state.user_query,
            max_results=20,
        )

        # Parse the result (tool returns TOON format)
        try:
            metrics = unwrap_dict_from_toon(result)
            if isinstance(metrics, list):
                agent_state.metrics = metrics
            else:
                logger.warning("Unexpected result format from extract_metrics_from_papers")
                agent_state.metrics = []
        except Exception as parse_error:
            logger.warning("Failed to parse metric extraction results: {}", parse_error)
            agent_state.metrics = []

        logger.info("Extracted {} metrics", len(agent_state.metrics))

        # Add extraction results to history
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Metric Extraction Results] Extracted {len(agent_state.metrics)} evaluation metrics from {len(agent_state.papers)} papers.",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    except Exception as e:
        logger.exception("Metric extraction error")
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Metric Extraction Error] {str(e)}",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    return agent_state


def summary_node(agent_state: PaperSearchAgentState) -> PaperSearchAgentState:
    """Generate summary of search results."""
    logger.debug("summary_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("summary")

    # Build summary prompt with paper, dataset, and metric details
    if not agent_state.papers and not agent_state.datasets and not agent_state.metrics:
        agent_state.output_summary = (
            f"No papers, datasets, or metrics found for query: '{agent_state.user_query}'"
        )
        agent_state.add_message(
            Message(
                role="assistant",
                content=agent_state.output_summary,
                agent_sender=AGENT_NAME,
            ).with_log()
        )
        return agent_state

    # Format papers for summary
    papers_text = ""
    if agent_state.papers:
        papers_text = "\n\n".join(
            [
                f"**{i+1}. {p.get('title', 'N/A')}**\n"
                f"- Authors: {', '.join(p.get('authors', [])[:5])}{'...' if len(p.get('authors', [])) > 5 else ''}\n"
                f"- Published: {p.get('published', 'N/A')}\n"
                f"- Source: {p.get('source', 'N/A')}\n"
                f"- Summary: {p.get('summary', 'N/A')[:300]}...\n"
                f"- URL: {p.get('url', 'N/A')}"
                for i, p in enumerate(agent_state.papers[:10])
            ]
        )
    else:
        papers_text = "No papers found."

    # Format datasets for summary
    datasets_text = ""
    if agent_state.datasets:
        datasets_text = "\n\n".join(
            [
                f"**{i+1}. {d.get('name', 'N/A')}**\n"
                f"- Description: {d.get('description', 'N/A')[:300]}...\n"
                f"- Domain: {d.get('domain', 'N/A')}\n"
                f"- Size: {d.get('size', 'N/A')}\n"
                f"- Source: {d.get('source', 'N/A')}\n"
                f"- URL: {d.get('url', 'N/A')}\n"
                f"- License: {d.get('license', 'N/A')}"
                for i, d in enumerate(agent_state.datasets[:10])
            ]
        )
    else:
        datasets_text = "No datasets found."

    # Format metrics for summary
    metrics_text = ""
    if agent_state.metrics:
        metrics_text = "\n\n".join(
            [
                f"**{i+1}. {m.get('name', 'N/A')}**\n"
                f"- Description: {m.get('description', 'N/A')}\n"
                f"- Domain: {m.get('domain', 'N/A')}\n"
                f"- Paper: {m.get('paper_title', 'N/A')}\n"
                f"- Value: {m.get('value', 'N/A') if m.get('value') else 'Not specified'}\n"
                f"- Formula: {m.get('formula', 'N/A') if m.get('formula') else 'Not provided'}"
                for i, m in enumerate(agent_state.metrics[:15])
            ]
        )
    else:
        metrics_text = "No metrics extracted."

    summary_prompt = Message(
        role="user",
        content=f"""Summarize the following search results for the query: "{agent_state.user_query}"

Papers Found:
{papers_text}

Datasets Found:
{datasets_text}

Evaluation Metrics Extracted:
{metrics_text}

Provide a concise summary highlighting:
1. Key papers and their main contributions
2. Relevant datasets and their characteristics
3. Evaluation metrics commonly used in this research area
4. Common themes or trends across the papers
5. Notable authors or institutions
6. Any gaps or areas for further research
7. Recommendations for datasets and metrics that could be used for similar tasks
""",
        agent_sender=AGENT_NAME,
    )
    agent_state.add_message(summary_prompt)

    # Get summary from LLM
    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.patched_history,
        system_prompt="You are a research assistant that summarizes academic paper search results. Provide clear, structured summaries that help researchers understand the current state of research in the queried area.",
        agent_sender=AGENT_NAME,
        tools=None,  # No tools needed for summary
    ).with_log()

    # Store the summary text
    agent_state.output_summary = msg.content or ""
    agent_state.add_message(msg)

    logger.info(f"Summary generated: {len(agent_state.output_summary)} characters")

    return agent_state
