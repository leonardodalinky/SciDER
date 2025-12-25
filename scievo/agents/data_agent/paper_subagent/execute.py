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
from scievo.prompts.prompt_data import PROMPTS
from scievo.tools.arxiv_tool import search_papers
from scievo.tools.dataset_search_tool import search_datasets
from scievo.tools.metric_search_tool import extract_metrics_from_papers

from .state import PaperSearchAgentState

LLM_NAME = "paper_search"
AGENT_NAME = "paper_search"

# Minimum thresholds for considering search successful
MIN_PAPERS_THRESHOLD = 3
MIN_DATASETS_THRESHOLD = 2


def optimize_query_node(agent_state: PaperSearchAgentState) -> PaperSearchAgentState:
    """Optimize the search query using LLM to improve search results."""
    logger.debug("optimize_query_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("optimize_query")

    # Initialize current_query if not set
    if agent_state.current_query is None:
        agent_state.current_query = agent_state.user_query
        agent_state.query_history = [agent_state.user_query]

    # If we've already tried multiple queries, use the best one or stop
    if agent_state.search_iteration >= agent_state.max_search_iterations:
        logger.info("Reached max iterations, using current query")
        return agent_state

    # Build optimization prompt
    previous_results = ""
    if agent_state.search_iteration > 0:
        previous_results = f"""
Previous search results:
- Papers found: {len(agent_state.papers)}
- Datasets found: {len(agent_state.datasets)}
- Previous queries tried: {', '.join(agent_state.query_history[-3:])}
"""

    optimization_prompt = f"""You are a research assistant helping to optimize academic paper search queries.

Original user query: "{agent_state.user_query}"
{previous_results}

Your task is to generate an improved search query that is more likely to find relevant academic papers on arXiv.

Guidelines:
1. If previous search found few/no results, make the query MORE GENERAL (remove specific details, use broader terms)
2. If previous search found too many irrelevant results, make the query MORE SPECIFIC (add key terms, use domain-specific vocabulary)
3. Use standard academic terminology and keywords
4. Keep the query concise (2-5 key terms)
5. Consider synonyms and related terms
6. Focus on the core research topic, not implementation details

Generate ONLY the optimized search query (no explanation, just the query text):"""

    try:
        msg = ModelRegistry.completion(
            LLM_NAME,
            [Message(role="user", content=optimization_prompt)],
            system_prompt="You are an expert at crafting effective academic search queries. Return only the optimized query text.",
            agent_sender=AGENT_NAME,
            tools=None,
        )

        optimized_query = msg.content.strip()
        # Remove quotes if present
        optimized_query = optimized_query.strip('"').strip("'").strip()

        if optimized_query and optimized_query != agent_state.current_query:
            agent_state.current_query = optimized_query
            agent_state.query_history.append(optimized_query)
            logger.info(
                f"Optimized query (iteration {agent_state.search_iteration + 1}): {optimized_query}"
            )

            agent_state.add_message(
                Message(
                    role="assistant",
                    content=f"[Query Optimization] Optimized search query: '{optimized_query}'",
                    agent_sender=AGENT_NAME,
                ).with_log()
            )
        else:
            logger.info("Query optimization did not produce a new query, using current query")

    except Exception as e:
        logger.exception("Query optimization error")
        # Continue with current query if optimization fails
        if not agent_state.current_query:
            agent_state.current_query = agent_state.user_query

    return agent_state


def check_results_node(agent_state: PaperSearchAgentState) -> PaperSearchAgentState:
    """Check if paper search results are sufficient, decide whether to iterate."""
    logger.debug("check_results_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("check_results")

    papers_count = len(agent_state.papers)

    # Check if we have sufficient papers
    has_sufficient_papers = papers_count >= MIN_PAPERS_THRESHOLD

    # Decision: continue if we don't have enough papers and haven't exceeded max iterations
    should_continue = (
        not has_sufficient_papers
        and agent_state.search_iteration < agent_state.max_search_iterations
    )

    logger.info(
        f"Results check: {papers_count} papers found. "
        f"Sufficient: {has_sufficient_papers} (threshold: {MIN_PAPERS_THRESHOLD}). "
        f"Should continue: {should_continue} (iteration {agent_state.search_iteration}/{agent_state.max_search_iterations})"
    )

    # Store decision in state (we'll use this in conditional edge)
    agent_state.add_message(
        Message(
            role="assistant",
            content=f"[Results Check] Found {papers_count} papers. "
            f"{'Continuing search iteration' if should_continue else 'Proceeding with current results'}.",
            agent_sender=AGENT_NAME,
        ).with_log()
    )

    return agent_state


def should_continue_search(agent_state: PaperSearchAgentState) -> str:
    """Conditional function to decide whether to continue searching or proceed.

    Only iterates on paper search. Dataset search happens once after paper search is done.
    """
    papers_count = len(agent_state.papers)

    has_sufficient_papers = papers_count >= MIN_PAPERS_THRESHOLD

    should_continue = (
        not has_sufficient_papers
        and agent_state.search_iteration < agent_state.max_search_iterations
    )

    return "continue_search" if should_continue else "proceed"


def search_node(agent_state: PaperSearchAgentState) -> PaperSearchAgentState:
    """Execute paper search using the search_papers tool."""
    logger.debug("search_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("search")

    # Increment iteration count
    agent_state.search_iteration += 1

    # Use current_query if available, otherwise use user_query
    query_to_use = agent_state.current_query or agent_state.user_query

    try:
        # Call the search_papers tool directly
        # Use only arxiv by default to avoid rate limiting issues with Semantic Scholar
        # Semantic Scholar has strict rate limits (429 errors)
        result = search_papers(
            query=query_to_use,
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
                content=f"[Search Results] Found {len(agent_state.papers)} papers for query: '{query_to_use}' (iteration {agent_state.search_iteration})",
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
        # Use current_query if available, otherwise use user_query
        query_to_use = agent_state.current_query or agent_state.user_query

        # Pass data_summary if available to search for similar datasets
        result = search_datasets(
            query=query_to_use,
            sources=["paperswithcode", "huggingface"],  # Default sources
            max_results=10,
            data_summary=agent_state.data_summary,  # Pass data analysis summary
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
        # Extract metrics even if we don't have papers (fallback to common metrics)
        if not agent_state.papers:
            logger.info("No papers available for metric extraction, using fallback")
            # Still call the tool - it has fallback logic to suggest common metrics
            result = extract_metrics_from_papers(
                papers=[],  # Empty list triggers fallback
                task_query=agent_state.user_query,
                max_results=20,
            )
        else:
            # Call the extract_metrics_from_papers tool with actual papers
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

    # Format datasets for summary (more detailed)
    datasets_text = ""
    if agent_state.datasets:
        datasets_text = "\n\n".join(
            [
                f"**Dataset {i+1}: {d.get('name', 'N/A')}**\n"
                f"- **Source**: {d.get('source', 'N/A')}\n"
                f"- **Description**: {d.get('description', 'N/A')[:500]}{'...' if len(d.get('description', '')) > 500 else ''}\n"
                f"- **Domain**: {d.get('domain', 'N/A')}\n"
                f"- **Size**: {d.get('size', 'N/A')}\n"
                f"- **URL**: {d.get('url', 'N/A')}\n"
                f"- **Download URL**: {d.get('download_url', 'N/A') if d.get('download_url') else 'N/A'}\n"
                f"- **License**: {d.get('license', 'N/A') if d.get('license') else 'Not specified'}\n"
                f"- **Paper URL**: {d.get('paper_url', 'N/A') if d.get('paper_url') else 'N/A'}"
                for i, d in enumerate(agent_state.datasets[:15])  # Show more datasets
            ]
        )
    else:
        datasets_text = "No datasets found."

    # Format metrics for summary (more detailed with formulas)
    metrics_text = ""
    if agent_state.metrics:
        metrics_text = "\n\n".join(
            [
                f"**Metric {i+1}: {m.get('name', 'N/A')}**\n"
                f"- **Description**: {m.get('description', 'N/A')}\n"
                f"- **Domain**: {m.get('domain', 'N/A')}\n"
                f"- **Source Paper**: {m.get('paper_title', 'N/A')}\n"
                f"- **Paper URL**: {m.get('paper_url', 'N/A') if m.get('paper_url') else 'N/A'}\n"
                f"- **Reported Value**: {m.get('value', 'N/A') if m.get('value') else 'Not specified'}\n"
                f"- **Formula**: {m.get('formula', 'N/A') if m.get('formula') else 'Not provided'}"
                for i, m in enumerate(agent_state.metrics[:20])  # Show more metrics
            ]
        )
    else:
        metrics_text = "No metrics extracted."

    # Render summary prompt from template
    summary_prompt_content = PROMPTS.paper_subagent.summary_prompt.render(
        user_query=agent_state.user_query,
        papers_text=papers_text,
        datasets_text=datasets_text,
        metrics_text=metrics_text,
    )
    summary_prompt = Message(
        role="user",
        content=summary_prompt_content,
        agent_sender=AGENT_NAME,
    )
    agent_state.add_message(summary_prompt)

    # Get summary from LLM
    system_prompt = PROMPTS.paper_subagent.summary_system_prompt.render()
    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.patched_history,
        system_prompt=system_prompt,
        agent_sender=AGENT_NAME,
        tools=None,  # No tools needed for summary
    ).with_log()

    # Store the summary text
    agent_state.output_summary = msg.content or ""
    agent_state.add_message(msg)

    logger.info(f"Summary generated: {len(agent_state.output_summary)} characters")

    return agent_state
