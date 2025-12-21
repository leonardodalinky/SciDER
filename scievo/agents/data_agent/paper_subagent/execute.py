"""
Execution nodes for the Paper Search Agent

This module provides a minimal execution flow that searches for papers
and generates a summary. Flow: START -> search_node -> summary_node -> END
"""

from loguru import logger

from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.core.utils import unwrap_dict_from_toon
from scievo.tools.arxiv_tool import search_papers

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


def summary_node(agent_state: PaperSearchAgentState) -> PaperSearchAgentState:
    """Generate summary of search results."""
    logger.debug("summary_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("summary")

    # Build summary prompt with paper details
    if not agent_state.papers:
        agent_state.output_summary = f"No papers found for query: '{agent_state.user_query}'"
        agent_state.add_message(
            Message(
                role="assistant",
                content=agent_state.output_summary,
                agent_sender=AGENT_NAME,
            ).with_log()
        )
        return agent_state

    # Format papers for summary
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

    summary_prompt = Message(
        role="user",
        content=f"""Summarize the following paper search results for the query: "{agent_state.user_query}"

Search Results:
{papers_text}

Provide a concise summary highlighting:
1. Key papers and their main contributions
2. Common themes or trends across the papers
3. Notable authors or institutions
4. Any gaps or areas for further research
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
