"""
Execution nodes for the Ideation Agent

This agent generates research ideas through literature review.
Flow: START -> literature_search -> analyze_papers -> generate_ideas -> ideation_report -> END
"""

from loguru import logger

from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.core.utils import unwrap_dict_from_toon
from scievo.prompts.prompt_data import PROMPTS
from scievo.tools.arxiv_tool import search_papers
from scievo.tools.ideation_tool import analyze_papers_for_ideas
from scievo.tools.registry import ToolRegistry

from .state import IdeationAgentState

LLM_NAME = "ideation"
AGENT_NAME = "ideation"

# Built-in toolsets for ideation agent
BUILTIN_TOOLSETS = ["ideation", "paper_search"]


def literature_search_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    """Search for relevant literature on the research topic."""
    logger.debug("literature_search_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("literature_search")

    try:
        # Use the search_papers tool directly from arxiv_tool
        query = agent_state.user_query
        if agent_state.research_domain:
            query = f"{agent_state.research_domain} {query}"

        result = search_papers(
            query=query,
            sources=["arxiv"],  # Start with arXiv, can expand later
            max_results=15,  # Get more papers for ideation
        )

        # Parse the result (always TOON format now)
        try:
            parsed_result = unwrap_dict_from_toon(result)

            # Handle different return formats
            if isinstance(parsed_result, dict):
                # Check if it's an error response
                if "error" in parsed_result:
                    error_msg = parsed_result.get("error", "Unknown error")
                    logger.warning("Search error: {}", error_msg)
                    agent_state.papers = parsed_result.get("papers", [])
                    # Add error message to history
                    agent_state.add_message(
                        Message(
                            role="assistant",
                            content=f"[Search Warning] {error_msg}",
                            agent_sender=AGENT_NAME,
                        ).with_log()
                    )
                # Check if it has 'papers' field (shouldn't happen, but be defensive)
                elif "papers" in parsed_result:
                    agent_state.papers = parsed_result["papers"]
                # Otherwise, treat the whole dict as unexpected format
                else:
                    logger.warning(
                        "Unexpected result format from search_papers: {}",
                        list(parsed_result.keys()),
                    )
                    agent_state.papers = []
            elif isinstance(parsed_result, list):
                # Direct list of papers (normal case)
                agent_state.papers = parsed_result
            else:
                logger.warning("Unexpected result type from search_papers: {}", type(parsed_result))
                agent_state.papers = []
        except Exception as parse_error:
            logger.warning("Failed to parse search results: {}", parse_error)
            logger.debug(
                "Result content preview: {}",
                result[:500] if isinstance(result, str) else str(result)[:500],
            )
            agent_state.papers = []

        logger.info("Found {} papers", len(agent_state.papers))

        # Add search results to history
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Literature Search] Found {len(agent_state.papers)} papers for query: '{query}'",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

        # Note: We don't fetch abstracts - only use title, authors, and metadata for ideation

    except Exception as e:
        logger.exception("Literature search error")
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Search Error] {str(e)}",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    return agent_state


def analyze_papers_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    """Analyze papers to identify research gaps and opportunities."""
    logger.debug("analyze_papers_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("analyze_papers")

    try:
        if not agent_state.papers:
            logger.warning("No papers to analyze")
            agent_state.add_message(
                Message(
                    role="assistant",
                    content="[Analysis] No papers found to analyze. Please refine your search query.",
                    agent_sender=AGENT_NAME,
                ).with_log()
            )
            return agent_state

        # Use the analyze_papers_for_ideas tool
        research_domain = agent_state.research_domain or agent_state.user_query
        result = analyze_papers_for_ideas(
            papers=agent_state.papers,
            research_domain=research_domain,
        )

        # Parse the result
        try:
            analysis = unwrap_dict_from_toon(result)
            if isinstance(analysis, dict):
                agent_state.analyzed_papers = agent_state.papers  # Mark as analyzed
                logger.info("Analyzed {} papers", len(agent_state.papers))
            else:
                logger.warning("Unexpected result format from analyze_papers_for_ideas")
        except Exception as parse_error:
            logger.warning("Failed to parse analysis results: {}", parse_error)

        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Paper Analysis] Analyzed {len(agent_state.papers)} papers in domain: {research_domain}",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    except Exception as e:
        logger.exception("Paper analysis error")
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Analysis Error] {str(e)}",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    return agent_state


def generate_ideas_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    """Generate research ideas using LLM based on analyzed papers."""
    logger.debug("generate_ideas_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("generate_ideas")

    try:
        # Format papers for prompt (without abstracts)
        papers_text = ""
        if agent_state.papers:
            papers_text = "\n\n".join(
                [
                    f"**{i+1}. {p.get('title', 'N/A')}**\n"
                    f"- Authors: {', '.join(p.get('authors', [])[:5])}\n"
                    f"- Published: {p.get('published', 'N/A')}\n"
                    f"- URL: {p.get('url', 'N/A')}"
                    for i, p in enumerate(agent_state.papers[:15])  # Top 15 papers
                ]
            )
        else:
            papers_text = "No papers found."

        # Build prompt from template
        user_prompt_content = PROMPTS.ideation.user_prompt.render(
            user_query=agent_state.user_query,
            papers=papers_text,
            research_domain=agent_state.research_domain or "",
        )
        user_prompt = Message(
            role="user",
            content=user_prompt_content,
            agent_sender=AGENT_NAME,
        )
        agent_state.add_message(user_prompt)

        # Get system prompt
        system_prompt = PROMPTS.ideation.system_prompt.render()

        # Generate ideas using LLM
        # Note: We don't pass tools here because at this stage, we want the LLM
        # to directly generate ideas as text, not call any tools
        # The literature search has already been completed in previous nodes
        msg = ModelRegistry.completion(
            LLM_NAME,
            agent_state.patched_history,
            system_prompt=system_prompt,
            agent_sender=AGENT_NAME,
            tools=None,  # No tools - just generate text ideas
            tool_choice="none",  # Explicitly disable tool calls
        ).with_log()

        agent_state.add_message(msg)

        # Log the response for debugging
        if msg.content:
            logger.info("Generated research ideas (content length: {})", len(msg.content))
            logger.debug("Ideas content preview: {}", msg.content[:200])
        elif msg.tool_calls:
            logger.warning("LLM returned tool calls instead of ideas: {}", len(msg.tool_calls))
        else:
            logger.warning("LLM returned empty response (no content, no tool calls)")

    except Exception as e:
        logger.exception("Idea generation error")
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Idea Generation Error] {str(e)}",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    return agent_state


def ideation_report_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    """Generate final ideation report summarizing research ideas."""
    logger.debug("ideation_report_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("ideation_report")

    try:
        # Extract the ideas from the conversation history
        # Look for the assistant message from generate_ideas_node
        # We need to find the message that was added after the user prompt in generate_ideas_node
        ideas_text = ""

        # Search through history in reverse to find the most recent assistant message
        # that was generated by generate_ideas_node (should be after "generate_ideas" in node_history)
        found_generate_ideas = False
        for msg in reversed(agent_state.history):
            if msg.role == "assistant" and msg.agent_sender == AGENT_NAME:
                # Check if this message is from generate_ideas_node
                # The message should be after "generate_ideas" appears in node_history
                if "generate_ideas" in agent_state.node_history:
                    # Get the content, handling None case
                    if msg.content:
                        ideas_text = msg.content
                    elif msg.tool_calls:
                        # If only tool calls, format them
                        ideas_text = f"[Tool calls made: {len(msg.tool_calls)}]"
                    else:
                        ideas_text = "[No content generated]"
                    found_generate_ideas = True
                    break

        # Fallback: if not found, try patched_history
        if not found_generate_ideas or not ideas_text:
            for msg in reversed(agent_state.patched_history):
                if msg.role == "assistant" and msg.agent_sender == AGENT_NAME:
                    if msg.content:
                        ideas_text = msg.content
                        break
                    elif msg.tool_calls:
                        ideas_text = f"[Tool calls made: {len(msg.tool_calls)}]"
                        break

        # If still no content, provide a default message
        if not ideas_text:
            ideas_text = (
                "No research ideas were generated. Please check the LLM response or try again."
            )

        # Create a summary report
        domain_section = ""
        if agent_state.research_domain:
            domain_section = f"\n## Research Domain\n{agent_state.research_domain}\n"

        report = f"""# Research Ideation Report

## Research Topic
{agent_state.user_query}
{domain_section}## Literature Review
- Papers reviewed: {len(agent_state.papers)}
- Papers analyzed: {len(agent_state.analyzed_papers)}

## Generated Research Ideas

{ideas_text}

## Summary
This ideation report was generated through literature review of {len(agent_state.papers)} academic papers.
The research ideas presented above are based on analysis of current research gaps and opportunities.
"""

        agent_state.output_summary = report
        agent_state.add_message(
            Message(
                role="assistant",
                content=report,
                agent_sender=AGENT_NAME,
            ).with_log()
        )

        logger.info(
            "Generated ideation report with ideas: {}", ideas_text[:100] if ideas_text else "None"
        )

    except Exception as e:
        logger.exception("Report generation error")
        agent_state.output_summary = f"Error generating report: {str(e)}"
        agent_state.add_message(
            Message(
                role="assistant",
                content=agent_state.output_summary,
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    return agent_state
