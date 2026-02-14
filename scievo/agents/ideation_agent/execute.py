"""
Execution nodes for the Ideation Agent

This agent generates research ideas through literature review.
Flow: START -> keyword_construct -> literature_search -> analyze_papers -> generate_ideas -> novelty_check -> ideation_report -> END
"""

import json
import re

from loguru import logger
from pydantic import BaseModel, TypeAdapter

from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.core.utils import parse_json_from_text
from scievo.prompts.prompt_data import PROMPTS
from scievo.tools.arxiv_tool import search_papers
from scievo.tools.ideation_tool import analyze_papers_for_ideas
from scievo.tools.registry import ToolRegistry

from .state import IdeationAgentState

LLM_NAME = "ideation"
AGENT_NAME = "ideation"

# Built-in toolsets for ideation agent
BUILTIN_TOOLSETS = ["ideation", "paper_search"]


def keyword_construct_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    """Extract search keywords from user query using LLM."""
    logger.debug("keyword_construct_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("keyword_construct")

    try:
        system_prompt = PROMPTS.ideation.keyword_construct_system_prompt.render()
        user_prompt_content = PROMPTS.ideation.keyword_construct_user_prompt.render(
            user_query=agent_state.user_query,
            research_domain=agent_state.research_domain or "",
        )
        user_prompt = Message(
            role="user",
            content=user_prompt_content,
            agent_sender=AGENT_NAME,
        )
        agent_state.add_message(user_prompt)

        msg = ModelRegistry.completion(
            LLM_NAME,
            agent_state.patched_history,
            system_prompt=system_prompt,
            agent_sender=AGENT_NAME,
            tools=None,
            tool_choice="none",
        ).with_log()

        agent_state.add_message(msg)

        # Parse keywords from LLM response
        if msg.content:
            try:
                KeywordsAdapter = TypeAdapter(list[str])
                KeywordsAdapter.model_validate_json = (
                    lambda json_str: KeywordsAdapter.validate_json(json_str)
                )
                keywords = parse_json_from_text(msg.content, KeywordsAdapter)
                if keywords and len(keywords) > 0:
                    agent_state.search_keywords = keywords
                else:
                    logger.warning("Empty keywords list, falling back to user query")
                    agent_state.search_keywords = [agent_state.user_query]
            except Exception as e:
                logger.warning("Failed to parse keywords: {}", e)
                agent_state.search_keywords = [agent_state.user_query]
        else:
            agent_state.search_keywords = [agent_state.user_query]

        logger.info(
            "Extracted {} search keywords: {}",
            len(agent_state.search_keywords),
            agent_state.search_keywords,
        )

        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Keyword Construction] Extracted {len(agent_state.search_keywords)} keywords: {agent_state.search_keywords}",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    except Exception as e:
        logger.exception("Keyword construction error")
        agent_state.search_keywords = [agent_state.user_query]
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Keyword Construction Error] {str(e)}, falling back to raw query",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    agent_state.intermediate_state.append(
        {
            "node_name": "keyword_construct",
            "output": f"Keywords: {agent_state.search_keywords}",
        }
    )

    logger.info("intermediate_state length: ", len(agent_state.intermediate_state))

    return agent_state


def literature_search_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    """Search for relevant literature on the research topic."""
    logger.debug("literature_search_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("literature_search")

    try:
        # Use extracted keywords for search (fall back to user query if none)
        keywords = agent_state.search_keywords or [agent_state.user_query]
        if agent_state.research_domain:
            keywords = [f"{agent_state.research_domain} {kw}" for kw in keywords]

        # Join all keywords into a single query string
        combined_query = " ".join(keywords)
        logger.info("Searching papers with combined query: '{}'", combined_query)

        result = search_papers(
            query=combined_query,
            sources=["arxiv"],
            max_results=15,
        )

        # Parse the result
        try:
            if isinstance(result, str):
                parsed_result = json.loads(result)
            else:
                parsed_result = result

            # Handle different return formats
            if isinstance(parsed_result, dict):
                if "error" in parsed_result:
                    error_msg = parsed_result.get("error", "Unknown error")
                    logger.warning("Search error: {}", error_msg)
                    agent_state.papers = parsed_result.get("papers", [])
                elif "papers" in parsed_result:
                    agent_state.papers = parsed_result["papers"]
                else:
                    logger.warning(
                        "Unexpected result format from search_papers: {}",
                        list(parsed_result.keys()),
                    )
                    agent_state.papers = []
            elif isinstance(parsed_result, list):
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

        logger.info("Found {} papers using keywords: {}", len(agent_state.papers), keywords)

        # Add search results to history
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Literature Search] Found {len(agent_state.papers)} unique papers using {len(keywords)} keywords: {keywords}",
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

    logger.bind(
        agent=AGENT_NAME,
        node="literature_search",
        papers=len(agent_state.papers),
    ).info("literature_search_node completed")

    search_result_text = f"Found {len(agent_state.papers)} unique papers using keywords: {keywords if 'keywords' in locals() else [agent_state.user_query]}\n\n"
    if agent_state.papers:
        search_result_text += "\n".join(
            [
                f"{i+1}. {p.get('title', 'Unknown')} - {', '.join(p.get('authors', [])[:3])}"
                for i, p in enumerate(agent_state.papers[:10])
            ]
        )

    agent_state.intermediate_state.append(
        {
            "node_name": "literature_search",
            "output": search_result_text,
        }
    )

    logger.info("intermediate_state length: ", len(agent_state.intermediate_state))

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

        logger.info(
            "agent={}, result={}",
            AGENT_NAME,
            result,
        )

        analysis_text = str(result)

        # Parse the result
        try:
            if isinstance(result, str):
                analysis = json.loads(result)
            else:
                analysis = result
            if isinstance(analysis, dict):
                agent_state.analyzed_papers = agent_state.papers
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

    logger.info(
        "agent={}, analyzed_papers={}",
        AGENT_NAME,
        len(agent_state.analyzed_papers),
    )

    analysis_output = analysis_text if "analysis_text" in locals() else "No analysis output"

    agent_state.intermediate_state.append(
        {
            "node_name": "analyze_papers",
            "output": analysis_output,
        }
    )

    logger.info("intermediate_state length: ", len(agent_state.intermediate_state))

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

        logger.info(
            "agent={}, msg={}",
            AGENT_NAME,
            msg.content,
        )

        agent_state.add_message(msg)

        # Log the response for debugging
        if msg.content:
            logger.info("Generated research ideas (content length: {})", len(msg.content))
            logger.debug(f"Ideas content preview: {msg.content}")
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

    ideas_output = ""
    if "msg" in locals() and msg.content:
        ideas_output = msg.content
    elif "msg" in locals() and msg.tool_calls:
        ideas_output = f"[Tool calls made: {len(msg.tool_calls)}]"
    else:
        ideas_output = "[No content generated]"

    agent_state.intermediate_state.append(
        {
            "node_name": "generate_ideas",
            "output": ideas_output,
        }
    )

    # parse research ideas
    class ResearchIdeaModel(BaseModel):
        title: str
        description: str
        rationale: str
        potential_impact: str
        experiment: str
        related_papers: list[str]

    ResearchIdeasModel = TypeAdapter(list[ResearchIdeaModel])
    ResearchIdeasModel.model_validate_json = lambda json_str: ResearchIdeasModel.validate_json(
        json_str
    )

    agent_state.research_ideas = []
    if ideas_output and ideas_output not in ["[No content generated]", "[Tool calls made: 1]"]:
        # Extract the "Proposed Research Ideas" section
        ideas_section_match = re.search(
            r"### Proposed Research Ideas\s*\n(.*?)(?:### |$)", ideas_output, re.DOTALL
        )

        if ideas_section_match:
            ideas_section = ideas_section_match.group(1)
            research_ideas = parse_json_from_text(ideas_section, ResearchIdeasModel)
            research_ideas = [idea.model_dump() for idea in research_ideas]
            agent_state.research_ideas = research_ideas

        logger.info("Parsed {} research ideas", len(agent_state.research_ideas))
    else:
        logger.warning("No valid ideas output to parse")

    logger.info("intermediate_state length: ", len(agent_state.intermediate_state))

    return agent_state


def novelty_check_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    """Check the novelty of each generated research idea and assign per-idea scores (0-10)."""
    logger.debug("novelty_check_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("novelty_check")

    # If no research ideas were parsed, set everything to None and return early
    if not agent_state.research_ideas:
        logger.warning("No research ideas to evaluate for novelty")
        agent_state.idea_novelty_assessments = []
        agent_state.novelty_score = None
        agent_state.novelty_feedback = None

        agent_state.intermediate_state.append(
            {
                "node_name": "novelty_check",
                "output": "No research ideas to evaluate.",
            }
        )
        return agent_state

    # Format papers summary for context (shared across all idea evaluations)
    papers_summary = ""
    if agent_state.papers:
        papers_summary = "\n\n".join(
            [
                f"- {p.get('title', 'Unknown')} ({p.get('published', 'Unknown')})"
                for p in agent_state.papers[:20]
            ]
        )
    else:
        papers_summary = "No papers were reviewed."

    system_prompt = PROMPTS.ideation.novelty_check_system_prompt.render()

    agent_state.idea_novelty_assessments = []

    # Format ALL ideas into a single text block for batch evaluation
    all_ideas_text = ""
    for idx, idea in enumerate(agent_state.research_ideas):
        idea_title = idea.get("title", f"Idea {idx + 1}")
        all_ideas_text += f"### Research Idea (idea_idx={idx}): {idea_title}\n\n"
        for key in ["description", "rationale", "potential_impact", "experiment"]:
            if idea.get(key):
                all_ideas_text += f"**{key.replace('_', ' ').title()}**: {idea[key]}\n\n"
        if idea.get("related_papers"):
            all_ideas_text += (
                f"**Related Papers**: {', '.join(str(p) for p in idea['related_papers'])}\n\n"
            )
        all_ideas_text += "---\n\n"

    logger.info(
        "Evaluating novelty for {} ideas in a single batch",
        len(agent_state.research_ideas),
    )

    # Build prompt for ALL ideas at once
    user_prompt_content = PROMPTS.ideation.novelty_check_user_prompt.render(
        ideas_text=all_ideas_text,
        papers_summary=papers_summary,
    )
    user_prompt = Message(
        role="user",
        content=user_prompt_content,
        agent_sender=AGENT_NAME,
    )
    agent_state.add_message(user_prompt)

    try:
        msg = ModelRegistry.completion(
            LLM_NAME,
            agent_state.patched_history,
            system_prompt=system_prompt,
            agent_sender=AGENT_NAME,
            tools=None,
            tool_choice="none",
        ).with_log()

        agent_state.add_message(msg)

        logger.info(
            "agent={}, node=novelty_check, msg={}",
            AGENT_NAME,
            msg.content,
        )

        # Parse all novelty assessments from the single LLM response
        agent_state.idea_novelty_assessments = _parse_batch_novelty_response(
            msg.content if msg.content else "",
            agent_state.research_ideas,
        )

        for a in agent_state.idea_novelty_assessments:
            a["experiment"] = agent_state.research_ideas[a["idea_idx"]].get("experiment", "N/A")

        for a in agent_state.idea_novelty_assessments:
            logger.info(
                "Novelty for '{}': {:.2f}/10",
                a["title"],
                a["novelty_score"],
            )

    except Exception as e:
        logger.exception("Novelty check error")
        # Fallback: assign default scores to all ideas
        for idx, idea in enumerate(agent_state.research_ideas):
            idea_title = idea.get("title", f"Idea {idx + 1}")
            agent_state.idea_novelty_assessments.append(
                {
                    "idea_idx": idx,
                    "title": idea_title,
                    "novelty_score": 5.0,
                    "feedback": f"Error during novelty check: {str(e)}",
                    "breakdown": None,
                }
            )
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"[Novelty Check Error] {str(e)}",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    # Compute aggregate novelty score (average of all per-idea scores)
    scores = [a["novelty_score"] for a in agent_state.idea_novelty_assessments]
    agent_state.novelty_score = sum(scores) / len(scores) if scores else None

    # Compose aggregate feedback
    feedback_lines = []
    for a in agent_state.idea_novelty_assessments:
        feedback_lines.append(
            f"- **{a['title']}**: {a['novelty_score']:.2f}/10 — {a.get('feedback', 'N/A')}"
        )
    agent_state.novelty_feedback = "\n".join(feedback_lines) if feedback_lines else None

    logger.info(
        "Novelty check completed for {} ideas. Average score: {:.2f}/10",
        len(agent_state.idea_novelty_assessments),
        agent_state.novelty_score if agent_state.novelty_score is not None else 0.0,
    )

    # Build intermediate state output
    novelty_output_parts = [
        (
            f"Evaluated {len(agent_state.idea_novelty_assessments)} ideas.\n"
            f"Average Novelty Score: {agent_state.novelty_score:.2f}/10\n"
            if agent_state.novelty_score is not None
            else "No ideas evaluated.\n"
        )
    ]
    for a in agent_state.idea_novelty_assessments:
        novelty_output_parts.append(
            f"\n### {a['title']}\n"
            f"Score: {a['novelty_score']:.2f}/10\n"
            f"Feedback: {a.get('feedback', 'N/A')}"
        )

    agent_state.intermediate_state.append(
        {
            "node_name": "novelty_check",
            "output": "\n".join(novelty_output_parts),
        }
    )

    logger.info("intermediate_state length: ", len(agent_state.intermediate_state))

    return agent_state


def _parse_batch_novelty_response(content: str, research_ideas: list[dict]) -> list[dict]:
    """Parse a batch novelty check LLM response into a list of structured assessment dicts.

    The LLM is expected to return a JSON array with one assessment per idea.

    Args:
        content: The LLM response text containing a JSON array of assessments.
        research_ideas: The original list of research ideas (for fallback titles).

    Returns:
        list of dicts, each with keys: idea_idx (int), title (str), novelty_score (float), feedback (str), breakdown (dict|None)
    """
    num_ideas = len(research_ideas)

    if not content:
        return [
            {
                "idea_idx": i,
                "title": idea.get("title", f"Idea {i + 1}"),
                "novelty_score": 5.0,
                "feedback": "No assessment was generated.",
                "breakdown": None,
            }
            for i, idea in enumerate(research_ideas)
        ]

    # Define Pydantic models for structured parsing
    class BreakdownModel(BaseModel):
        uniqueness: float | None = None
        innovation: float | None = None
        gap_addressing: float | None = None
        potential_impact: float | None = None

    class NoveltyAssessmentModel(BaseModel):
        idea_idx: int | None = None
        title: str | None = None
        novelty_score: float
        feedback: str | None = None
        comparison_with_literature: str | None = None
        breakdown: BreakdownModel | None = None

    BatchAdapter = TypeAdapter(list[NoveltyAssessmentModel])
    BatchAdapter.model_validate_json = lambda json_str: BatchAdapter.validate_json(json_str)

    def _build_result(assessment: dict, fallback_idx: int, fallback_title: str) -> dict:
        """Convert a parsed assessment dict into the standard result format."""
        score = float(assessment.get("novelty_score", 5.0))
        score = max(0.0, min(10.0, score))

        idea_idx = assessment.get("idea_idx")
        if idea_idx is None:
            idea_idx = fallback_idx

        feedback_parts = []
        if assessment.get("feedback"):
            feedback_parts.append(assessment["feedback"])
        if assessment.get("comparison_with_literature"):
            feedback_parts.append(
                f"Comparison with Literature: {assessment['comparison_with_literature']}"
            )
        feedback = "\n\n".join(feedback_parts) if feedback_parts else "No feedback provided."

        breakdown = assessment.get("breakdown", None)

        return {
            "idea_idx": idea_idx,
            "title": assessment.get("title") or fallback_title,
            "novelty_score": score,
            "feedback": feedback,
            "breakdown": breakdown,
        }

    try:
        # Try parsing as a JSON array via TypeAdapter
        parsed = parse_json_from_text(content, BatchAdapter)

        if parsed and isinstance(parsed, list) and len(parsed) > 0:
            results = []
            for i, assessment_obj in enumerate(parsed):
                # Convert pydantic model to dict if needed
                if hasattr(assessment_obj, "model_dump"):
                    assessment = assessment_obj.model_dump()
                elif isinstance(assessment_obj, dict):
                    assessment = assessment_obj
                else:
                    assessment = dict(assessment_obj)

                fallback_title = (
                    research_ideas[i].get("title", f"Idea {i + 1}")
                    if i < num_ideas
                    else f"Idea {i + 1}"
                )
                results.append(_build_result(assessment, i, fallback_title))

            # If LLM returned fewer assessments than ideas, fill in defaults
            for i in range(len(results), num_ideas):
                idea_title = research_ideas[i].get("title", f"Idea {i + 1}")
                logger.warning("No assessment returned for idea: {}", idea_title)
                results.append(
                    {
                        "idea_idx": i,
                        "title": idea_title,
                        "novelty_score": 5.0,
                        "feedback": "No assessment was returned for this idea.",
                        "breakdown": None,
                    }
                )

            return results

        # Fallback: try parsing as a single object (in case LLM didn't return an array)
        logger.warning(
            "Batch novelty response was not a valid array, attempting single-object fallback"
        )

    except Exception as parse_error:
        logger.warning("Failed to parse batch novelty assessment: {}", parse_error)

    # Fallback: try extracting individual scores via regex
    try:
        score_matches = re.findall(r"novelty[_\s]*score[:\s]*([0-9.]+)", content, re.IGNORECASE)
        if score_matches and len(score_matches) >= num_ideas:
            results = []
            for i, score_str in enumerate(score_matches[:num_ideas]):
                score = max(0.0, min(10.0, float(score_str)))
                idea_title = research_ideas[i].get("title", f"Idea {i + 1}")
                results.append(
                    {
                        "idea_idx": i,
                        "title": idea_title,
                        "novelty_score": score,
                        "feedback": content,
                        "breakdown": None,
                    }
                )
            return results
    except Exception:
        pass

    # Final fallback: return default scores for all ideas
    logger.warning("Could not parse any novelty scores from batch response")
    return [
        {
            "idea_idx": i,
            "title": idea.get("title", f"Idea {i + 1}"),
            "novelty_score": 5.0,
            "feedback": f"Failed to parse assessment.\n\nLLM Response: {content}",
            "breakdown": None,
        }
        for i, idea in enumerate(research_ideas)
    ]


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

        # Include novelty assessment if available
        novelty_section = ""
        if agent_state.idea_novelty_assessments:
            avg_score = agent_state.novelty_score
            novelty_section = f"""
## Novelty Assessment

**Average Novelty Score: {avg_score:.2f}/10** ({len(agent_state.idea_novelty_assessments)} ideas evaluated)

"""
            for a in agent_state.idea_novelty_assessments:
                novelty_section += f"### {a['title']}\n\n"
                novelty_section += f"**Score: {a['novelty_score']:.2f}/10**\n\n"
                if a.get("feedback"):
                    novelty_section += f"{a['feedback']}\n\n"
                if a.get("breakdown"):
                    bd = a["breakdown"]
                    novelty_section += (
                        f"- Uniqueness: {bd.get('uniqueness', 'N/A')}\n"
                        f"- Innovation: {bd.get('innovation', 'N/A')}\n"
                        f"- Gap Addressing: {bd.get('gap_addressing', 'N/A')}\n"
                        f"- Potential Impact: {bd.get('potential_impact', 'N/A')}\n\n"
                    )
                if a.get("experiment"):
                    novelty_section += f"**Suggested Experiment**: {a['experiment']}\n\n"
                novelty_section += "---\n\n"
        elif agent_state.novelty_score is not None:
            novelty_section = f"""
## Novelty Assessment

**Novelty Score: {agent_state.novelty_score:.2f}/10**

{agent_state.novelty_feedback or "No feedback available."}
"""

        report = f"""# Research Ideation Report

## Research Topic
{agent_state.user_query}
{domain_section}## Literature Review
- Papers reviewed: {len(agent_state.papers)}
- Papers analyzed: {len(agent_state.analyzed_papers)}

## Generated Research Ideas

{ideas_text}
{novelty_section}## Summary
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

    report_output = (
        agent_state.output_summary
        if agent_state.output_summary
        else report if "report" in locals() else "No report generated"
    )

    agent_state.intermediate_state.append(
        {
            "node_name": "ideation_report",
            "output": report_output,
        }
    )

    logger.info("intermediate_state length: ", len(agent_state.intermediate_state))

    return agent_state
