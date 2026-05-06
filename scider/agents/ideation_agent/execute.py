"""Ideation agent — generates novel research ideas from literature.

Single agent loop: searches papers, identifies gaps, generates ideas.
"""

from loguru import logger

# Ensure paper_subagent agent type is registered
import scider.agents.paper_subagent.build  # noqa: F401
from scider.core.agent_utils import inject_system_reminder, today_part
from scider.core.llms import ModelRegistry
from scider.core.query import QueryResult, gather_tools, query
from scider.core.types import Message
from scider.prompts import PROMPTS

from .idea_search import IdeaSearchResult, run_idea_search
from .state import IdeationAgentState

LLM_NAME = "ideation"
AGENT_NAME = "ideation"

AGENT_TOOLS = [
    "WebSearch",
    "WebFetch",
    "Agent",
    "AskUserQuestion",
    "TodoWrite",
]


def _get_system_prompt() -> str:
    return PROMPTS.ideation_agent.system_prompt.render()


def _inject_system_context(agent_state: IdeationAgentState) -> None:
    from scider.core.utils import detect_python_runtime

    parts = [today_part(), detect_python_runtime()]
    if agent_state.research_domain:
        parts.append(f"research_domain: {agent_state.research_domain}")
    inject_system_reminder(agent_state.history, agent_name=AGENT_NAME, parts=parts)


def agent_loop_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    logger.debug("agent_loop_node of IdeationAgent")
    agent_state.add_node_history("agent_loop")

    _inject_system_context(agent_state)

    system_prompt = _get_system_prompt()
    tools = gather_tools(AGENT_TOOLS)

    result: QueryResult = query(
        model_name=LLM_NAME,
        agent_state=agent_state,
        system_prompt=system_prompt,
        tools=tools,
        agent_name=AGENT_NAME,
    )

    agent_state.intermediate_state.append(
        {
            "node_name": "agent_loop",
            "output": f"Completed in {result.turn_count} turns ({result.stop_reason})",
        }
    )

    return agent_state


def generate_report_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    logger.debug("generate_report_node of IdeationAgent")
    agent_state.add_node_history("generate_report")

    agent_state.add_message(
        Message(
            role="user",
            content=PROMPTS.ideation_agent.summary_user_prompt.render(),
            agent_sender=AGENT_NAME,
            is_meta=True,
        )
    )

    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.messages,
        system_prompt=PROMPTS.ideation_agent.summary_system_prompt.render(),
        agent_sender=AGENT_NAME,
    ).with_log()

    agent_state.add_message(msg)
    agent_state.output_summary = msg.content or ""

    agent_state.intermediate_state.append(
        {
            "node_name": "generate_report",
            "output": agent_state.output_summary[:200] if agent_state.output_summary else "",
        }
    )

    return agent_state


_EXTRACT_IDEAS_PROMPT = """\
Extract ALL research ideas from the following ideation report as a JSON array.
Each idea must have these fields:
- "title": short title (string)
- "description": 2-3 sentence description (string)
- "novelty_score": 0-10 score if mentioned, otherwise null (number or null)

Return ONLY a JSON array, no other text. Example:
[{"title": "...", "description": "...", "novelty_score": 7.5}]

Report:
"""


def extract_ideas_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    """Extract structured ideas from the ideation report."""
    logger.debug("extract_ideas_node of IdeationAgent")
    agent_state.add_node_history("extract_ideas")

    if not agent_state.output_summary:
        return agent_state

    try:
        msg = ModelRegistry.completion(
            LLM_NAME,
            [Message(role="user", content=_EXTRACT_IDEAS_PROMPT + agent_state.output_summary)],
            system_prompt="You extract structured data from text. Return only valid JSON.",
            agent_sender=AGENT_NAME,
        )

        import json

        from json_repair import repair_json

        raw = msg.content or "[]"
        # Strip markdown code fences if present
        if "```" in raw:
            import re

            match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
            if match:
                raw = match.group(1)
        raw = repair_json(raw)
        ideas = json.loads(raw)

        if isinstance(ideas, list):
            agent_state.research_ideas = ideas
            # Extract average novelty score
            scores = [i["novelty_score"] for i in ideas if i.get("novelty_score") is not None]
            if scores:
                agent_state.novelty_score = sum(scores) / len(scores)

        logger.info("Extracted {} research ideas", len(agent_state.research_ideas))

    except Exception as e:
        logger.warning("Failed to extract structured ideas: {}", e)

    agent_state.intermediate_state.append(
        {
            "node_name": "extract_ideas",
            "output": f"Extracted {len(agent_state.research_ideas)} ideas"
            + (
                f", avg novelty: {agent_state.novelty_score:.1f}/10"
                if agent_state.novelty_score
                else ""
            ),
        }
    )

    return agent_state


def idea_search_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    """Run evolutionary idea search on the extracted ideas."""
    logger.debug("idea_search_node of IdeationAgent")
    agent_state.add_node_history("idea_search")

    if not agent_state.idea_search_enabled:
        agent_state.intermediate_state.append(
            {"node_name": "idea_search", "output": "Skipped (idea_search_enabled=False)"}
        )
        return agent_state

    if not agent_state.research_ideas:
        agent_state.intermediate_state.append(
            {"node_name": "idea_search", "output": "Skipped (no ideas extracted)"}
        )
        return agent_state

    result: IdeaSearchResult = run_idea_search(
        seed_ideas=agent_state.research_ideas,
        user_query=agent_state.user_query,
        max_llm_calls=agent_state.max_idea_search_calls,
    )

    # Replace research_ideas with the enriched, ranked output
    agent_state.research_ideas = result.best_ideas
    agent_state.composite_scores = [
        idea.get("composite_score") or 0.0 for idea in result.best_ideas
    ]

    # Update novelty_score to reflect composite for backward-compat display
    if agent_state.composite_scores:
        agent_state.novelty_score = sum(agent_state.composite_scores) / len(
            agent_state.composite_scores
        )

    agent_state.idea_search_result = {
        "llm_calls_used": result.llm_calls_used,
        "iterations_completed": result.iterations_completed,
        "search_budget_hit": result.search_budget_hit,
        "n_ideas": len(result.best_ideas),
    }

    best_score = max(agent_state.composite_scores, default=0.0)
    agent_state.intermediate_state.append(
        {
            "node_name": "idea_search",
            "output": (
                f"Search complete: {result.iterations_completed} iterations, "
                f"{result.llm_calls_used} LLM calls"
                + (" [BUDGET HIT]" if result.search_budget_hit else "")
                + f". Best composite: {best_score:.3f}"
            ),
        }
    )

    return agent_state
