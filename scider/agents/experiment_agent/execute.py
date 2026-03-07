"""
Execution nodes for the Experiment Agent.
"""

import json
import os
from typing import Literal

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from scider.core import constant
from scider.core.llms import ModelRegistry
from scider.core.types import Message
from scider.core.utils import parse_json_from_llm_response
from scider.prompts import PROMPTS

from .exec_subagent import build as exec_build
from .exec_subagent.state import ExecAgentState
from .state import ExperimentAgentState
from .summary_subagent import build as summary_build
from .summary_subagent.state import SummaryAgentState

AGENT_NAME = "experiment_agent"
LLM_NAME = "experiment_agent"

load_dotenv()
CODING_AGENT_VERSION = os.getenv("CODING_AGENT_VERSION", "v3")  # default to Claude (v3)
_OPENHANDS_ENABLED = os.getenv("SCIDER_ENABLE_OPENHANDS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
match CODING_AGENT_VERSION:
    case "v2":
        if not _OPENHANDS_ENABLED:
            raise RuntimeError(
                "CODING_AGENT_VERSION=v2 requires OpenHands, but OpenHands is disabled.\n"
                "Hint: set `CODING_AGENT_VERSION=v3` to use the Claude coding agent, or enable OpenHands with "
                "`SCIDER_ENABLE_OPENHANDS=1`."
            )
        from .coding_subagent_v2 import build as coding_build
        from .coding_subagent_v2.state import CodingAgentState
    case "v3":
        from .coding_subagent_v3_claude import build as coding_build
        from .coding_subagent_v3_claude.state import CodingAgentState
    case _:
        raise ValueError(f"Unsupported CODING_AGENT_VERSION: {CODING_AGENT_VERSION}")

# Compile sub-agent graphs as global variables
coding_graph = coding_build().compile()
exec_graph = exec_build().compile()
summary_graph = summary_build().compile()


def init_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    """Initialize the experiment agent.

    Prepares the initial context message with data summary and user query.
    The repo_source will be passed to the coding subagent which handles
    git cloning and workspace setup.
    """
    logger.info("Initializing Experiment Agent")
    agent_state.current_phase = "init"

    # Add initial message to history
    init_msg = Message(
        role="user",
        content=PROMPTS.experiment_agent.init_prompt.render(
            data_summary=agent_state.data_summary,
            user_query=agent_state.user_query,
            repo_source=agent_state.repo_source or "Not specified",
        ),
        agent_sender=AGENT_NAME,
    ).with_log()
    agent_state.add_message(init_msg)

    agent_state.intermediate_state.append(
        {
            "node_name": "init",
            "output": init_msg.content,
        }
    )

    return agent_state


def run_coding_subagent(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    """Run the Coding Subagent (stateless invocation).

    The coding subagent receives repo_source and handles git cloning
    and workspace setup internally. By default this uses the Claude Agent SDK/Claude Code path (v3).
    """
    logger.info(f"Running Coding Subagent (revision {agent_state.current_revision})")
    agent_state.current_phase = "coding"

    # Build revision feedback context if available
    revision_feedback_list = []
    if agent_state.revision_summaries:
        for i, summary in enumerate(agent_state.revision_summaries):
            revision_feedback_list.append({"revision_number": i + 1, "summary": summary})

    # Collect all previous coding summaries
    previous_coding_summaries = []
    for i, loop in enumerate(agent_state.loop_results):
        prev_summary = loop.get("coding_summary", "")
        if prev_summary:
            previous_coding_summaries.append({"revision": i, "summary": prev_summary})

    # Also include accumulated analysis
    revision_analysis_text = (
        agent_state.revision_analysis
        if agent_state.revision_analysis
        else "No previous analysis yet."
    )

    # Build user query using prompt template
    coding_query = PROMPTS.experiment_agent.coding_subagent_query_prompt.render(
        user_query=agent_state.user_query,
        repo_source=agent_state.repo_source or "Not specified",
        # TODO: limit to last revision and coding summary for now
        revision_feedback_list=revision_feedback_list[-1:],
        previous_coding_summaries=previous_coding_summaries[-1:],
        revision_analysis=revision_analysis_text,
        current_revision=agent_state.current_revision,
    )

    coding_state = CodingAgentState(
        data_summary=agent_state.data_summary,  # Keep data_summary separate
        user_query=coding_query,
        workspace=agent_state.workspace,
    )

    # Invoke coding subagent (stateless call)
    result_state = coding_graph.invoke(coding_state)

    # Extract only needed data from result - don't store full state (graph.invoke returns dict)
    agent_state.history = result_state["history"]  # Merge back history

    # Store coding summary for this loop (for later analysis)
    # Use .get() for safe access in case output_summary is not set
    coding_summary = result_state.get("output_summary") or "No summary available"

    if (
        not agent_state.loop_results
        or agent_state.loop_results[-1].get("revision") != agent_state.current_revision
    ):
        agent_state.loop_results.append(
            {
                "revision": agent_state.current_revision,
                "coding_summary": coding_summary,
            }
        )
    else:
        agent_state.loop_results[-1]["coding_summary"] = coding_summary

    coding_output = coding_summary
    if isinstance(result_state, dict) and "intermediate_state" in result_state:
        coding_intermediate = result_state.get("intermediate_state", [])
        if coding_intermediate:
            coding_output = (
                f"{coding_summary}\n\n[Coding Subagent Intermediate States]\n"
                + "\n".join(
                    [
                        f"{item.get('node_name', 'unknown')}: {item.get('output', '')[:200]}"
                        for item in coding_intermediate
                    ]
                )
            )

    agent_state.intermediate_state.append(
        {
            "node_name": "run_coding_subagent",
            "output": coding_output,
        }
    )

    return agent_state


def run_exec_subagent(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    """Run the Exec Subagent (stateless invocation).

    The workspace path should be extracted from the conversation history
    left by the coding subagent.
    """
    logger.info(f"Running Exec Subagent (revision {agent_state.current_revision})")
    agent_state.current_phase = "exec"

    # Collect all coding summaries from loop results
    coding_summaries = [
        loop.get("coding_summary", "")
        for loop in agent_state.loop_results
        if loop.get("coding_summary")
    ]

    exec_state = ExecAgentState(
        user_query="Run the modified code/experiments and verify the output.",
        workspace=agent_state.workspace,
        coding_summaries=coding_summaries if coding_summaries else None,
        toolsets=["exec"],
    )

    # Invoke exec subagent (stateless call)
    result_state = exec_graph.invoke(exec_state)

    # Extract only needed data from result - don't store full state (graph.invoke returns dict)
    agent_state.history = result_state["history"]
    agent_state.all_execution_results.append(result_state["execution_summary_dict"])

    # Store exec results for this loop
    if (
        agent_state.loop_results
        and agent_state.loop_results[-1].get("revision") == agent_state.current_revision
    ):
        agent_state.loop_results[-1]["exec_result"] = result_state["execution_summary_dict"]

    exec_output = json.dumps(result_state.get("execution_summary_dict", {}), indent=2)
    if isinstance(result_state, dict) and "intermediate_state" in result_state:
        exec_intermediate = result_state.get("intermediate_state", [])
        if exec_intermediate:
            exec_output = f"{exec_output}\n\n[Exec Subagent Intermediate States]\n" + "\n".join(
                [
                    f"{item.get('node_name', 'unknown')}: {item.get('output', '')[:200]}"
                    for item in exec_intermediate
                ]
            )

    agent_state.intermediate_state.append(
        {
            "node_name": "run_exec_subagent",
            "output": exec_output,
        }
    )

    return agent_state


def run_summary_subagent(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    """Run the Summary Subagent (stateless invocation).

    The workspace path should be extracted from the conversation history.
    """
    logger.info(f"Running Summary Subagent (revision {agent_state.current_revision})")
    agent_state.current_phase = "summary"

    summary_state = SummaryAgentState(
        workspace=agent_state.workspace,
        history=agent_state.history.copy(),
        output_path=None,  # Or specify a path for saving
        toolsets=["fs"],
    )

    # Invoke summary subagent (stateless call)
    result_state = summary_graph.invoke(summary_state)

    # Extract only needed data from result - don't store full state (graph.invoke returns dict)
    agent_state.history = result_state["history"]
    agent_state.revision_summaries.append(result_state["summary_text"])

    # Store summary for this loop
    if (
        agent_state.loop_results
        and agent_state.loop_results[-1].get("revision") == agent_state.current_revision
    ):
        agent_state.loop_results[-1]["summary"] = result_state["summary_text"]

    summary_output = result_state.get("summary_text", "No summary generated")
    if isinstance(result_state, dict) and "intermediate_state" in result_state:
        summary_intermediate = result_state.get("intermediate_state", [])
        if summary_intermediate:
            summary_output = (
                f"{summary_output}\n\n[Summary Subagent Intermediate States]\n"
                + "\n".join(
                    [
                        f"{item.get('node_name', 'unknown')}: {item.get('output', '')[:200]}"
                        for item in summary_intermediate
                    ]
                )
            )

    agent_state.intermediate_state.append(
        {
            "node_name": "run_summary_subagent",
            "output": summary_output,
        }
    )

    return agent_state


def analysis_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    """Analyze the current loop results and generate insights.

    This node uses an LLM to analyze what went wrong, what succeeded,
    and what needs improvement. The analysis is accumulated across revisions.
    """
    logger.info(f"Analyzing loop results for revision {agent_state.current_revision}")
    agent_state.current_phase = "analysis"

    # Get current loop results
    current_loop = agent_state.loop_results[-1] if agent_state.loop_results else {}

    # Use LLM to analyze the loop
    analysis_prompt = PROMPTS.experiment_agent.analysis_prompt.render(
        revision_number=agent_state.current_revision,
        coding_summary=current_loop.get("coding_summary", "No coding summary available"),
        exec_result=json.dumps(current_loop.get("exec_result", {}), indent=2),
        summary=current_loop.get("summary", "No summary available"),
        previous_analysis=agent_state.revision_analysis or "No previous analysis.",
        user_query=agent_state.user_query,
    )

    response = ModelRegistry.completion(
        LLM_NAME,
        [Message(role="user", content=analysis_prompt)],
        system_prompt=(
            Message(
                role="system",
                content=PROMPTS.experiment_agent.analysis_system_prompt.render(),
            )
            .with_log(cond=constant.LOG_SYSTEM_PROMPT)
            .content
        ),
        agent_sender=AGENT_NAME,
    )

    # Accumulate analysis
    analysis_text = response.content
    if agent_state.revision_analysis:
        agent_state.revision_analysis += (
            f"\n\n---\n\n## Revision {agent_state.current_revision} Analysis\n{analysis_text}"
        )
    else:
        agent_state.revision_analysis = (
            f"## Revision {agent_state.current_revision} Analysis\n{analysis_text}"
        )

    # Save analysis result to file
    try:
        import os

        analysis_dir = os.path.join(agent_state.workspace.working_dir, "experiment_analyses")
        os.makedirs(analysis_dir, exist_ok=True)

        analysis_file = os.path.join(
            analysis_dir, f"revision_{agent_state.current_revision}_analysis.md"
        )

        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(f"# Revision {agent_state.current_revision} Analysis\n\n")
            f.write(analysis_text)

        logger.info(f"Analysis saved to {analysis_file}")
    except Exception as e:
        logger.warning(f"Failed to save analysis to file: {e}")

    logger.debug(f"Analysis for revision {agent_state.current_revision} completed")

    agent_state.intermediate_state.append(
        {
            "node_name": "analysis",
            "output": analysis_text,
        }
    )

    return agent_state


def revision_judge_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    """Judge whether a revision is needed based on the summary.

    This node analyzes the experiment summary and decides:
    1. COMPLETE - Experiment succeeded, no more revisions needed
    2. CONTINUE - Issues found, need another revision loop
    3. COMPLETE (max_revisions) - Hit max revisions limit
    """
    logger.info("Revision Judge evaluating results")
    agent_state.current_phase = "judge"

    # Check max revisions
    if agent_state.current_revision >= agent_state.max_revisions - 1:
        logger.warning("Max revisions reached")
        agent_state.final_status = "max_revisions_reached"
        judge_output = "Max revisions reached - stopping"
        agent_state.intermediate_state.append(
            {
                "node_name": "revision_judge",
                "output": judge_output,
            }
        )
        return agent_state

    # Get the latest summary
    latest_summary = (
        agent_state.revision_summaries[-1]
        if agent_state.revision_summaries
        else "No summary available"
    )
    exec_result = agent_state.all_execution_results[-1] if agent_state.all_execution_results else {}

    # Use LLM to judge whether revision is needed (with accumulated analysis)
    judge_prompt = PROMPTS.experiment_agent.judge_prompt.render(
        latest_summary=latest_summary,
        exec_result=json.dumps(exec_result, indent=2),
        user_query=agent_state.user_query,
        revision_analysis=agent_state.revision_analysis or "No analysis available.",
    )

    response = ModelRegistry.completion(
        LLM_NAME,
        [Message(role="user", content=judge_prompt)],
        system_prompt=(
            Message(
                role="system",
                content=PROMPTS.experiment_agent.judge_system_prompt.render(),
            )
            .with_log(cond=constant.LOG_SYSTEM_PROMPT)
            .content
        ),
        agent_sender=AGENT_NAME,
    )

    class JudgeDecisionModel(BaseModel):
        """Model for revision judge decision"""

        decision: str  # "COMPLETE" or "CONTINUE"
        reason: str
        issues_to_fix: list[str] = []

    # Parse the response using utility function
    judge_output = response.content
    try:
        result = parse_json_from_llm_response(response, JudgeDecisionModel)

        if result.decision == "COMPLETE":
            logger.info("Revision judge decided: COMPLETE")
            agent_state.final_status = "success"
            judge_output = f"Decision: COMPLETE\nReason: {result.reason}"
        else:
            logger.info(f"Revision judge decided: CONTINUE - {result.reason}")
            # Prepare for next revision
            agent_state.current_revision += 1
            # Add feedback to history for next coding iteration
            feedback_msg = Message(
                role="user",
                content=PROMPTS.experiment_agent.revision_feedback_prompt.render(
                    attempt_number=agent_state.current_revision + 1,
                    reason=result.reason,
                    issues_to_fix=result.issues_to_fix,
                ),
                agent_sender=AGENT_NAME,
            )
            agent_state.add_message(feedback_msg)
            judge_output = f"Decision: CONTINUE\nReason: {result.reason}\nIssues to fix: {result.issues_to_fix}"
    except Exception as e:
        logger.error(f"Error parsing judge response: {e}")
        agent_state.final_status = "success"
        judge_output = f"Error parsing judge response: {e}"

    agent_state.intermediate_state.append(
        {
            "node_name": "revision_judge",
            "output": judge_output,
        }
    )

    return agent_state


def should_continue_revision(
    agent_state: ExperimentAgentState,
) -> Literal["continue", "complete"]:
    """Conditional edge function to determine next step after revision judge."""
    if agent_state.final_status is None:
        return "continue"
    return "complete"


def finalize_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    """Finalize the experiment and prepare output."""
    logger.info("Finalizing Experiment Agent")
    agent_state.current_phase = "complete"

    # Compile final summary
    exec_results_text = json.dumps(agent_state.all_execution_results, indent=2)

    agent_state.final_summary = f"""# Experiment Complete

## Status: {agent_state.final_status}

## Total Revisions: {agent_state.current_revision + 1}

## Final Summary
{agent_state.revision_summaries[-1] if agent_state.revision_summaries else 'No summary generated'}

## Accumulated Analysis
{agent_state.revision_analysis or 'No analysis available'}

## All Execution Results
{exec_results_text}
"""

    agent_state.intermediate_state.append(
        {
            "node_name": "finalize",
            "output": agent_state.final_summary,
        }
    )

    return agent_state
