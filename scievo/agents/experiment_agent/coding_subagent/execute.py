import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from functional import seq
from loguru import logger

from scievo import history_compression
from scievo.core import constant
from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.core.utils import wrap_dict_to_toon, wrap_text_with_block
from scievo.prompts import PROMPTS
from scievo.rbank.subgraph import mem_extraction, mem_retrieval
from scievo.tools import Tool, ToolRegistry

from .state import ExperimentAgentState

if TYPE_CHECKING:
    from scievo.rbank.memo import Memo

LLM_NAME = "execute"
AGENT_NAME = "experiment_executor"

BUILTIN_TOOLSETS = [
    "state",
    "history",
]
ALLOWED_TOOLSETS = ["shell", "fs", "cursor", "environment"]


def gateway_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    # NOTE: this node does nothing, it's just a placeholder for the conditional edges
    # Check `gateway_conditional` for the actual logic
    logger.trace("gateway_node of Agent {}", AGENT_NAME)
    # Initialize consecutive_questions if it doesn't exist (safety check for Pydantic)
    if not hasattr(agent_state, "consecutive_questions") or not isinstance(
        getattr(agent_state, "consecutive_questions", None), int
    ):
        agent_state.consecutive_questions = 0
    return agent_state


def _parse_and_select_action(content: str, current_plan: str | None) -> str | None:
    """
    Parse action options from LLM response and select the most appropriate one.
    Returns the selected action instruction, or None if no valid options found.
    """
    # Look for numbered or bulleted action lists
    # Pattern: "- Action description" or "1. Action description" or "- Action: description"
    action_patterns = [
        r"[-*]\s*(.+?)(?:—|–|-|:|\n|$)",
        r"\d+[\.)]\s*(.+?)(?:—|–|-|:|\n|$)",
    ]

    actions = []
    for pattern in action_patterns:
        matches = re.findall(pattern, content, re.MULTILINE)
        if matches:
            actions = [
                m.strip() for m in matches if len(m.strip()) > 10
            ]  # Filter out very short matches
            break

    if not actions:
        return None

    # Map actions to tool calls based on keywords
    action_mapping = {
        "install": "activate_toolset('environment') then pip_install_requirements",
        "dependencies": "activate_toolset('environment') then pip_install_requirements",
        "requirements": "activate_toolset('environment') then pip_install_requirements",
        "inspect": "read_file or list_files",
        "read": "read_file",
        "run": "activate_toolset('shell') then run_bash_cmd",
        "execute": "activate_toolset('shell') then run_bash_cmd",
        "test": "activate_toolset('shell') then run_bash_cmd",
        "edit": "activate_toolset('cursor') then cursor_edit",
        "refactor": "activate_toolset('cursor') then cursor_edit",
    }

    # Select action based on current plan and action keywords
    selected_action = None
    plan_lower = (current_plan or "").lower()

    # Priority: match plan keywords first, then common workflow order
    for action in actions:
        action_lower = action.lower()

        # Check if action matches plan keywords
        if any(keyword in plan_lower for keyword in ["install", "dependency", "requirements"]):
            if any(
                keyword in action_lower for keyword in ["install", "dependency", "requirements"]
            ):
                selected_action = action
                break

        if any(keyword in plan_lower for keyword in ["run", "execute", "test"]):
            if any(
                keyword in action_lower for keyword in ["run", "execute", "test", "script", "demo"]
            ):
                selected_action = action
                break

        if any(keyword in plan_lower for keyword in ["read", "inspect", "check"]):
            if any(keyword in action_lower for keyword in ["inspect", "read", "file"]):
                selected_action = action
                break

    # If no match with plan, use default priority: install > inspect > run
    if not selected_action:
        for priority_keyword in [
            "install",
            "dependencies",
            "requirements",
            "inspect",
            "read",
            "run",
            "execute",
        ]:
            for action in actions:
                if priority_keyword in action.lower():
                    selected_action = action
                    break
            if selected_action:
                break

    # If still no selection, use first action
    if not selected_action and actions:
        selected_action = actions[0]

    return selected_action


def gateway_conditional(agent_state: ExperimentAgentState) -> str:

    last_msg = agent_state.patched_history[-1]

    # If the last message is a tool message (tool execution completed),
    # move the current step from remaining_plans to past_plans
    # and trigger execution of the next step
    if (
        last_msg.role == "tool"
        and agent_state.remaining_plans
        and len(agent_state.remaining_plans) > 0
    ):
        completed_step = agent_state.remaining_plans.pop(0)
        agent_state.past_plans.append(completed_step)
        logger.debug(f"Step completed and moved to past_plans: {completed_step[:80]}...")

        # If there are more steps remaining, add a user message to trigger next step execution
        if agent_state.remaining_plans and len(agent_state.remaining_plans) > 0:
            agent_state.add_message(
                Message(
                    role="user",
                    content=PROMPTS.experiment.replanner_user_response.render(
                        next_step=agent_state.remaining_plans[0],
                    ),
                )
            )
            logger.debug(
                f"Added user message to trigger next step: {agent_state.remaining_plans[0][:80]}..."
            )

    if (tool_calls := last_msg.tool_calls) and len(tool_calls) > 0:
        return "tool_calling"

    match last_msg.role:
        case "user" | "tool":
            return "llm_chat"
        case "assistant":
            # Initialize consecutive_questions if it doesn't exist (safety check)
            if not hasattr(agent_state, "consecutive_questions"):
                agent_state.consecutive_questions = 0

            # Assistant message without tool calls - check if it's asking questions or providing options
            content = last_msg.content or ""
            content_lower = content.lower()

            # Check for option list pattern
            has_options = (
                "choose" in content_lower or "select" in content_lower or "which" in content_lower
            ) and ("- " in content or "* " in content or re.search(r"\d+[\.)]", content))

            is_asking = any(
                phrase in content_lower
                for phrase in [
                    "which",
                    "what should",
                    "what would you like",
                    "recommendation",
                    "option",
                    "choose",
                    "select",
                    "should i",
                    "would you like",
                ]
            )

            if has_options:
                # LLM provided options - automatically select and execute the most appropriate one
                current_plan = (
                    agent_state.remaining_plans[0] if agent_state.remaining_plans else None
                )
                selected_action = _parse_and_select_action(content, current_plan)

                if selected_action:
                    logger.info(f"Auto-selected action from options: {selected_action}")
                    agent_state.add_message(
                        Message(
                            role="user",
                            content=(
                                f"Execute this action immediately using tools: {selected_action}. "
                                f"Current plan step: {current_plan or 'N/A'}. "
                                "Do not ask questions - just call the appropriate tool(s) to execute this action. "
                                "If you need to activate a toolset first, use activate_toolset, then use tools from that toolset."
                            ),
                        )
                    )
                    agent_state.consecutive_questions = 0  # Reset counter since we're handling it
                    return "llm_chat"
                else:
                    # Could not parse options, fall through to asking detection
                    logger.warning("Could not parse action options from LLM response")

            if is_asking:
                # LLM is asking questions instead of using tools
                agent_state.consecutive_questions += 1
                max_questions = 3

                if agent_state.consecutive_questions >= max_questions:
                    # Too many consecutive questions - skip to replanner
                    logger.error(
                        f"LLM returned {agent_state.consecutive_questions} consecutive questions. "
                        "Skipping to replanner to avoid infinite loop."
                    )
                    agent_state.consecutive_questions = 0  # Reset counter
                    return "replanner"

                # Add stronger correction message
                logger.warning(
                    f"LLM returned question instead of tool call (attempt {agent_state.consecutive_questions}/{max_questions}), "
                    "adding correction message"
                )
                agent_state.add_message(
                    Message(
                        role="user",
                        content=(
                            "ERROR: You must use tools to execute actions. Do not ask questions or provide options. "
                            "You are an autonomous agent - make decisions and execute them immediately. "
                            f"Current plan step: {agent_state.remaining_plans[0] if agent_state.remaining_plans else 'N/A'}. "
                            "Call the appropriate tool NOW. If you need to activate a toolset, use activate_toolset first, then use tools from that toolset."
                        ),
                    )
                )
                return "llm_chat"
            else:
                # Reset counter if LLM didn't ask a question
                agent_state.consecutive_questions = 0

            # Assistant message without tool calls - check if we should continue or replan
            # If we have remaining plans, add user message and continue to llm_chat
            # Otherwise, go to replanner
            if agent_state.remaining_plans and len(agent_state.remaining_plans) > 0:
                # Add user message to trigger execution of next step
                agent_state.add_message(
                    Message(
                        role="user",
                        content=PROMPTS.experiment.replanner_user_response.render(
                            next_step=agent_state.remaining_plans[0],
                        ),
                    )
                )
                return "llm_chat"
            else:
                return "replanner"
        case _:
            raise ValueError(f"Unknown message role: {last_msg.role}")


mem_retrieval_subgraph = mem_retrieval.build()
mem_retrieval_subgraph_compiled = mem_retrieval_subgraph.compile()


def _memos_to_markdown(memos: list["Memo"]) -> str:
    ret = ""
    if len(memos) == 0:
        return "No memory retrieved."
    for i, memo in enumerate(memos):
        ret += f"# Memo {i + 1}\n\n{memo.to_markdown()}\n\n"
    return ret


def llm_chat_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    logger.debug("llm_chat_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("llm_chat")

    selected_state = {
        "current_working_dir": agent_state.local_env.working_dir,
        "current_activated_toolsets": agent_state.toolsets,
        "repo_dir": str(agent_state.repo_dir) if agent_state.repo_dir else None,
    }

    # retrieve memos
    if constant.REASONING_BANK_ENABLED:
        try:
            mem_dirs = [agent_state.sess_dir / "short_term"]
            if hasattr(agent_state, "long_term_mem_dir") and agent_state.long_term_mem_dir:
                mem_dirs.append(agent_state.long_term_mem_dir)
            if hasattr(agent_state, "project_mem_dir") and agent_state.project_mem_dir:
                mem_dirs.append(agent_state.project_mem_dir)
            res = mem_retrieval_subgraph_compiled.invoke(
                mem_retrieval.MemRetrievalState(
                    input_msgs=agent_state.patched_history,
                    mem_dirs=mem_dirs,
                    max_num_memos=constant.MEM_RETRIEVAL_MAX_NUM_MEMOS,
                )
            )
            memos: list["Memo"] = res.get("output_memos", [])
            memory_text = _memos_to_markdown(memos)
        except Exception:
            logger.exception("mem_retrieval_error")
            memory_text = None
    else:
        memory_text = None

    # update system prompt
    system_prompt = PROMPTS.experiment.experiment_chat_system_prompt.render(
        state_text=wrap_dict_to_toon(selected_state),
        toolsets_desc=ToolRegistry.get_toolsets_desc(BUILTIN_TOOLSETS + ALLOWED_TOOLSETS),
        memory_text=wrap_text_with_block(memory_text, "markdown"),
        current_plan=(
            agent_state.remaining_plans[0] if len(agent_state.remaining_plans) > 0 else None
        ),
    )

    # construct tools
    tools: dict[str, Tool] = {}
    for toolset in agent_state.toolsets:
        tools.update(ToolRegistry.get_toolset(toolset))
    for toolset in BUILTIN_TOOLSETS:
        tools.update(ToolRegistry.get_toolset(toolset))

    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.patched_history,
        system_prompt=(
            Message(role="system", content=system_prompt)
            .with_log(cond=constant.LOG_SYSTEM_PROMPT)
            .content
        ),
        agent_sender=AGENT_NAME,
        tools=[tool.name for tool in tools.values()],
    ).with_log()
    agent_state.add_message(msg)

    return agent_state


def tool_calling_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    logger.debug("tool_calling_node of Agent experiment_executor")
    last_msg = agent_state.patched_history[-1]

    if not getattr(last_msg, "tool_calls", None):
        return agent_state

    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        logger.debug("No tool calls in last LLM message.")
        return agent_state

    tools: dict[str, Tool] = {}
    for toolset in agent_state.toolsets:
        tools.update(ToolRegistry.get_toolset(toolset))
    for toolset in BUILTIN_TOOLSETS:
        tools.update(ToolRegistry.get_toolset(toolset))

    function_map = {tool.name: tool.func for tool in tools.values()}

    for tool_call in last_msg.tool_calls:
        tool_name = tool_call.function.name
        if tool_name not in function_map:
            error_msg = f"Tool {tool_name} not found."
            logger.error(error_msg)
            agent_state.add_message(
                Message(
                    role="tool",
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    content=error_msg,
                )
            )
            continue

        import json

        try:
            args = json.loads(tool_call.function.arguments)
            assert isinstance(args, dict)
        except Exception as e:
            error_msg = f"Invalid tool arguments: {e}"
            logger.error(error_msg)
            agent_state.add_message(
                Message(
                    role="tool",
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    content=error_msg,
                )
            )
            continue

        try:
            func = function_map[tool_name]

            # Check if function expects agent_state parameter
            import inspect

            sig = inspect.signature(func)
            if constant.__AGENT_STATE_NAME__ in sig.parameters:
                args.update({constant.__AGENT_STATE_NAME__: agent_state})
            if constant.__CTX_NAME__ in sig.parameters:
                args.update({constant.__CTX_NAME__: {"current_agent": AGENT_NAME}})

            # Execute the tool in the agent's local environment
            logger.info(f"Executing tool: {tool_name} with args: {args}")
            with agent_state.local_env:
                result = func(**args)
            logger.info(
                f"Tool {tool_name} executed successfully. Result length: {len(str(result))} chars"
            )

            # Reset consecutive questions counter on successful tool execution
            agent_state.consecutive_questions = 0

            agent_state.add_message(
                Message(
                    role="tool",
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    content=str(result),
                ).with_log()  # 添加日志输出
            )

        except Exception as e:
            error_msg = f"Tool {tool_name} execution failed: {e}"
            logger.error(error_msg)
            agent_state.add_message(
                Message(
                    role="tool",
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    content=error_msg,
                )
            )

    return agent_state


def report_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    """Generate and save experiment report as markdown file."""
    logger.debug("report_node of Agent {}", AGENT_NAME)

    try:
        # Generate report using LLM
        system_prompt = PROMPTS.experiment.experiment_summary_prompt.render()

        # Collect execution history for summary
        execution_summary = []
        execution_summary.append(f"# Experiment Report\n\n")
        execution_summary.append(f"## Repository Information\n\n")
        if agent_state.repo_dir:
            execution_summary.append(f"- Repository Directory: {agent_state.repo_dir}\n")
        if agent_state.user_query:
            execution_summary.append(f"- Repository URL: {agent_state.user_query}\n")
        if agent_state.user_instructions:
            execution_summary.append(f"- User Instructions: {agent_state.user_instructions}\n\n")

        execution_summary.append(f"## Execution Plan\n\n")
        if agent_state.plans:
            for i, step in enumerate(agent_state.plans.steps, 1):
                status = "✓" if i <= len(agent_state.past_plans) else "○"
                execution_summary.append(f"{status} Step {i}: {step}\n")
        execution_summary.append(f"\n")

        execution_summary.append(f"## Execution History\n\n")
        execution_summary.append(f"Total messages: {len(agent_state.patched_history)}\n\n")

        # Add key execution steps
        for i, msg in enumerate(agent_state.patched_history[-20:], 1):  # Last 20 messages
            if msg.role == "tool" and msg.tool_name:
                execution_summary.append(f"### Tool Execution {i}\n")
                execution_summary.append(f"- Tool: {msg.tool_name}\n")
                execution_summary.append(f"- Result: {str(msg.content)[:200]}...\n\n")
            elif msg.role == "assistant" and msg.tool_calls:
                execution_summary.append(f"### Assistant Action {i}\n")
                for tc in msg.tool_calls:
                    execution_summary.append(f"- Called: {tc.function.name}\n")
                execution_summary.append(f"\n")

        # Use LLM to generate a comprehensive summary
        user_prompt = f"""
Please generate a comprehensive experiment report based on the following information:

{''.join(execution_summary)}

Please provide a well-structured markdown report that includes:
1. Executive Summary
2. Experiment Objectives
3. Execution Steps and Results
4. Key Findings
5. Issues Encountered (if any)
6. Conclusions

Format the report in markdown.
"""

        summary_msg = ModelRegistry.completion(
            LLM_NAME,
            agent_state.patched_history[-10:],  # Use last 10 messages for context
            system_prompt=system_prompt,
            agent_sender=AGENT_NAME,
        )

        # Combine the summary with execution details
        from datetime import datetime

        report_content = f"""# Experiment Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{summary_msg.content}

---

## Detailed Execution Log

{''.join(execution_summary)}

## Complete Message History

Total messages processed: {len(agent_state.patched_history)}

"""

        # Save report to scievo/experiment_report.md
        # Find project root (scievo directory)
        # Path structure: scievo/agents/experiment_agent/execute.py
        # We want: scievo/experiment_report.md
        current_file = Path(__file__).resolve()
        # Navigate: scievo/agents/experiment_agent/execute.py -> scievo/
        project_root = current_file.parent.parent.parent
        report_path = project_root / "experiment_report.md"

        logger.info(f"Saving experiment report to: {report_path}")

        # Use fs tool to save the file
        from scievo.tools import ToolRegistry

        fs_tools = ToolRegistry.get_toolset("fs")
        if "save_file" in fs_tools:
            save_func = fs_tools["save_file"].func
            result = save_func(path=str(report_path), content=report_content)
            logger.info(f"Experiment report saved: {result}")
            agent_state.add_message(
                Message(
                    role="assistant",
                    content=f"Experiment report generated and saved to {report_path}",
                    agent_sender=AGENT_NAME,
                ).with_log()
            )
        else:
            # Fallback: direct file write
            report_path.write_text(report_content, encoding="utf-8")
            logger.info(f"Experiment report saved to {report_path}")
            agent_state.add_message(
                Message(
                    role="assistant",
                    content=f"Experiment report generated and saved to {report_path}",
                    agent_sender=AGENT_NAME,
                ).with_log()
            )

    except Exception as e:
        logger.exception("Error generating experiment report")
        agent_state.add_message(
            Message(
                role="assistant",
                content=f"Error generating experiment report: {e}",
                agent_sender=AGENT_NAME,
            ).with_log()
        )

    return agent_state
