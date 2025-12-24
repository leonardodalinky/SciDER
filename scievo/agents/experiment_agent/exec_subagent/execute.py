"""
Experiment Execution Agent - handles running experiments in local shell sessions
"""

import inspect
import json
import time

from loguru import logger
from pydantic import BaseModel

from scievo.core import constant
from scievo.core.llms import ModelRegistry
from scievo.core.types import Message
from scievo.core.utils import parse_json_from_llm_response, wrap_dict_to_toon
from scievo.prompts import PROMPTS
from scievo.tools import Tool, ToolRegistry

from .state import ExecAgentState

LLM_NAME = "experiment_execute"
LLM_MONITOR_NAME = "experiment_monitor"
AGENT_NAME = "experiment_exec"

BUILTIN_TOOLSETS = [
    "state",
    "exec",  # The exec toolset is built-in for this agent
]
ALLOWED_TOOLSETS = [
    "history",
    "fs",
]  # Can be extended if needed

MONITORING_INTERVALS = [5, 10, 10, 20, 20, 30, 45, 60, 60, 120]  # in seconds


def gateway_node(agent_state: ExecAgentState) -> ExecAgentState:
    """Gateway node - placeholder for conditional routing logic"""
    logger.trace("gateway_node of Agent {}", AGENT_NAME)
    return agent_state


def gateway_conditional(agent_state: ExecAgentState) -> str:
    """Determine the next node based on the last message"""
    # Check if there's a command currently running in the session
    if agent_state.is_monitor_mode:
        # A command is running -> go to monitoring node
        time2sleep = MONITORING_INTERVALS[
            min(agent_state.monitoring_attempts, len(MONITORING_INTERVALS) - 1)
        ]
        logger.debug(
            f"A command is currently running. Waiting for {time2sleep} seconds before monitoring again."
        )
        time.sleep(time2sleep)
        return "monitoring"

    last_msg = agent_state.patched_history[-1]

    # If the last message contains tool calls, execute them
    if (tool_calls := last_msg.tool_calls) and len(tool_calls) > 0:
        return "tool_calling"

    # Route based on message role
    match last_msg.role:
        case "user" | "tool":
            # User or tool message -> call LLM
            return "llm_chat"
        case "assistant":
            # Assistant responded without tool calls -> execution is complete, go to summary
            return "summary"
        case _:
            raise ValueError(f"Unknown message role: {last_msg.role}")


def llm_chat_node(agent_state: ExecAgentState) -> ExecAgentState:
    """LLM chat node - gets next action from the model"""
    logger.debug("llm_chat_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("llm_chat")

    selected_state = {
        "workspace": agent_state.workspace,
        "current_activated_toolsets": agent_state.toolsets,
    }

    # Update system prompt
    system_prompt = PROMPTS.experiment_exec.exec_system_prompt.render(
        state_text=wrap_dict_to_toon(selected_state),
        toolsets_desc=ToolRegistry.get_toolsets_desc(BUILTIN_TOOLSETS + ALLOWED_TOOLSETS),
    )

    # Construct tools
    tools: dict[str, Tool] = {}
    for toolset in agent_state.toolsets:
        tools.update(ToolRegistry.get_toolset(toolset))
    for toolset in BUILTIN_TOOLSETS:
        tools.update(ToolRegistry.get_toolset(toolset))

    # Get completion from LLM
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


def monitoring_node(agent_state: ExecAgentState) -> ExecAgentState:
    """Monitor a running command and decide whether to continue waiting or interrupt it"""
    logger.debug("monitoring_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("monitoring")
    agent_state.monitoring_attempts += 1

    if agent_state.monitoring_attempts <= len(MONITORING_INTERVALS):
        total_monitoring_seconds = sum(MONITORING_INTERVALS[: agent_state.monitoring_attempts])
    else:
        total_monitoring_seconds = (
            sum(MONITORING_INTERVALS)
            + (agent_state.monitoring_attempts - len(MONITORING_INTERVALS))
            * MONITORING_INTERVALS[-1]
        )

    # Get the current running command context
    ctx = agent_state.session.get_current_context()
    if ctx is None:
        # No command running, this shouldn't happen but handle it gracefully
        logger.warning("monitoring_node called but no command is running")
        agent_state.monitoring_attempts = 0
        agent_state.is_monitor_mode = False
        return agent_state

    # Get current output from the running command
    current_output = ctx.get_input_output()

    if not agent_state.session.is_running_command():
        # Command has completed while we were waiting
        logger.debug("The monitored command has completed.")
        agent_state.monitoring_attempts = 0
        agent_state.is_monitor_mode = False

        # Add monitoring end user prompt message
        monitoring_end_user_msg = Message(
            role="user",
            content=PROMPTS.experiment_exec.monitoring_end_user_prompt.render(
                command=ctx.command,
                final_output=current_output,
                error_text=ctx.get_error(),
                total_monitoring_seconds=total_monitoring_seconds,
            ),
            agent_sender=AGENT_NAME,
        ).with_log()
        agent_state.add_message(monitoring_end_user_msg)

        return agent_state

    history = agent_state.patched_history.copy()
    # Prepare monitoring prompt
    monitoring_user_msg = Message(
        role="user",
        content=PROMPTS.experiment_exec.monitoring_user_prompt.render(
            command=ctx.command,
            monitoring_attempts=agent_state.monitoring_attempts,
            current_output=current_output,
            total_monitoring_seconds=total_monitoring_seconds,
        ),
        agent_sender=AGENT_NAME,
    )
    history.append(monitoring_user_msg)

    # Ask monitoring LLM to decide
    msg = ModelRegistry.completion(
        LLM_MONITOR_NAME,
        history,
        system_prompt=(
            Message(
                role="system",
                content=PROMPTS.experiment_exec.monitoring_system_prompt.render(),
            )
            .with_log(cond=constant.LOG_SYSTEM_PROMPT)
            .content
        ),
        agent_sender=AGENT_NAME,
        tools=None,
    ).with_log()

    class MonitorDecisionModel(BaseModel):
        action: str

    r = parse_json_from_llm_response(msg, MonitorDecisionModel)  # just to validate JSON format

    if "wait" in r.action.lower():
        logger.debug("Monitoring decision: continue waiting for the command to complete.")
        agent_state.is_monitor_mode = True
    elif "ctrlc" in r.action.lower():
        logger.debug("Monitoring decision: interrupting the running command.")
        ctx.cancel()
        agent_state.is_monitor_mode = False
        monitoring_ctrlc_user_msg = Message(
            role="user",
            content=PROMPTS.experiment_exec.monitoring_ctrlc_user_prompt.render(
                command=ctx.command,
                output_before_interrupt=current_output,
                total_monitoring_seconds=total_monitoring_seconds,
            ),
            agent_sender=AGENT_NAME,
        )
        agent_state.add_message(monitoring_ctrlc_user_msg)
    else:
        logger.warning(
            f"Unknown monitoring action '{r.action}' received. Continuing to wait by default."
        )
        agent_state.is_monitor_mode = True

    return agent_state


def summary_node(agent_state: ExecAgentState) -> ExecAgentState:
    """Generate a summary of the experiment execution"""
    logger.debug("summary_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("summary")

    # Construct a prompt to generate the summary
    summary_prompt = Message(
        role="user",
        content=PROMPTS.experiment_exec.summary_user_prompt.render(),
        agent_sender=AGENT_NAME,
    )
    agent_state.add_message(summary_prompt)

    # Get summary from LLM
    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.patched_history,
        system_prompt=(
            Message(
                role="system",
                content=PROMPTS.experiment_exec.summary_system_prompt.render(),
            )
            .with_log(cond=constant.LOG_SYSTEM_PROMPT)
            .content
        ),
        agent_sender=AGENT_NAME,
        tools=None,  # No tools needed for summary
    ).with_log()

    # Store the summary text
    agent_state.execution_summary = msg.content or ""
    agent_state.add_message(msg)

    # Parse JSON summary from the response
    try:

        class ExecutionSummary(BaseModel):
            status: str
            commands_executed: list[str]
            key_outputs: str
            errors_issues: str

        summary_dict = parse_json_from_llm_response(msg, ExecutionSummary)
        agent_state.execution_summary_dict = summary_dict.model_dump()
    except Exception as e:
        logger.warning(f"Failed to parse execution summary as JSON: {e}")
        # If JSON parsing fails, store the text response in a basic dict structure
        agent_state.execution_summary_dict = {
            "status": "Unknown",
            "commands_executed": [],
            "key_outputs": agent_state.execution_summary,
            "errors_issues": str(e),
        }

    return agent_state


def tool_calling_node(agent_state: ExecAgentState) -> ExecAgentState:
    """Execute tool calls from the last message"""
    logger.debug("tool_calling_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("tool_calling")

    # Get the last message which contains tool calls
    last_msg = agent_state.patched_history[-1]

    if not last_msg.tool_calls:
        raise ValueError("No tool calls found in the last message")

    # Construct tools
    tools: dict[str, Tool] = {}
    for toolset in agent_state.toolsets:
        tools.update(ToolRegistry.get_toolset(toolset))
    for toolset in BUILTIN_TOOLSETS:
        tools.update(ToolRegistry.get_toolset(toolset))

    function_map = {tool.name: tool.func for tool in tools.values()}

    # Execute each tool call
    for tool_call in last_msg.tool_calls:
        tool_name = tool_call.function.name

        # Check if tool exists in function map
        if tool_name not in function_map:
            error_msg = f"Tool {tool_name} not found"
            tool_response = {
                "role": "tool",
                "tool_name": tool_name,
                "tool_call_id": tool_call.id,
                "content": error_msg,
            }
            agent_state.add_message(Message(**tool_response).with_log())
            continue

        # Parse tool arguments
        try:
            args = json.loads(tool_call.function.arguments)
            assert isinstance(args, dict)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in tool arguments: {e}"
            tool_response = {
                "role": "tool",
                "tool_name": tool_name,
                "tool_call_id": tool_call.id,
                "content": error_msg,
            }
            agent_state.add_message(Message(**tool_response).with_log())
            continue
        except AssertionError as e:
            error_msg = f"Invalid tool arguments: {e}"
            tool_response = {
                "role": "tool",
                "tool_name": tool_name,
                "tool_call_id": tool_call.id,
                "content": error_msg,
            }
            agent_state.add_message(Message(**tool_response).with_log())
            continue

        # Execute the tool
        try:
            func = function_map[tool_name]

            # Check if function expects agent_state parameter
            sig = inspect.signature(func)
            if constant.__AGENT_STATE_NAME__ in sig.parameters:
                args.update({constant.__AGENT_STATE_NAME__: agent_state})
            if constant.__CTX_NAME__ in sig.parameters:
                args.update({constant.__CTX_NAME__: {"current_agent": AGENT_NAME}})

            # Execute the tool
            result = func(**args)

            # Create tool response message
            tool_response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": tool_name,
                "content": str(result),  # Ensure result is string
            }

            # if this is a long-running exec_command, check for monitoring flag
            flag_text = "Try to check the execution status later."
            if tool_name == "exec_command" and flag_text in tool_response["content"]:
                logger.debug("The executed command is still running, entering monitor mode.")
                assert (
                    agent_state.session.get_current_context() is not None
                ), "Expected a current context when entering monitor mode"
                # The command is still running, go into monitor mode in the next step
                agent_state.is_monitor_mode = True

        except Exception as e:
            logger.exception(f"Tool {tool_name} execution failed")
            error_msg = f"Tool {tool_name} execution failed: {e}"
            tool_response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": tool_name,
                "content": error_msg,
            }

        agent_state.add_message(Message(**tool_response).with_log())

    # Reset monitoring attempts after tool execution
    agent_state.monitoring_attempts = 0

    return agent_state
