"""
Node Monitor Wrapper

Automatically wraps all node functions to capture intermediate outputs.
This module provides utilities to monitor node execution in LangGraph workflows.
"""

import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Callable

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from workflow_monitor import PhaseType, get_monitor
except ImportError:
    # If workflow_monitor is not available, create a dummy
    class PhaseType:
        DATA_EXECUTION = "data_execution"
        EXPERIMENT_EXEC = "experiment_exec"
        IDEATION_LITERATURE_SEARCH = "ideation_literature_search"

    def get_monitor():
        class DummyMonitor:
            def log_node_update(self, *args, **kwargs):
                pass

        return DummyMonitor()


def _capture_state_snapshot(agent_state) -> dict[str, Any]:
    """Capture a snapshot of agent state for intermediate output."""
    snapshot = {}

    try:
        # Capture basic state info
        if hasattr(agent_state, "history"):
            snapshot["message_count"] = len(agent_state.history)
            if agent_state.history:
                last_msg = agent_state.history[-1]
                snapshot["last_message_preview"] = (
                    last_msg.content[:200] if hasattr(last_msg, "content") else str(last_msg)[:200]
                )

        if hasattr(agent_state, "node_history"):
            snapshot["node_history"] = (
                agent_state.node_history.copy() if agent_state.node_history else []
            )

        if hasattr(agent_state, "workspace"):
            if hasattr(agent_state.workspace, "working_dir"):
                snapshot["workspace"] = str(agent_state.workspace.working_dir)
            else:
                snapshot["workspace"] = str(agent_state.workspace)

        if hasattr(agent_state, "remaining_plans"):
            snapshot["remaining_plans_count"] = (
                len(agent_state.remaining_plans) if agent_state.remaining_plans else 0
            )

        if hasattr(agent_state, "past_plans"):
            snapshot["past_plans_count"] = (
                len(agent_state.past_plans) if agent_state.past_plans else 0
            )

        # Capture other relevant state fields
        for attr in ["user_query", "talk_mode", "output_summary", "data_desc"]:
            if hasattr(agent_state, attr):
                value = getattr(agent_state, attr)
                # Convert to string if needed
                if isinstance(value, (str, int, float, bool, type(None))):
                    snapshot[attr] = value
                elif value is not None:
                    snapshot[attr] = str(value)[:100]

        # Capture toolset info
        if hasattr(agent_state, "toolsets"):
            snapshot["toolsets"] = agent_state.toolsets.copy() if agent_state.toolsets else []

    except Exception as e:
        snapshot["capture_error"] = str(e)

    return snapshot


def _get_new_messages(state_before: dict | None, state_after: dict | None) -> list[dict]:
    """Extract new messages added during node execution."""
    if not state_before or not state_after:
        return []

    before_count = state_before.get("message_count", 0)
    after_count = state_after.get("message_count", 0)

    if after_count > before_count:
        messages_added = []
        for i in range(after_count - before_count):
            messages_added.append(
                {
                    "index": before_count + i,
                    "preview": state_after.get("last_message_preview", "")[:200],
                }
            )
        return messages_added
    return []


def monitor_node(
    node_name: str,
    agent_name: str | None = None,
    phase: PhaseType | None = None,
    capture_state: bool = True,
):
    """
    Decorator to automatically monitor node execution and capture intermediate outputs.

    Args:
        node_name: Name of the node
        agent_name: Name of the agent (will try to infer from state if not provided)
        phase: Phase type (will use default if not provided)
        capture_state: Whether to capture agent state as intermediate output

    Usage:
        @monitor_node(node_name="llm_chat", agent_name="Data Agent", phase=PhaseType.DATA_EXECUTION)
        def llm_chat_node(agent_state: DataAgentState) -> DataAgentState:
            # ... node logic ...
            return agent_state
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(agent_state, *args, **kwargs):
            monitor = get_monitor()

            # Try to infer agent name from state
            inferred_agent_name = agent_name
            if not inferred_agent_name:
                # Try to get from agent_state attributes
                if hasattr(agent_state, "agent_sender"):
                    inferred_agent_name = agent_state.agent_sender
                elif hasattr(agent_state, "__class__"):
                    class_name = agent_state.__class__.__name__
                    if "DataAgent" in class_name:
                        inferred_agent_name = "Data Agent"
                    elif "ExperimentAgent" in class_name:
                        inferred_agent_name = "Experiment Agent"
                    elif "IdeationAgent" in class_name:
                        inferred_agent_name = "Ideation Agent"
                    elif "CriticAgent" in class_name:
                        inferred_agent_name = "Critic Agent"

            if not inferred_agent_name:
                inferred_agent_name = "Unknown Agent"

            # Determine phase if not provided
            inferred_phase = phase
            if not inferred_phase:
                if "Data" in inferred_agent_name:
                    inferred_phase = PhaseType.DATA_EXECUTION
                elif "Experiment" in inferred_agent_name:
                    inferred_phase = PhaseType.EXPERIMENT_EXEC
                elif "Ideation" in inferred_agent_name:
                    inferred_phase = PhaseType.IDEATION_LITERATURE_SEARCH
                else:
                    inferred_phase = PhaseType.DATA_EXECUTION  # Default

            # Log node start
            monitor.log_node_update(
                phase=inferred_phase,
                node_name=node_name,
                status="started",
                message=f"Node '{node_name}' started",
                agent_name=inferred_agent_name,
                message_type="status",
            )

            # Capture state before execution
            state_before = None
            if capture_state:
                try:
                    state_before = _capture_state_snapshot(agent_state)
                except Exception as e:
                    state_before = {"error": f"Failed to capture state: {e}"}

            try:
                # Execute node
                result = func(agent_state, *args, **kwargs)

                # Capture state after execution
                state_after = None
                intermediate_output = {}

                if capture_state:
                    try:
                        state_after = _capture_state_snapshot(result)

                        # Calculate intermediate output (what changed)
                        intermediate_output = {
                            "state_before": state_before,
                            "state_after": state_after,
                            "messages_added": _get_new_messages(state_before, state_after),
                            "node_history": state_after.get("node_history", []),
                        }

                        # Add other useful info
                        if "message_count" in state_after:
                            intermediate_output["message_count"] = state_after["message_count"]
                        if "remaining_plans_count" in state_after:
                            intermediate_output["remaining_plans_count"] = state_after[
                                "remaining_plans_count"
                            ]
                        if "workspace" in state_after:
                            intermediate_output["workspace"] = state_after["workspace"]

                    except Exception as e:
                        intermediate_output = {
                            "error": f"Failed to capture state: {e}",
                            "state_before": state_before,
                        }

                # Log node completion
                monitor.log_node_update(
                    phase=inferred_phase,
                    node_name=node_name,
                    status="completed",
                    message=f"Node '{node_name}' completed",
                    intermediate_output=intermediate_output,
                    agent_name=inferred_agent_name,
                    message_type="result",
                )

                return result

            except Exception as e:
                # Log node error
                monitor.log_node_update(
                    phase=PhaseType.ERROR if hasattr(PhaseType, "ERROR") else inferred_phase,
                    node_name=node_name,
                    status="error",
                    message=f"Node '{node_name}' failed: {str(e)}",
                    intermediate_output={"error": str(e), "state_before": state_before},
                    agent_name=inferred_agent_name,
                    message_type="error",
                )
                raise

        return wrapper

    return decorator


def wrap_node_for_monitoring(
    node_func: Callable,
    node_name: str,
    agent_name: str | None = None,
    phase: PhaseType | None = None,
) -> Callable:
    """
    Wrap a node function for monitoring without using decorator syntax.

    Useful when you can't modify the node function definition.

    Args:
        node_func: The node function to wrap
        node_name: Name of the node
        agent_name: Name of the agent
        phase: Phase type

    Returns:
        Wrapped function with monitoring
    """
    return monitor_node(node_name=node_name, agent_name=agent_name, phase=phase)(node_func)
