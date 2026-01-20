"""
Loguru Logger Handler for Streamlit

Captures loguru logger output and makes it available for display in Streamlit.
"""

import sys
import time
from typing import Any

try:
    from loguru import logger

    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    logger = None

from workflow_monitor import PhaseType, get_monitor


class StreamlitLogHandler:
    """Handler that captures loguru logs and sends them to workflow monitor."""

    def __init__(self):
        self.logs: list[dict[str, Any]] = []
        self.monitor = get_monitor()
        self._original_handlers = []

    def setup(self, min_level: str = "DEBUG"):
        """Setup the handler to capture logs."""
        if not LOGURU_AVAILABLE:
            return

        # Remove default handler
        logger.remove()

        # Add our custom handler
        logger.add(
            self._log_handler,
            level=min_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            colorize=False,  # We'll handle colors in Streamlit
        )

        # Also keep console output (optional)
        logger.add(
            sys.stderr,
            level=min_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        )

    def _log_handler(self, message):
        """Handle log messages from loguru."""
        record = message.record

        # Extract information
        try:
            if hasattr(record["time"], "timestamp"):
                timestamp = record["time"].timestamp()
            else:
                timestamp = time.time()
        except Exception:
            timestamp = time.time()

        # Get raw message text without HTML tags
        # Use record["message"] to get the original message text
        raw_message = record.get("message", str(message))

        # Clean HTML tags if present
        import re

        clean_message = re.sub(r"<[^>]+>", "", raw_message)

        log_entry = {
            "timestamp": timestamp,
            "level": record["level"].name,
            "message": clean_message,
            "module": record.get("name", "unknown"),
            "function": record.get("function", "unknown"),
            "line": record.get("line", 0),
            "file": record["file"].name if record.get("file") else None,
        }

        # Store log
        self.logs.append(log_entry)

        # Also send to workflow monitor
        # Map log levels to message types
        level_to_message_type = {
            "TRACE": "status",
            "DEBUG": "status",
            "INFO": "status",
            "SUCCESS": "result",
            "WARNING": "action",
            "ERROR": "error",
            "CRITICAL": "error",
        }

        # Try to infer agent/node from module name
        agent_name = None
        node_name = None
        module_name = record["name"]

        if "data_agent" in module_name:
            agent_name = "Data Agent"
        elif "experiment_agent" in module_name:
            agent_name = "Experiment Agent"
        elif "ideation_agent" in module_name:
            agent_name = "Ideation Agent"
        elif "critic_agent" in module_name:
            agent_name = "Critic Agent"

        # Try to infer node from function name
        function_name = record["function"]
        if "_node" in function_name:
            node_name = function_name.replace("_node", "")
        elif function_name.endswith("node"):
            node_name = function_name[:-4]

        # Determine phase
        phase = PhaseType.DATA_EXECUTION  # Default
        if "experiment" in module_name:
            phase = PhaseType.EXPERIMENT_EXEC
        elif "ideation" in module_name:
            phase = PhaseType.IDEATION_LITERATURE_SEARCH

        # Send to monitor
        self.monitor.log_update(
            phase=phase,
            status="progress",
            message=f"[{record['level'].name}] {str(message)}",
            agent_name=agent_name,
            message_type=level_to_message_type.get(record["level"].name, "status"),
            node_name=node_name,
            intermediate_output={
                "log_level": record["level"].name,
                "module": module_name,
                "function": function_name,
                "line": record["line"],
                "file": record["file"].name if record.get("file") else None,
            },
        )

    def get_logs(self, level: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Get logs, optionally filtered by level."""
        logs = self.logs
        if level:
            logs = [log for log in logs if log["level"] == level.upper()]
        if limit:
            logs = logs[-limit:]
        return logs

    def clear(self):
        """Clear all logs."""
        self.logs.clear()


# Global logger handler instance
_global_log_handler: StreamlitLogHandler | None = None


def get_log_handler() -> StreamlitLogHandler:
    """Get the global log handler instance."""
    global _global_log_handler
    if _global_log_handler is None:
        _global_log_handler = StreamlitLogHandler()
    return _global_log_handler


def setup_streamlit_logging(min_level: str = "DEBUG"):
    """Setup loguru to capture logs for Streamlit display."""
    handler = get_log_handler()
    handler.setup(min_level=min_level)
    return handler


def reset_log_handler():
    """Reset the global log handler."""
    global _global_log_handler
    if _global_log_handler:
        _global_log_handler.clear()
    _global_log_handler = StreamlitLogHandler()
    if LOGURU_AVAILABLE:
        setup_streamlit_logging()
