"""
Serialization utilities for agent states and histories.
"""

import json
from pathlib import Path

from scievo.core.types import HistoryState


def save_agent_conversations(agents: dict[str, HistoryState], output_path: str | Path):
    """Save conversations of multiple agents to a file.

    Args:
        agents (dict[str, HistoryState]): A dictionary mapping agent names to their states.
        output_path (str): The json path to the file where conversations will be saved.
    """
    # If `output_path` is not a valid json path, this will raise an error.
    output_path = Path(output_path)
    if not output_path.suffix == ".json":
        raise ValueError("Output path must be a .json file")

    # Extract and serialize history from each agent
    serialized_agents = {}
    for name, state in agents.items():
        # Serialize each Message in the history
        serialized_history = [msg.model_dump() for msg in state.history]
        serialized_agents[name] = serialized_history

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to JSON file with proper formatting
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized_agents, f, indent=2, ensure_ascii=False)
