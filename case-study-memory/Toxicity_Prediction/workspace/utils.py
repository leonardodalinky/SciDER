"""Utility functions for mutagenicity pipeline."""

from pathlib import Path

RANDOM_STATE = 42
WORKSPACE = Path(__file__).parent


# Data paths
def get_data_path() -> Path:
    """Return path to Mutagenicity CSV (handles alternate filenames)."""
    for name in ["Mutagenicity_N6512.csv", "Mutagenicity_N6512 2.csv"]:
        p = WORKSPACE / name
        if p.exists():
            return p
    return WORKSPACE / "Mutagenicity_N6512.csv"
