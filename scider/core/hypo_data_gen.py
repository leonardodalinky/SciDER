"""Hypothetical/synthetic data generation.

Generates CSV data files based on user-described features and LLM-determined
distribution parameters. The generated files can then be fed into the existing
DataWorkflow for analysis.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, field_validator

from scider.core.llms import ModelRegistry
from scider.core.types import Message
from scider.core.utils import parse_json_from_text
from scider.prompts.prompt_data import PROMPTS

# ── Pydantic models for the generation spec ──────────────────────────────


class FeatureSpec(BaseModel):
    name: str
    dtype: Literal["numeric", "categorical", "integer"]
    distribution: Literal["gaussian", "uniform", "categorical", "integer_range"]
    params: dict[str, Any]
    description: str = ""

    @field_validator("name")
    @classmethod
    def name_must_be_identifier(cls, v: str) -> str:
        v = v.strip().replace(" ", "_")
        if not v.isidentifier():
            raise ValueError(f"Feature name must be a valid identifier, got: {v!r}")
        return v


class DataGenSpec(BaseModel):
    dataset_name: str = "synthetic_data"
    features: list[FeatureSpec]


# ── Core functions ───────────────────────────────────────────────────────


LLM_NAME = "data"  # reuse the data agent model for spec generation


def generate_data_spec(
    feature_desc: str,
    num_rows: int = 1000,
    feedback: str | None = None,
) -> DataGenSpec:
    """Ask LLM to produce a ``DataGenSpec`` from a natural-language description.

    Args:
        feature_desc: Free-text description of desired features.
        num_rows: Target row count (informational for the LLM).
        feedback: Optional user feedback from a rejected attempt.

    Returns:
        Validated ``DataGenSpec``.
    """
    prompt = PROMPTS.hypo_data_gen.user_prompt.render(
        feature_desc=feature_desc,
        num_rows=num_rows,
    )
    if feedback:
        prompt += (
            f"\n\nUser feedback on a previous attempt:\n{feedback}\nPlease adjust accordingly."
        )

    msg = ModelRegistry.completion(
        LLM_NAME,
        [Message(role="user", content=prompt)],
        system_prompt=PROMPTS.hypo_data_gen.system_prompt.render(),
        agent_sender="hypo_data_gen",
        tools=None,
    )

    raw = msg.content.strip()
    spec_dict = parse_json_from_text(raw)
    return DataGenSpec.model_validate(spec_dict)


def format_spec_for_review(spec: DataGenSpec) -> str:
    """Format a ``DataGenSpec`` as a readable table for user approval."""
    lines = [f"**Dataset**: {spec.dataset_name}", ""]
    lines.append("| Feature | Type | Distribution | Parameters |")
    lines.append("|---------|------|--------------|------------|")
    for f in spec.features:
        params_str = ", ".join(f"{k}={v}" for k, v in f.params.items())
        lines.append(f"| {f.name} | {f.dtype} | {f.distribution} | {params_str} |")
        if f.description:
            lines.append(f"|  | | | _{f.description}_ |")
    return "\n".join(lines)


def generate_csv_from_spec(
    spec: DataGenSpec,
    workspace_path: Path,
    num_rows: int = 1000,
) -> Path:
    """Generate a CSV file from a validated ``DataGenSpec``.

    Builds a small Python script and executes it with the current interpreter.

    Returns:
        Path to the directory containing the generated CSV.
    """
    output_dir = Path(workspace_path) / "generated_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{spec.dataset_name}.csv"

    # Build the generation script
    script_lines = [
        "import numpy as np",
        "import pandas as pd",
        "",
        "np.random.seed(42)",
        f"n = {num_rows}",
        "data = {}",
    ]

    for f in spec.features:
        p = f.params
        if f.distribution == "gaussian":
            mean = float(p.get("mean", 0))
            std = float(p.get("std", 1))
            script_lines.append(f'data["{f.name}"] = np.random.normal({mean}, {std}, n)')
        elif f.distribution == "uniform":
            low = float(p.get("low", 0))
            high = float(p.get("high", 1))
            script_lines.append(f'data["{f.name}"] = np.random.uniform({low}, {high}, n)')
        elif f.distribution == "categorical":
            cats = p.get("categories", ["A", "B"])
            probs = p.get("probs", None)
            if probs:
                script_lines.append(f'data["{f.name}"] = np.random.choice({cats}, n, p={probs})')
            else:
                script_lines.append(f'data["{f.name}"] = np.random.choice({cats}, n)')
        elif f.distribution == "integer_range":
            low = int(p.get("low", 0))
            high = int(p.get("high", 100))
            script_lines.append(f'data["{f.name}"] = np.random.randint({low}, {high + 1}, n)')

    script_lines += [
        "",
        "df = pd.DataFrame(data)",
        f'df.to_csv("{csv_path}", index=False)',
        'print(f"Generated {len(df)} rows, {len(df.columns)} columns")',
    ]

    script = "\n".join(script_lines)
    logger.debug(f"Data generation script:\n{script}")

    # Write and execute
    script_path = output_dir / "_generate.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Data generation failed:\n{result.stderr}")

    logger.info(f"Generated hypothetical data: {result.stdout.strip()}")
    logger.info(f"CSV saved to: {csv_path}")

    return output_dir
