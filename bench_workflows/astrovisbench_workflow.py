"""
Workflow for AstroVisBench — generate the final visualization image for each
query using SciDER's Claude coding subagent.

Scope
-----
The upstream AstroVisBench pipeline is three stages:

    1. Code generation (processing + visualization)
    2. Execution + variable-inspection eval (`exec_bench.py`)
    3. LLM-as-judge visualization eval (`vis_evaluation.py`)
          ↓
       `aggregate_results.py`

This workflow only covers what is needed to *produce the final visualization
image*, so stages 2 and 3 are intentionally dropped. Concretely:

- We do NOT generate `processing_gen_code`. When building the visualization
  prompt we reuse `processing_gt_code` directly — this matches the upstream
  "Visualization" prompt recipe (see AstroVisBench/README.md §"Using the
  Benchmark") and avoids spending an extra LLM call on code we would throw
  away.
- We do NOT run `exec_bench.py`'s notebook / variable-inspection machinery,
  the cache-ballooning logic, the MPI splitter, or the LLM-as-judge.
- We DO run the LLM-produced visualization code once, in a subprocess, just
  long enough to save a PNG via `plt.savefig`.

Usage
-----
    # from the SciEvo repo root, with the astrovisbench venv set up
    # (see benchmarks/astrovisbench/README.md)
    python -m bench_workflows.astrovisbench_workflow \\
        --queries benchmarks/astrovisbench/data/astrovisbench_queries.json \\
        --output-dir benchmarks/astrovisbench/data/scider_vis/ \\
        --bench-env benchmarks/astrovisbench/data/bench_env \\
        --python   benchmarks/astrovisbench/.venv/bin/python \\
        --models   gemini-medium-high
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

# Make scider + bench modules importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench_workflows.register_models.gemini import (
    register_gemini3_medium_high_models,
    register_gemini_low_medium_models,
    register_gemini_medium_high_models,
)
from bench_workflows.register_models.gpt import (
    register_gpt_low_medium_models,
    register_gpt_medium_high_models,
)
from scider.agents.coding_subagent_native import NativeCodingAgentState, build
from scider.core.code_env import LocalEnv

# Fixed file name the subagent is instructed to populate — we read it back
# after the graph completes to recover the generated visualization code.
VIS_OUTPUT_FILENAME = "vis_output.py"


# --------------------------------------------------------------------------- #
# Prompting                                                                   #
# --------------------------------------------------------------------------- #

# Framed as the task for the native coding subagent. Unlike the Claude
# subagent (which returns the raw LLM output), the native subagent returns a
# summary string — so we instead ask it to write the visualization code to a
# specific file in the workspace, which we then read back.
_VIS_TASK_PREAMBLE = (
    "You are completing a Jupyter notebook about astronomy. You are given "
    "markdown cells (the task description) and python code cells (already "
    "executed; treat their results as already in scope). Your job is to "
    "produce ONLY the python code for the final visualization cell.\n\n"
    "## Output protocol (important)\n"
    f"- Write the visualization code to the file `{VIS_OUTPUT_FILENAME}` in "
    "the workspace root using the `FileWrite` tool. That file is what gets "
    "graded — its contents are the visualization cell.\n"
    "- The code in that file will be prepended at runtime with "
    "`setup_gt_code` and `processing_gt_code`, so you may reference any "
    "variables they define.\n"
    "- Do NOT call `plt.show()`; the caller runs `plt.savefig` on the "
    "current figure. Do NOT wrap the file contents in triple backticks.\n"
    "- Do NOT attempt to run the code, install packages, or fetch data — "
    "this workspace does NOT have the astronomy deps (astropy, etc.). "
    "Just write the file and finish.\n\n"
    "---\n\n"
)


def _build_visualization_prompt(query: dict, add_vis_underspec: bool = False) -> str:
    """Assemble the visualization prompt exactly as upstream does for the
    "Visualization" task recipe, minus the processing-gen step."""
    parts = [
        query["setup_query"],
        "```\n\n" + query["setup_gt_code"] + "\n\n```",
        query["processing_query"],
        "```\n\n" + query["processing_gt_code"] + "\n\n```",
        query["visualization_query"],
    ]
    if add_vis_underspec and query.get("visualization_underspecifications"):
        parts.append(query["visualization_underspecifications"])
    return _VIS_TASK_PREAMBLE + "\n\n".join(parts)


def _strip_code_fences(code: str) -> str:
    """Drop ``` fences and stray <CELL END> markers in case the agent wrote
    fenced content despite instructions."""
    kept = [
        line
        for line in code.splitlines()
        if not line.lstrip().startswith("```") and "<CELL END>" not in line
    ]
    return "\n".join(kept).strip()


# --------------------------------------------------------------------------- #
# Code generation                                                             #
# --------------------------------------------------------------------------- #


def generate_visualization_code(
    query: dict,
    workspace_dir: str | Path | None = None,
    add_vis_underspec: bool = False,
) -> str:
    """Drive the native coding subagent and read back the generated code.

    The native subagent returns a free-form summary (not the raw code), so we
    instead instruct it — in `_VIS_TASK_PREAMBLE` — to `FileWrite` the
    visualization code to `VIS_OUTPUT_FILENAME` in the workspace. After the
    graph completes we read that file back as the authoritative output.
    """
    user_query = _build_visualization_prompt(query, add_vis_underspec=add_vis_underspec)

    if workspace_dir is None:
        workspace_dir = tempfile.mkdtemp(prefix="astrovisbench_coding_")
        logger.debug(f"Coding subagent scratch workspace: {workspace_dir}")
    workspace = LocalEnv(working_dir=str(workspace_dir), create_dir_if_missing=True)

    state = NativeCodingAgentState(
        user_query=user_query,
        workspace=workspace,
        data_summary="",
    )
    graph = build().compile()
    graph.invoke(state)

    vis_file = Path(workspace.working_dir) / VIS_OUTPUT_FILENAME
    if not vis_file.exists():
        raise RuntimeError(
            f"Native coding subagent did not write {VIS_OUTPUT_FILENAME} in "
            f"workspace {workspace.working_dir}. Check the agent output."
        )
    code = vis_file.read_text(encoding="utf-8")
    if not code.strip():
        raise RuntimeError(f"{vis_file} is empty — agent produced no code.")

    return _strip_code_fences(code)


# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #


def render_visualization(
    query: dict,
    vis_code: str,
    output_image: str | Path,
    bench_env: str | Path | None = None,
    python_exec: str = sys.executable,
    timeout: int = 900,
) -> None:
    """Run (setup_gt_code → processing_gt_code → vis_code → savefig) in a
    subprocess with the astrovisbench venv, saving the figure to output_image.

    If `bench_env` is provided the subprocess runs with cwd=bench_env so any
    relative paths inside setup_gt_code / processing_gt_code resolve against
    the downloaded bench_env state (matching upstream exec_bench.py behavior).
    """
    output_image = Path(output_image).resolve()
    output_image.parent.mkdir(parents=True, exist_ok=True)

    script = "\n".join(
        [
            "import matplotlib",
            "matplotlib.use('Agg')",
            "import matplotlib.pyplot as plt",
            "",
            "# --- setup (ground truth) ---",
            query["setup_gt_code"],
            "",
            "# --- processing (ground truth) ---",
            query["processing_gt_code"],
            "",
            "# --- visualization (LLM-generated) ---",
            vis_code,
            "",
            "# --- save current figure ---",
            f"plt.savefig({str(output_image)!r}, dpi=150, bbox_inches='tight')",
            "plt.close('all')",
            "",
        ]
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        proc = subprocess.run(
            [python_exec, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(bench_env) if bench_env else None,
        )
    finally:
        os.unlink(script_path)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Visualization script failed (rc={proc.returncode}).\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    if not output_image.exists():
        raise RuntimeError(
            f"Visualization script exited OK but produced no image at {output_image}"
        )


# --------------------------------------------------------------------------- #
# Top-level per-query entry point                                             #
# --------------------------------------------------------------------------- #


def run_astrovisbench_workflow(
    query: dict,
    output_image: str | Path,
    bench_env: str | Path | None = None,
    workspace_dir: str | Path | None = None,
    python_exec: str = sys.executable,
    add_vis_underspec: bool = False,
    skip_render: bool = False,
) -> dict:
    """End-to-end: generate vis code for one query, then render the image.

    Returns a dict with `visualization_gen_code` and (if rendered)
    `image_path`. Raises on failure; callers decide retry/skip policy.
    """
    snippet = str(query.get("setup_query", ""))[:80].replace("\n", " ")
    logger.info(f"Generating visualization code for query: {snippet!r}")

    vis_code = generate_visualization_code(
        query=query,
        workspace_dir=workspace_dir,
        add_vis_underspec=add_vis_underspec,
    )
    out: dict = {"visualization_gen_code": vis_code}

    if skip_render:
        return out

    logger.info(f"Rendering visualization to {output_image}")
    render_visualization(
        query=query,
        vis_code=vis_code,
        output_image=output_image,
        bench_env=bench_env,
        python_exec=python_exec,
    )
    out["image_path"] = str(Path(output_image).resolve())
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _register_models(name: str) -> None:
    match name:
        case "gpt-low-medium":
            register_gpt_low_medium_models()
        case "gpt-medium-high":
            register_gpt_medium_high_models()
        case "gemini-low-medium":
            register_gemini_low_medium_models()
        case "gemini-medium-high":
            register_gemini_medium_high_models()
        case "gemini3-medium-high":
            register_gemini3_medium_high_models()
        case _:
            raise ValueError(f"Unknown model set: {name!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "AstroVisBench Workflow — generate the visualization image for "
            "each query via SciDER's Claude coding subagent. No code eval."
        ),
        prog="python -m bench_workflows.astrovisbench_workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--queries",
        "-q",
        required=True,
        help="Path to astrovisbench_queries.json",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory to write vis_*.png and results.json into",
    )
    parser.add_argument(
        "--bench-env",
        default=None,
        help="Path to the extracted bench_env directory (subprocess cwd). "
        "Many upstream queries load files via relative paths that require this.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter that has the AstroVisBench deps installed. "
        "Default is the current interpreter; usually you want "
        "benchmarks/astrovisbench/.venv/bin/python",
    )
    parser.add_argument(
        "--models",
        choices=[
            "gpt-low-medium",
            "gpt-medium-high",
            "gemini-low-medium",
            "gemini-medium-high",
            "gemini3-medium-high",
        ],
        default="gemini-medium-high",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N queries")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip queries with index < START (useful to resume)",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Only generate visualization_gen_code, do not run it",
    )
    parser.add_argument(
        "--add-vis-underspec",
        action="store_true",
        help="Include visualization_underspecifications in the prompt",
    )

    args = parser.parse_args()

    logger.info(f"Registering models: {args.models}")
    _register_models(args.models)

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(queries)} queries from {args.queries}")

    sliced = queries[args.start : (args.start + args.limit) if args.limit else None]
    logger.info(f"Processing {len(sliced)} queries (start={args.start}, limit={args.limit})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    # Resume support: load prior results if any.
    results: list[dict] = []
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        logger.info(f"Resuming from existing {results_path} ({len(results)} prior records)")
    done_indices = {r["index"] for r in results}

    for offset, q in enumerate(sliced):
        idx = args.start + offset
        if idx in done_indices:
            continue
        image_path = output_dir / f"vis_{idx:04d}.png"
        record: dict = {"index": idx}
        try:
            out = run_astrovisbench_workflow(
                query=q,
                output_image=image_path,
                bench_env=args.bench_env,
                python_exec=args.python,
                add_vis_underspec=args.add_vis_underspec,
                skip_render=args.skip_render,
            )
            record.update({"ok": True, **out})
        except Exception as e:
            logger.exception(f"Query {idx} failed: {e}")
            record.update({"ok": False, "error": str(e)})
        results.append(record)
        # Persist after every query so crashes don't lose progress.
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.get("ok"))
    logger.info(
        f"Done. {n_ok}/{len(results)} queries produced a visualization. "
        f"Results at {results_path}"
    )
