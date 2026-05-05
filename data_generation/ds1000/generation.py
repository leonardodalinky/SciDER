"""DS-1000 data-generation workflow.

Source: https://huggingface.co/datasets/xlangai/DS-1000

For each problem in the DS-1000 benchmark we hand the StackOverflow-style
prompt to SciDER's coding subagent and persist the full message history at
``<workspace>/coding_agent_history.json``. The trajectories become SFT data
for OpenSciDER (consumed later by ``train/prepare_data.py``).

We don't run the DS-1000 evaluator here — the goal is data, not metric. The
``code.py`` deliverable is preserved per workspace if you ever want to score
it later, but ``output.json.ok`` only requires that the file was written.

Usage
-----
    python -m data_generation.ds1000.generation \\
        --output-root data_generation/datasets/ds1000 \\
        --skip-existing \\
        --limit 50

    # Run only a specific library:
    python -m data_generation.ds1000.generation --library Pandas --limit 20 ...

uid format: ``ds1000_<library>_<problem_id>``  e.g. ``ds1000_Pandas_42``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from loguru import logger

# Make project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scider.default.models import register_defaults_from_yaml

from ._common import (
    DEFAULT_CODE_FILENAME,
    HISTORY_FILENAME,
    PROMPT_FILENAME,
    output_protocol_block,
    run_coding_task,
    scan_completed_uids,
    write_output_json,
)

# ---- Constants ----
HF_DATASET_ID = "xlangai/DS-1000"
ROLES_YAML_PATH = Path(__file__).parent / "roles.yaml"

# DS-1000 has a single split named "test"; this is the only one that exists
# upstream (it's a benchmark, no train split). We run "test" purely as the
# source of unsolved problems for our coding agent — there is no risk of
# train/test contamination because we don't evaluate against it here.
DEFAULT_SPLIT = "test"


def _slug(text: str) -> str:
    """k8s-safe slug for use in uid / dirname."""
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")
    return text or "unknown"


def _build_uid(library: str, problem_id: int | str) -> str:
    return f"ds1000_{_slug(library)}_{_slug(str(problem_id))}"


def _register_models_from_yaml() -> None:
    if not ROLES_YAML_PATH.exists():
        raise FileNotFoundError(f"Role yaml missing at {ROLES_YAML_PATH}")
    logger.info("Registering roles from {}", ROLES_YAML_PATH)
    registered = register_defaults_from_yaml(ROLES_YAML_PATH)
    logger.info("Registered {} roles: {}", len(registered), sorted(registered))


# --------------------------------------------------------------------------- #
# Dataset loading                                                             #
# --------------------------------------------------------------------------- #


def _load_ds1000(split: str = DEFAULT_SPLIT, library_filter: str | None = None) -> list[dict]:
    """Pull DS-1000 from HuggingFace, return a list of dict rows.

    Each row carries:
        - problem_id: int
        - prompt: str (StackOverflow-style code question)
        - reference_code: str (gold solution; we do NOT inject this into the
          agent's view — we persist it only for downstream offline analysis)
        - library: str (Pandas / NumPy / TensorFlow / PyTorch / SciPy /
          Sklearn / Matplotlib)
        - perturbation_type: str
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "DS-1000 data loading requires `datasets` (HuggingFace). Install with "
            "`uv add datasets` or `pip install datasets`."
        ) from e

    ds = load_dataset(HF_DATASET_ID, split=split)
    rows: list[dict] = []
    for row in ds:
        # `metadata` is a JSON-encoded string in DS-1000 — parse it.
        meta_raw = row.get("metadata")
        if isinstance(meta_raw, str):
            try:
                meta = json.loads(meta_raw)
            except json.JSONDecodeError:
                meta = {}
        elif isinstance(meta_raw, dict):
            meta = meta_raw
        else:
            meta = {}

        library = meta.get("library") or row.get("library") or "Unknown"
        if library_filter and library_filter.lower() != library.lower():
            continue

        rows.append(
            {
                "problem_id": meta.get("problem_id", row.get("problem_id", len(rows))),
                "library": library,
                "perturbation_type": meta.get("perturbation_type", ""),
                "prompt": row.get("prompt", ""),
                "reference_code": row.get("reference_code", ""),
                # Needed by data_generation.ds1000.eval — defines
                # test_execution / test_string for in-process scoring.
                "code_context": row.get("code_context", ""),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Prompt building                                                             #
# --------------------------------------------------------------------------- #


def _build_user_query(row: dict, code_filename: str) -> str:
    """Wrap the DS-1000 prompt with the upstream-faithful output contract.

    DS-1000 evaluates by substituting the model's solution into a pre-built
    test harness that already defines the input variables (e.g. ``df``,
    ``List``, ``X``, ``y``). The agent's deliverable must therefore be ONLY
    the snippet that fills the ``[insert]`` slot between BEGIN SOLUTION and
    END SOLUTION in the prompt — NOT a standalone script that re-imports
    libraries or re-defines the inputs (re-defining a non-deterministic
    input like ``np.random.permutation(...)`` will mismatch the harness's
    expected output and the eval will fail).
    """
    raw_prompt = (row.get("prompt") or "").strip()
    library = row.get("library", "Unknown")

    parts = [
        f"# DS-1000 Coding Task ({library})",
        "",
        "Read the StackOverflow-style problem below carefully, then write a "
        "SOLUTION SNIPPET that fills the `[insert]` placeholder between "
        "`BEGIN SOLUTION` and `END SOLUTION` in the prompt.",
        "",
        "## Problem",
        "",
        raw_prompt,
        "",
        "## Output protocol (READ CAREFULLY — different from a standalone script)",
        "",
        f"- Write ONLY the missing solution snippet to `./{code_filename}`. "
        "Do NOT include a `__main__` block or `print(...)` statements.",
        "- The setup code shown above (imports, `df = pd.DataFrame(...)`, "
        "etc.) is ALREADY EXECUTED before your snippet runs. Re-importing or "
        "re-defining those variables will SHADOW the test harness's inputs "
        "and break evaluation — especially when the setup uses randomness "
        "(e.g. `np.random.permutation(...)`).",
        "- Reference the variables defined in the prompt's setup block "
        "(typical names: `df`, `List`, `X`, `y`, `arr`, `model`, ...) and "
        "store the answer in the variable name the prompt specifies (almost "
        "always `result`).",
        f"- Self-check: open `./{code_filename}` and confirm it contains "
        "ONLY the lines that would go between `BEGIN SOLUTION` / "
        "`END SOLUTION`. If it has its own `import` lines or its own "
        "`df = ...` assignment, that's wrong.",
        "- For testing, you may write a SEPARATE scratch file (e.g. "
        "`scratch_test.py`) that defines the inputs and runs your snippet — "
        f"only `./{code_filename}` is consumed by the harness.",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Per-task runner                                                             #
# --------------------------------------------------------------------------- #


def run_one_task(
    *,
    row: dict,
    output_root: Path,
    code_filename: str = DEFAULT_CODE_FILENAME,
) -> dict:
    """Run one DS-1000 row, persist trajectory + output.json. Returns a record
    summarising the outcome (suitable for appending to results.json)."""
    uid = _build_uid(row["library"], row["problem_id"])
    workspace = output_root / uid
    workspace.mkdir(parents=True, exist_ok=True)

    user_query = _build_user_query(row, code_filename)
    (workspace / PROMPT_FILENAME).write_text(user_query, encoding="utf-8")

    logger.info("[{}] running coding task ({} chars prompt)", uid, len(user_query))
    code_text, history = run_coding_task(
        user_query=user_query,
        workspace_dir=workspace,
        code_filename=code_filename,
    )

    ok = bool(code_text.strip())
    record = {
        "uid": uid,
        "library": row["library"],
        "problem_id": row["problem_id"],
        "perturbation_type": row.get("perturbation_type", ""),
        "ok": ok,
        "code_chars": len(code_text),
        "n_messages": len(history),
        "history_path": str((workspace / HISTORY_FILENAME).relative_to(output_root)),
    }
    if not ok:
        record["error"] = f"agent did not produce a non-empty {code_filename}"
    write_output_json(workspace, record)
    logger.info("[{}] done — ok={} code_chars={} msgs={}", uid, ok, len(code_text), len(history))
    return record


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SFT trajectories from DS-1000 via SciDER's coding subagent. "
            "Each problem produces a workspace under <output_root>/<uid>/ "
            "containing prompt.md, code.py, coding_agent_history.json (the "
            "trajectory you train on), and output.json (skip-existing marker)."
        ),
        prog="python -m data_generation.ds1000.generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        "-o",
        required=True,
        help="Directory under which per-problem workspaces are created " "(<output_root>/<uid>/).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"DS-1000 split (default: {DEFAULT_SPLIT}; the only split upstream).",
    )
    parser.add_argument(
        "--library",
        default=None,
        help="Filter to a single library (Pandas / NumPy / TensorFlow / "
        "PyTorch / SciPy / Sklearn / Matplotlib). Case-insensitive.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip uids whose <output_root>/<uid>/output.json marks the run "
        "complete (ok=true + non-empty code.py + non-empty history).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most the first N tasks (after --skip-existing / "
        "--library filter). Omit to run all.",
    )
    parser.add_argument(
        "--uids",
        default=None,
        help="Comma-separated list of uids to run; overrides --limit. "
        "Useful for targeted debug runs.",
    )
    parser.add_argument(
        "--code-filename",
        default=DEFAULT_CODE_FILENAME,
        help=f"Override the deliverable filename (default {DEFAULT_CODE_FILENAME}).",
    )
    args = parser.parse_args()

    # 1. Register models first so config errors surface before any I/O.
    _register_models_from_yaml()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 2. Load + filter dataset rows.
    logger.info("Loading {} (split={}, library={})", HF_DATASET_ID, args.split, args.library)
    rows = _load_ds1000(args.split, library_filter=args.library)
    logger.info("Discovered {} rows", len(rows))

    # 3. Annotate uids and apply --skip-existing.
    for row in rows:
        row["_uid"] = _build_uid(row["library"], row["problem_id"])
    if args.skip_existing:
        completed = scan_completed_uids(output_root, code_filename=args.code_filename)
        if completed:
            logger.info("--skip-existing: {} uids already complete; skipping them", len(completed))
            rows = [r for r in rows if r["_uid"] not in completed]

    # 4. --uids takes precedence over --limit.
    if args.uids:
        wanted = {u.strip() for u in args.uids.split(",") if u.strip()}
        selected = [r for r in rows if r["_uid"] in wanted]
        missing = wanted - {r["_uid"] for r in selected}
        if missing:
            logger.warning("Requested uids not in candidate set: {}", sorted(missing))
    elif args.limit is not None:
        selected = rows[: args.limit]
    else:
        selected = rows
    logger.info("Processing {} tasks", len(selected))

    # 5. Resume results.json across runs (dedup by uid).
    results_path = output_root / "results.json"
    results: list[dict] = []
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
            logger.info("Resuming from {} ({} prior records)", results_path, len(results))
        except json.JSONDecodeError:
            logger.warning("Existing results.json is malformed; starting fresh")

    for row in selected:
        record = run_one_task(row=row, output_root=output_root, code_filename=args.code_filename)
        results = [r for r in results if r.get("uid") != record["uid"]]
        results.append(record)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.get("ok"))
    logger.info(
        "Done. {}/{} produced a non-empty solution. Trajectories under {}",
        n_ok,
        len(results),
        output_root,
    )


if __name__ == "__main__":
    _main()
