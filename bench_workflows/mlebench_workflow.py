"""MLE-Bench Workflow — run SciDER's FullWorkflow on an MLE-Bench competition.

MLE-Bench hands the agent three files per task:

* ``instructions.md`` — generic benchmark instructions (submission path,
  validator endpoint, no-cheating rules, ...); becomes part of ``user_query``.
* ``description.md`` — competition-specific problem statement (domain,
  metric, dataset layout); becomes ``data_desc``.
* ``/home/data/`` — the competition's prepared data directory.

This wrapper additionally builds a **strategy preamble** — analogous to the
airsbench workflow — that inlines the grader source, Kaggle leaderboard
thresholds, and submission-format sanity checks BEFORE the stock instructions.
Those extras come from the mle-bench source tree that the base image copies
to ``/mlebench/`` at build time, keyed off the ``$COMPETITION_ID`` env var
that ``mle-bench``'s harness sets on the container (``agents/run.py:130``).

Usage
-----
    python -m bench_workflows.mlebench_workflow \\
        --instructions competition/instructions.md \\
        --description  competition/description.md \\
        --data         competition/data \\
        --workspace    workspace
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from loguru import logger

# Make scider + bench modules importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scider.default.models import register_defaults_from_yaml
from scider.workflows.full_workflow import run_full_workflow

# ---- Constants ----
# The per-project model-role assignment for this benchmark. Edit the yaml to
# swap models — don't reintroduce a --models CLI knob; this matches the
# airsbench / astrovisbench refactor.
ROLES_YAML_PATH = Path(__file__).parent / "model_configs" / "mlebench_roles.yaml"

# Where mle-bench's base image puts its own source tree. We read per-competition
# config.yaml / grade.py / leaderboard.csv from under here to build the
# preamble. Outside the container this path won't exist — the preamble builder
# degrades gracefully in that case.
MLE_BENCH_SRC = Path("/mlebench/mlebench/competitions")

# Hard-coded paths that mle-bench mounts on the container. We don't derive
# these from CLI flags because the values are baked into mle-bench's harness
# (environment/entrypoint.sh, instructions.txt, agents/run.py).
MLEBENCH_DATA_DIR = Path("/home/data")
MLEBENCH_SUBMISSION_PATH = Path("/home/submission/submission.csv")
MLEBENCH_VALIDATE_SCRIPT = Path("/home/validate_submission.sh")

# How many top-N leaderboard rows to quote in the preamble. Kaggle leaderboards
# can be thousands of rows; 5 top scores + a medal threshold is enough context
# for the agent and keeps the preamble compact.
LEADERBOARD_TOP_N = 5


def _register_models_from_yaml() -> None:
    """Load role assignments from the mlebench roles yaml."""
    if not ROLES_YAML_PATH.exists():
        raise FileNotFoundError(
            f"Role yaml missing at {ROLES_YAML_PATH}. This file pins the "
            "experiment / critic / approval / ... roles for MLE-Bench; "
            "don't rename it."
        )
    logger.info("Registering roles from {}", ROLES_YAML_PATH)
    registered = register_defaults_from_yaml(ROLES_YAML_PATH)
    logger.info("Registered {} roles: {}", len(registered), sorted(registered))


# --------------------------------------------------------------------------- #
# Preamble helpers                                                            #
# --------------------------------------------------------------------------- #


def _safe_read_text(path: Path, max_bytes: int | None = None) -> str | None:
    """Read ``path`` as UTF-8 text. Returns ``None`` if missing / unreadable.

    ``max_bytes`` caps the read size — leaderboard.csv can be >1MB on popular
    competitions; we only need the head rows so clamp early.
    """
    try:
        if not path.is_file():
            return None
        if max_bytes is not None:
            with path.open("rb") as f:
                return f.read(max_bytes).decode("utf-8", errors="replace")
        return path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Could not read {}: {}", path, e)
        return None


def _summarize_leaderboard(csv_text: str) -> dict | None:
    """Pull top-N + medal-threshold scores out of Kaggle's leaderboard.csv.

    Columns are typically ``scoreNullable,teamId,hasTeamName,submissionDate,
    score,hasScore`` — ordered by Kaggle's final ranking (best first), so
    row 0 is the winner regardless of metric direction. We don't try to
    classify higher-is-better vs lower-is-better; the leaderboard ordering
    is already authoritative.
    """
    import csv
    from io import StringIO

    try:
        reader = csv.DictReader(StringIO(csv_text))
        rows = list(reader)
    except csv.Error as e:
        logger.warning("Leaderboard CSV parse failed: {}", e)
        return None
    if not rows:
        return None

    def _score(row: dict) -> str | None:
        s = row.get("score") or row.get("scoreNullable") or ""
        s = s.strip()
        return s or None

    top_scores = [s for s in (_score(r) for r in rows[:LEADERBOARD_TOP_N]) if s]

    # Kaggle medal thresholds: gold ≈ top 10, silver ≈ top 50, bronze ≈ top 100
    # (actual rules depend on pool size, but these ranks are a useful anchor).
    def _row_at(rank: int) -> str | None:
        idx = rank - 1
        return _score(rows[idx]) if 0 <= idx < len(rows) else None

    return {
        "total_teams": len(rows),
        "top_scores": top_scores,
        "gold_threshold": _row_at(10),
        "silver_threshold": _row_at(50),
        "bronze_threshold": _row_at(100),
        "median": _row_at(len(rows) // 2) if rows else None,
        "worst": _score(rows[-1]),
    }


def _summarize_sample_submission(path: Path) -> dict | None:
    """Return {'rows': N, 'columns': [...], 'head': "<first 3 lines>"} or None."""
    text = _safe_read_text(path, max_bytes=8192)
    if text is None:
        return None
    lines = text.splitlines()
    if not lines:
        return None
    header = lines[0].split(",")
    # Row count: count all lines in the file, not just the capped read —
    # re-scan cheaply to get the true row count.
    try:
        with path.open("rb") as f:
            total_rows = sum(1 for _ in f) - 1  # minus header
    except OSError:
        total_rows = max(0, len(lines) - 1)
    head = "\n".join(lines[:3])
    return {"rows": total_rows, "columns": header, "head": head}


def _fmt_scores(scores: list[str]) -> str:
    return ", ".join(f"`{s}`" for s in scores) if scores else "(none)"


def _build_preamble(
    competition_id: str | None,
    mle_bench_src: Path = MLE_BENCH_SRC,
    data_dir: Path = MLEBENCH_DATA_DIR,
) -> str:
    """Assemble a strategy preamble. Returns ``""`` if no sources are available.

    Degrades gracefully: any missing file just drops its section. On local dev
    (no /mlebench/ tree, no $COMPETITION_ID) every section is empty and the
    function returns ``""`` — the workflow then behaves identically to before.
    """
    if not competition_id:
        logger.info("$COMPETITION_ID not set — skipping strategy preamble.")
        return ""

    comp_dir = mle_bench_src / competition_id
    if not comp_dir.is_dir():
        logger.info("Competition source dir missing at {} — skipping preamble.", comp_dir)
        return ""

    # --- 1. Competition metadata ------------------------------------------
    cfg_text = _safe_read_text(comp_dir / "config.yaml")
    cfg: dict = {}
    if cfg_text:
        try:
            cfg = yaml.safe_load(cfg_text) or {}
        except yaml.YAMLError as e:
            logger.warning("config.yaml parse failed: {}", e)

    name = cfg.get("name") or competition_id
    grader_name = (cfg.get("grader") or {}).get("name") or "<unknown>"
    comp_type = cfg.get("competition_type") or "<unknown>"
    awards_medals = cfg.get("awards_medals")

    # --- 2. Grader source -------------------------------------------------
    grader_src = _safe_read_text(comp_dir / "grade.py")

    # --- 3. Leaderboard summary ------------------------------------------
    lb_text = _safe_read_text(comp_dir / "leaderboard.csv", max_bytes=256_000)
    lb = _summarize_leaderboard(lb_text) if lb_text else None

    # --- 4. Sample submission shape --------------------------------------
    sample_info = _summarize_sample_submission(data_dir / "sample_submission.csv")

    # --- Assemble ---------------------------------------------------------
    parts: list[str] = [
        "# Solution Strategy (read before the task spec below)",
        "",
        "## Competition at a glance",
        "",
        f"- **Name**: {name}",
        f"- **ID**: `{competition_id}`",
        f"- **Metric**: `{grader_name}` (see grader source below — this is "
        "EXACTLY how your submission is scored)",
        f"- **Competition type**: {comp_type}",
    ]
    if awards_medals is not None:
        parts.append(f"- **Awards medals**: {awards_medals}")

    # Grader source block — the single highest-value thing the agent can see.
    if grader_src:
        parts += [
            "",
            "## Grader source — this is EXACTLY how your submission is scored",
            "",
            "Copy the ``grade()`` function below verbatim into your own code, "
            "run it on a held-out slice of the training set (with labels you "
            "know), and use the result as a proxy for the real score BEFORE "
            "writing your final submission. This is the fastest way to catch "
            "silent failures (predicting the wrong column, collapsed output, "
            "row-count mismatch, ...) that the validator alone won't catch.",
            "",
            "```python",
            grader_src.strip(),
            "```",
        ]

    # Leaderboard anchor — gives the agent a concrete target.
    if lb:
        parts += [
            "",
            "## Kaggle leaderboard reference (from the original comp)",
            "",
            f"- **Total teams on leaderboard**: {lb['total_teams']}",
            f"- **Top-{LEADERBOARD_TOP_N} scores**: {_fmt_scores(lb['top_scores'])}",
            f"- **Gold medal (rank 10)**: `{lb['gold_threshold']}`",
            f"- **Silver medal (rank 50)**: `{lb['silver_threshold']}`",
            f"- **Bronze medal (rank 100)**: `{lb['bronze_threshold']}`",
            f"- **Median score**: `{lb['median']}`",
            f"- **Worst score**: `{lb['worst']}`",
            "",
            "These numbers are from the ORIGINAL Kaggle competition and are "
            "routinely beatable by modern methods. Your first WebSearch query "
            f"should be: `{name} kaggle top solution writeup`. Skim 1-2 of the "
            "top-1/top-10 writeups before designing your own pipeline.",
        ]

    # Submission format + row-count guard.
    if sample_info:
        cols = ", ".join(f"`{c}`" for c in sample_info["columns"])
        parts += [
            "",
            "## Submission format (from `/home/data/sample_submission.csv`)",
            "",
            f"- **Output path** (HARD REQUIREMENT): `{MLEBENCH_SUBMISSION_PATH}` — "
            "mle-bench's grader looks here exclusively. A submission anywhere "
            "else (workspace root, /tmp, ...) will score 0.",
            f"- **Expected data rows**: {sample_info['rows']} (plus 1 header)",
            f"- **Columns** (exact names, exact order): {cols}",
            "",
            "First 3 lines of the sample submission:",
            "",
            "```",
            sample_info["head"],
            "```",
            "",
            "Row-count sanity check before declaring done:",
            "",
            "```bash",
            f"wc -l {MLEBENCH_SUBMISSION_PATH}  # must equal {sample_info['rows'] + 1}",
            f"head -3 {MLEBENCH_SUBMISSION_PATH}",
            f"bash {MLEBENCH_VALIDATE_SCRIPT} {MLEBENCH_SUBMISSION_PATH}",
            "```",
            "",
            f"If `{MLEBENCH_VALIDATE_SCRIPT}` returns an error, your "
            "submission is malformed — fix it before ending the run.",
        ]

    # Generic training recipe — same spirit as airsbench's preamble but scoped
    # to what's true across MLE-bench task types (CV / NLP / tabular / time-series).
    parts += [
        "",
        "## How to solve this (default approach)",
        "",
        "1. **Read `/home/data/description.md` end-to-end first.** Understand "
        "the domain, data layout, and metric direction (higher-is-better vs "
        "lower-is-better) before writing code.",
        "2. **Hunt for a recent SOTA writeup.** Kaggle comps often have "
        "public top-5 solutions on Kaggle Discussions / GitHub. "
        f'`WebSearch("{name} kaggle 1st place solution")` is the fastest '
        "path to a strong baseline. For newer comps add year: "
        '`"<name> 2024 2025 solution"`.',
        "3. **Baseline first, optimise second.** Write the simplest possible "
        "pipeline that produces a VALID `submission.csv` (correct rows, "
        "correct columns, self-score above worst). Run "
        f"`{MLEBENCH_VALIDATE_SCRIPT}` on it. ONLY after that works, iterate "
        "on model quality.",
        "4. **Prefer fine-tuning a pretrained checkpoint** over training from "
        "scratch — HuggingFace Hub has a pretrained model for almost every "
        "task type in MLE-bench (timm for vision, transformers for NLP, "
        "tabnet/xgboost for tabular, etc.).",
        "5. **Hold out a validation split** from the training data; NEVER "
        "train on it. Self-score with the grader's `grade()` function "
        "verbatim on this held-out split.",
        "",
        "## Framework choice — use PyTorch, avoid TensorFlow",
        "",
        "The container ships with **PyTorch 2.10 pre-installed on GPU** "
        "(NVIDIA NGC build with sm_121 / CUDA 13.0 support). TensorFlow, "
        "when present, is CPU-only on this image — its GPU extras do NOT "
        "support the Blackwell GB10 on DGX Spark.",
        "",
        "- **Default to `torch` for everything** — modelling, training, "
        "inference, even for tasks whose canonical Kaggle solutions used "
        "TF. There is almost always a timm / transformers / segmentation-"
        "models-pytorch / torchvision equivalent of any TF/Keras recipe "
        "you might find in a writeup.",
        "- **If you must install torch yourself** (e.g. a dep pulls it in "
        "a venv or subprocess), **pin `torch>=2.9`**. Anything older than "
        "2.7 lacks sm_121 kernels and will either crash with `no kernel "
        "image is available for execution on the device` or silently fall "
        "back to CPU (giving you the worst-of-both-worlds: slow runtime "
        "with no error).",
        "- **Do NOT install `tensorflow[and-cuda]`** on this image — its "
        "CUDA-runtime pins (CUDA 12.x userspace) don't work on sm_121. "
        "Plain `tensorflow` (CPU) is fine for quick data pre-processing "
        "(e.g. reading `TFRecord` files) but not for training.",
        "- **Keras 3 is wired to the torch backend** via "
        "`KERAS_BACKEND=torch` env var. If you absolutely need Keras API, "
        "it will run GPU-accelerated on top of torch without extra work.",
        '- **If a competition writeup says "we used TensorFlow"** — '
        "translate the architecture to PyTorch (e.g. `tf.keras.applications."
        "EfficientNetB3` → `timm.create_model('efficientnet_b3', "
        "pretrained=True)`). Don't install TF-GPU.",
        "",
        "## Training recipe defaults (avoid common pitfalls)",
        "",
        "### Numerical precision — use bf16, NOT fp16",
        "",
        "- **When in doubt, `bf16=True` over `fp16=True`** for any modern "
        "encoder (ModernBERT, DeBERTa-v3, RoBERTa) or any causal LM on "
        "Ampere/Hopper/Blackwell. bf16 has the same dynamic range as fp32 "
        "and avoids unscale / overflow issues.",
        "- **DeBERTa + fp16 is infamously broken** — produces `ValueError: "
        "Attempting to unscale FP16 gradients` or NaN loss within the first "
        "epoch. Use bf16.",
        "- Only use fp16 on pre-Ampere GPUs (V100 / T4 / P100). Anything "
        "A100 / L4 / H100 / GB10 / GH200 → bf16.",
        "",
        "### transformers Trainer API (≥4.46)",
        "",
        "- The `tokenizer=` kwarg to `Trainer(...)` was renamed to "
        "`processing_class=`. If you see `TypeError: Trainer.__init__() got "
        "an unexpected keyword argument 'tokenizer'`, rename it.",
        "- `overwrite_output_dir` was dropped from `TrainingArguments`; "
        "remove it if you see a related TypeError.",
        "",
        "### When training fails — read the error before downgrading",
        "",
        "Do NOT immediately switch to a smaller model. Map symptoms to fixes:",
        "",
        "1. `ValueError: Attempting to unscale FP16 gradients` / NaN loss → "
        "change `fp16=True` to `bf16=True`.",
        "2. `TypeError: Trainer got unexpected keyword` → see API notes above.",
        "3. `CUDA out of memory` (literal string — don't assume OOM without "
        "seeing it!) → in order: enable `gradient_checkpointing=True`, "
        "reduce `per_device_train_batch_size` + raise "
        "`gradient_accumulation_steps`, shorten `max_length` / image size, "
        "enable bf16. Only after all those fail, consider a smaller model.",
        "4. Task killed with exit code -15 (SIGTERM) → likely a wall-clock "
        "timeout, not a memory issue. Reduce epochs / increase batch to "
        "finish faster.",
        "",
        "## Output protocol (HARD gates)",
        "",
        f"- Final submission lives at: `{MLEBENCH_SUBMISSION_PATH}` (exact path).",
        f"- Run `{MLEBENCH_VALIDATE_SCRIPT} {MLEBENCH_SUBMISSION_PATH}` and "
        "confirm it responds `valid`.",
        "- Row count + column names match the sample submission above.",
        "- Self-score on a held-out slice with the grader's `grade()` "
        "function before finishing — if the number is near worst, iterate.",
        "",
        "---",
        "",
        "(The generic MLE-bench benchmark instructions follow below.)",
        "",
        "",
    ]
    return "\n".join(parts)


def build_mlebench_user_query(
    instructions_path: Path,
    description_path: Path,
    competition_id: str | None,
) -> tuple[str, str]:
    """Read instructions.md + description.md + build strategy preamble.

    Returns ``(user_query, data_desc)``. The preamble (if any) is prepended
    to ``user_query`` so the experiment agent sees it before the generic
    benchmark rules.
    """
    if not instructions_path.exists():
        raise FileNotFoundError(f"Instructions file not found: {instructions_path}")
    if not description_path.exists():
        raise FileNotFoundError(f"Description file not found: {description_path}")

    preamble = _build_preamble(competition_id)
    instructions = instructions_path.read_text(encoding="utf-8")
    description = description_path.read_text(encoding="utf-8")
    return preamble + instructions, description


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MLE-Bench workflow — runs SciDER's FullWorkflow (data + experiment) "
            "on one MLE-Bench competition. The deliverable is "
            "/home/submission/submission.csv, which MLE-Bench's grader picks up."
        ),
        prog="python -m bench_workflows.mlebench_workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instructions",
        "-i",
        required=True,
        help="Path to instructions.md (task-specific instructions, used as user_query).",
    )
    parser.add_argument(
        "--description",
        "-d",
        required=True,
        help="Path to description.md (task background, used as data_desc).",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the competition data directory (mounted at /home/data in the container).",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        required=True,
        help="Workspace directory for the experiment (submission.csv lands at <workspace>/).",
    )
    parser.add_argument(
        "--competition-id",
        default=None,
        help="MLE-Bench competition id (e.g. ``aerial-cactus-identification``). "
        "Defaults to $COMPETITION_ID (set by mle-bench's harness on the "
        "container). Used to locate config.yaml / grade.py / leaderboard.csv "
        "under /mlebench/ for the strategy preamble. If unset AND env var "
        "missing, the preamble is skipped and the workflow behaves as before.",
    )
    parser.add_argument(
        "--repo-source",
        default=None,
        help="Optional repository source (local path or git URL) for the experiment agent.",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=3,
        help="Maximum critic/approval revision loops (default 3).",
    )
    parser.add_argument(
        "--data-recursion-limit",
        type=int,
        default=512,
        help="Recursion limit for DataAgent (default 512).",
    )
    parser.add_argument(
        "--experiment-recursion-limit",
        type=int,
        default=512,
        help="Recursion limit for ExperimentAgent (default 512).",
    )
    args = parser.parse_args()

    # 1. Register models first so any config error surfaces before we start work.
    _register_models_from_yaml()

    # 2. Build user query + data description from the MLE-Bench files.
    competition_id = args.competition_id or os.environ.get("COMPETITION_ID")
    logger.info("Building user query from MLE-Bench task files (comp={})...", competition_id)
    user_query, data_desc = build_mlebench_user_query(
        instructions_path=Path(args.instructions),
        description_path=Path(args.description),
        competition_id=competition_id,
    )
    logger.info("user_query: {} chars; data_desc: {} chars", len(user_query), len(data_desc))

    # 3. Run FullWorkflow.
    result = run_full_workflow(
        data_path=args.data,
        workspace_path=args.workspace,
        user_query=user_query,
        data_desc=data_desc,
        repo_source=args.repo_source,
        max_revisions=args.max_revisions,
        data_agent_recursion_limit=args.data_recursion_limit,
        experiment_agent_recursion_limit=args.experiment_recursion_limit,
    )

    # FullWorkflow persists sub-workflow summaries as it goes; also dump the
    # top-level summary so the grader / debug scripts have it co-located.
    result.save_summary()
    logger.info("Final status: {}", result.final_status)
    print(f"\nStatus: {result.final_status}")


if __name__ == "__main__":
    _main()
