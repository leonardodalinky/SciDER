"""AI-Idea-Bench data-generation workflow.

Source: https://huggingface.co/datasets/yanshengqiu/AI_Idea_Bench_2025

For each row in the benchmark we hand the paper's research seed (topic +
motivation) to SciDER's IdeationAgent and persist the full message history
at ``<workspace>/ideation_agent_history.json``. The trajectories become SFT
data for OpenSciDER (consumed later by ``train/prepare_data.py``).

We don't run the upstream AI-Idea-Bench evaluator here — the goal is data,
not metric. Every generated trajectory is kept (no quality filter); the
``output.json`` ``ok`` flag merely tracks whether the workflow ran to
completion (status=success + non-empty history).

uid format: ``aiidea_<index>``  e.g. ``aiidea_13``.
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
    HISTORY_FILENAME,
    PROMPT_FILENAME,
    run_ideation_task,
    scan_completed_uids,
    write_output_json,
)

# ---- Constants ----
HF_DATASET_ID = "yanshengqiu/AI_Idea_Bench_2025"
ROLES_YAML_PATH = Path(__file__).parent / "roles.yaml"

# AI-Idea-Bench ships a single split called "test". It is the source of
# research seeds for our IdeationAgent — we don't evaluate against it, so
# there is no train/test contamination concern.
DEFAULT_SPLIT = "test"


def _slug(text: str) -> str:
    """k8s-safe slug for use in uid / dirname."""
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")
    return text or "unknown"


def _build_uid(index: int | str) -> str:
    return f"aiidea_{_slug(str(index))}"


def _register_models_from_yaml() -> None:
    if not ROLES_YAML_PATH.exists():
        raise FileNotFoundError(f"Role yaml missing at {ROLES_YAML_PATH}")
    logger.info("Registering roles from {}", ROLES_YAML_PATH)
    registered = register_defaults_from_yaml(ROLES_YAML_PATH)
    logger.info("Registered {} roles: {}", len(registered), sorted(registered))


# --------------------------------------------------------------------------- #
# Dataset loading                                                             #
# --------------------------------------------------------------------------- #


def _load_aiidea(split: str = DEFAULT_SPLIT) -> list[dict]:
    """Pull AI-Idea-Bench from HuggingFace, return a list of dict rows.

    Each row carries the fields we need to build the ideation seed prompt:
        - index: int (unique identifier upstream)
        - target_paper: str (paper title — used as research_domain hint)
        - topic: str (one-line topic blurb from summary.topic)
        - revised_topic: str | None (a generalised, less-leaky topic
          phrasing — preferred over ``topic`` when present because ``topic``
          often paraphrases the paper's contribution)
        - motivation: str (the problem the paper addresses; this is the
          actual ideation seed)

    We deliberately do NOT extract ``summary.method`` — that's the ground-
    truth solution and would defeat the purpose of asking the agent to
    ideate.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "AI-Idea-Bench data loading requires `datasets` (HuggingFace). Install with "
            "`uv add datasets` or `pip install datasets`."
        ) from e

    ds = load_dataset(HF_DATASET_ID, split=split)
    rows: list[dict] = []
    for row in ds:
        summary = row.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        rows.append(
            {
                "index": row.get("index", len(rows)),
                "target_paper": row.get("target_paper", "") or "",
                "topic": summary.get("topic", "") or "",
                "revised_topic": summary.get("revised_topic") or None,
                "motivation": summary.get("motivation", "") or "",
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Prompt building                                                             #
# --------------------------------------------------------------------------- #


def _build_user_query(row: dict) -> str:
    """Compose the ideation seed prompt.

    We give the agent:
      - A high-level research area (revised_topic, falling back to topic)
      - The motivating problem (motivation field — the gap the paper
        identifies)

    We deliberately withhold ``target_paper`` (title) and ``method``
    (solution) so the agent ideates from the problem statement rather than
    reverse-engineering the paper.
    """
    motivation = (row.get("motivation") or "").strip()
    revised = (row.get("revised_topic") or "").strip()
    topic = (row.get("topic") or "").strip()
    research_area = revised or topic or "an unspecified research area"

    parts = [
        "# Research Ideation Task",
        "",
        "You are given a research area and a motivating problem statement. "
        "Your job is to brainstorm novel research ideas that could plausibly "
        "address this gap. Use SciDER's literature-search tools to ground "
        "your ideas in prior work, then propose 2–4 distinct ideas with "
        "clear novelty justification.",
        "",
        "## Research area",
        "",
        research_area,
        "",
        "## Motivating problem",
        "",
        motivation or "(no motivation supplied — generate ideas from the research area alone)",
        "",
        "## What to deliver",
        "",
        "- Survey adjacent literature briefly to anchor the ideation in the " "current landscape.",
        "- Propose 2–4 candidate research ideas. For each, state: (1) the "
        "core mechanism, (2) why it is novel relative to what you found in "
        "the literature, (3) a sketch of how you would evaluate it.",
        "- The agent's standard ideation pipeline (idea generation → novelty "
        "scoring → final report) is the expected end state.",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Per-task runner                                                             #
# --------------------------------------------------------------------------- #


def run_one_task(
    *,
    row: dict,
    output_root: Path,
    recursion_limit: int = 50,
) -> dict:
    """Run one AI-Idea-Bench row, persist trajectory + output.json.
    Returns a record summarising the outcome (suitable for appending to
    results.json)."""
    uid = _build_uid(row["index"])
    workspace = output_root / uid
    workspace.mkdir(parents=True, exist_ok=True)

    user_query = _build_user_query(row)
    (workspace / PROMPT_FILENAME).write_text(user_query, encoding="utf-8")

    research_domain = (row.get("revised_topic") or row.get("topic") or "").strip() or None

    logger.info("[{}] running ideation task ({} chars prompt)", uid, len(user_query))
    workflow = run_ideation_task(
        user_query=user_query,
        workspace_dir=workspace,
        research_domain=research_domain,
        recursion_limit=recursion_limit,
    )

    history = workflow.ideation_agent_history or []
    n_ideas = len(workflow.research_ideas or [])
    status = workflow.final_status or "unknown"
    ok = status == "success" and len(history) > 0

    record = {
        "uid": uid,
        "index": row["index"],
        "target_paper": row.get("target_paper", ""),
        "ok": ok,
        "status": status,
        "n_ideas": n_ideas,
        "novelty_score": workflow.novelty_score,
        "n_messages": len(history),
        "history_path": str((workspace / HISTORY_FILENAME).relative_to(output_root)),
    }
    if not ok:
        record["error"] = (
            workflow.error_message or f"workflow status={status}, history_len={len(history)}"
        )
    write_output_json(workspace, record)
    logger.info(
        "[{}] done — ok={} status={} ideas={} msgs={}",
        uid,
        ok,
        status,
        n_ideas,
        len(history),
    )
    return record


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SFT trajectories from AI-Idea-Bench via SciDER's "
            "IdeationAgent. Each row produces a workspace under "
            "<output_root>/<uid>/ containing prompt.md, "
            "ideation_agent_history.json (the trajectory you train on), "
            "ideation_summary.md, and output.json (skip-existing marker)."
        ),
        prog="python -m data_generation.aiidea.generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        "-o",
        required=True,
        help="Directory under which per-row workspaces are created " "(<output_root>/<uid>/).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"AI-Idea-Bench split (default: {DEFAULT_SPLIT}; the only split upstream).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip uids whose <output_root>/<uid>/output.json marks the run "
        "complete (ok=true + non-empty history).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most the first N rows (after --skip-existing). Omit to run all.",
    )
    parser.add_argument(
        "--uids",
        default=None,
        help="Comma-separated list of uids to run; overrides --limit. "
        "Useful for targeted debug runs.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=50,
        help="LangGraph recursion limit passed to the IdeationAgent (default 50).",
    )
    args = parser.parse_args()

    # 1. Register models first so config errors surface before any I/O.
    _register_models_from_yaml()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset.
    logger.info("Loading {} (split={})", HF_DATASET_ID, args.split)
    rows = _load_aiidea(args.split)
    logger.info("Discovered {} rows", len(rows))

    # 3. Annotate uids and apply --skip-existing.
    for row in rows:
        row["_uid"] = _build_uid(row["index"])
    if args.skip_existing:
        completed = scan_completed_uids(output_root)
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
        record = run_one_task(
            row=row,
            output_root=output_root,
            recursion_limit=args.recursion_limit,
        )
        results = [r for r in results if r.get("uid") != record["uid"]]
        results.append(record)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.get("ok"))
    logger.info(
        "Done. {}/{} produced a non-empty ideation trajectory. Trajectories under {}",
        n_ok,
        len(results),
        output_root,
    )


if __name__ == "__main__":
    _main()
