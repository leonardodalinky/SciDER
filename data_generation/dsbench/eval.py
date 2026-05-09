"""DSBench modeling evaluator — score the submissions produced by
``data_generation.dsbench.generation`` against upstream's per-competition
eval scripts.

Analysis tasks are NOT scored here (no GPT-as-judge); they keep just the
``ok`` flag from generation.

Each modeling task carries its own metric (accuracy / RMSLE / log loss /
…). Upstream packages a separate ``<comp>_eval.py`` per competition under
``data_modeling/evaluation/``; we copied all 75 verbatim into
``data_generation/dsbench/upstream_eval_modeling/`` and invoke them via
subprocess::

    python <upstream>/<comp>_eval.py \\
        --answer_file <bench_root>/data_modeling/data/answers/<comp>/test_answer.csv \\
        --predict_file <workspace>/submission.csv \\
        --path <tmp> --name <comp>

The script writes a single score to ``<tmp>/<comp>/result.txt``; we read
that number back, store it in the workspace's ``output.json`` as
``score``, and set ``passed=True`` iff:
    1. ``submission.csv`` exists in the workspace,
    2. the eval subprocess returned 0 within the timeout,
    3. the resulting score is a finite number (catches NaN / Inf from
       broken submissions like all-zero predictions).

We do NOT auto-threshold "passed" by metric value — different competitions
have wildly different scales and "higher / lower is better" conventions.
Filter by the recorded ``score`` in your own post-processing if you want a
quality floor.

Usage::

    python -m data_generation.dsbench.eval \\
        --bench-root <dsbench-data dir> \\
        --output-root <out>

    # Re-evaluate every modeling workspace
    python -m data_generation.dsbench.eval -o ... --rerun

    # Score a subset only
    python -m data_generation.dsbench.eval -o ... --uids dsbench_modeling_titanic
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from loguru import logger

# Make project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ._common import MODELING_SUBMISSION_FILENAME, OUTPUT_FILENAME, write_output_json

UPSTREAM_EVAL_DIR = Path(__file__).parent / "upstream_eval_modeling"
EVAL_STDERR_KEEP_CHARS = 2000
DEFAULT_TIMEOUT = 60


def _comp_from_uid(uid: str) -> str | None:
    """``dsbench_modeling_titanic`` → ``titanic``. Non-modeling uids return None."""
    prefix = "dsbench_modeling_"
    return uid.removeprefix(prefix) if uid.startswith(prefix) else None


def _parse_score(text: str) -> float | None:
    """Upstream eval scripts write a single bare number to result.txt
    (e.g. ``0.7654`` for accuracy, ``1.234`` for RMSLE). Some old scripts
    may include trailing whitespace or scientific notation."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _run_one_eval(
    *,
    workspace: Path,
    comp: str,
    bench_root: Path,
    timeout: int,
) -> dict:
    """Run the upstream ``<comp>_eval.py`` against this workspace's
    ``submission.csv``. Returns a partial record dict to merge into
    ``output.json``: keys ``passed`` (bool), ``score`` (float|None),
    ``eval_stderr`` (str on failure), ``metric_unit`` (str best-effort)."""
    submission = workspace / MODELING_SUBMISSION_FILENAME
    eval_script = UPSTREAM_EVAL_DIR / f"{comp}_eval.py"
    answer_file = bench_root / "data_modeling" / "data" / "answers" / comp / "test_answer.csv"

    if not submission.is_file():
        return {
            "passed": False,
            "score": None,
            "eval_stderr": f"<no {MODELING_SUBMISSION_FILENAME} in workspace>",
        }
    if not eval_script.is_file():
        return {
            "passed": False,
            "score": None,
            "eval_stderr": f"<upstream eval script missing: {eval_script.name}>",
        }
    if not answer_file.is_file():
        return {
            "passed": False,
            "score": None,
            "eval_stderr": f"<gold answers missing: {answer_file}>",
        }

    # The upstream scripts open ``<path>/<name>/result.txt`` directly without
    # mkdir-ing the parent — upstream's ``score4each_com.py`` creates that
    # subdir BEFORE invoking the eval script. We mirror that here. Use a tmp
    # dir per call so concurrent eval is safe.
    with tempfile.TemporaryDirectory(prefix=f"dsbench_eval_{comp}_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / comp).mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(eval_script),
                    "--answer_file",
                    str(answer_file),
                    "--predict_file",
                    str(submission),
                    "--path",
                    str(tmp_path),
                    "--name",
                    comp,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "score": None, "eval_stderr": f"<timeout after {timeout}s>"}
        except Exception as e:
            return {"passed": False, "score": None, "eval_stderr": f"<subprocess error: {e}>"}

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            if len(err) > EVAL_STDERR_KEEP_CHARS:
                err = err[:EVAL_STDERR_KEEP_CHARS] + f"\n... <truncated, was {len(err)} chars>"
            return {"passed": False, "score": None, "eval_stderr": err}

        result_file = tmp_path / comp / "result.txt"
        if not result_file.is_file():
            return {
                "passed": False,
                "score": None,
                "eval_stderr": f"<eval ran but {result_file.name} not created>",
            }
        score = _parse_score(result_file.read_text(encoding="utf-8"))

    if score is None:
        return {"passed": False, "score": None, "eval_stderr": "<could not parse score>"}
    if not math.isfinite(score):
        return {"passed": False, "score": score, "eval_stderr": f"<non-finite score: {score}>"}

    return {"passed": True, "score": score}


def _eval_one_workspace(
    *,
    workspace: Path,
    bench_root: Path,
    timeout: int,
) -> dict:
    """Score one modeling workspace's submission.csv. Returns the updated
    output.json record. Skips analysis workspaces (returns the record
    unchanged with a note)."""
    uid = workspace.name
    out_path = workspace / OUTPUT_FILENAME

    if not out_path.is_file():
        logger.warning("[{}] no output.json — skipping", uid)
        return {"uid": uid, "skipped": True, "reason": "no output.json"}

    try:
        record = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("[{}] output.json malformed — skipping", uid)
        return {"uid": uid, "skipped": True, "reason": "malformed output.json"}

    family = record.get("family")
    if family != "modeling":
        # Analysis tasks aren't scored here. Don't touch their record.
        return record

    if not record.get("ok"):
        logger.info("[{}] generation failed (ok=false) — recording passed=false", uid)
        record["passed"] = False
        record["eval_stderr"] = "<generation ok=false; nothing to evaluate>"
        write_output_json(workspace, record)
        return record

    comp = _comp_from_uid(uid) or record.get("task_id")
    if not comp:
        logger.warning("[{}] could not resolve comp name; skipping", uid)
        record["passed"] = False
        record["eval_stderr"] = "<could not resolve competition name from uid>"
        write_output_json(workspace, record)
        return record

    eval_result = _run_one_eval(
        workspace=workspace,
        comp=comp,
        bench_root=bench_root,
        timeout=timeout,
    )
    record.update(eval_result)
    write_output_json(workspace, record)
    logger.info(
        "[{}] passed={} score={}",
        uid,
        eval_result["passed"],
        eval_result.get("score"),
    )
    return record


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _iter_workspaces(output_root: Path) -> Iterable[Path]:
    if not output_root.is_dir():
        return
    for ws in sorted(output_root.iterdir()):
        if ws.is_dir() and ws.name.startswith("dsbench_"):
            yield ws


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score DSBench modeling submissions against upstream's per-comp "
            "eval scripts. Updates each workspace's output.json with `passed` "
            "and `score`. Analysis workspaces are NOT scored here."
        ),
        prog="python -m data_generation.dsbench.eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bench-root",
        required=True,
        help="Path to the unzipped dsbench-data dir (we read "
        "<bench-root>/data_modeling/data/answers/<comp>/test_answer.csv).",
    )
    parser.add_argument(
        "--output-root",
        "-o",
        required=True,
        help="Same --output-root used for data_generation.dsbench.generation.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-task subprocess timeout (default {DEFAULT_TIMEOUT}s).",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-evaluate workspaces even if output.json already has a "
        "'passed' field. Default: skip already-evaluated.",
    )
    parser.add_argument(
        "--uids",
        default=None,
        help="Comma-separated list of uids to evaluate. Overrides full sweep.",
    )
    args = parser.parse_args()

    bench_root = Path(args.bench_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not output_root.is_dir():
        raise SystemExit(f"output-root not a directory: {output_root}")

    workspaces = list(_iter_workspaces(output_root))
    if args.uids:
        wanted = {u.strip() for u in args.uids.split(",") if u.strip()}
        workspaces = [w for w in workspaces if w.name in wanted]
        missing = wanted - {w.name for w in workspaces}
        if missing:
            logger.warning("Requested uids not found in output_root: {}", sorted(missing))

    # Filter to modeling workspaces with ok=true; skip already-evaluated
    # unless --rerun. We still touch analysis workspaces in the loop but
    # they early-return without writing.
    eligible: list[Path] = []
    skipped_already = 0
    skipped_analysis = 0
    skipped_no_ok = 0
    for ws in workspaces:
        out_path = ws / OUTPUT_FILENAME
        if not out_path.is_file():
            continue
        try:
            rec = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if rec.get("family") != "modeling":
            skipped_analysis += 1
            continue
        if not rec.get("ok"):
            skipped_no_ok += 1
            continue
        if not args.rerun and "passed" in rec:
            skipped_already += 1
            continue
        eligible.append(ws)

    logger.info(
        "Evaluating {} modeling workspace(s) (skipped: {} analysis, {} ok=false, {} already-evaluated)",
        len(eligible),
        skipped_analysis,
        skipped_no_ok,
        skipped_already,
    )

    summary: list[dict] = []
    for ws in eligible:
        rec = _eval_one_workspace(
            workspace=ws,
            bench_root=bench_root,
            timeout=args.timeout,
        )
        summary.append(rec)

    n_eval = sum(1 for r in summary if "passed" in r)
    n_pass = sum(1 for r in summary if r.get("passed"))
    logger.info(
        "Done. {}/{} modeling workspaces passed.",
        n_pass,
        n_eval,
    )


if __name__ == "__main__":
    _main()
