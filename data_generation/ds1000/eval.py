"""DS-1000 evaluator — score the trajectories produced by
``data_generation.ds1000.generation``.

DS-1000's HF rows ship a ``code_context`` field that defines a
``test_execution(solution_code: str)`` helper (and sometimes a
``test_string(solution_code: str)`` for matplotlib-style problems). We
build a tiny eval script that imports the context and calls
``test_execution(<agent's code.py>)`` (falling back to ``test_string``),
run it in a subprocess with a hard timeout, and treat ``returncode == 0``
as pass.

The result is written back into each workspace's ``output.json`` as
``passed: bool`` (and a truncated ``eval_stderr`` on failure for debugging),
so downstream filtering by correctness is just::

    jq 'select(.passed == true) | .history_path' \
        data_generation/datasets/ds1000/*/output.json

Usage
-----
    python -m data_generation.ds1000.eval \\
        --output-root data_generation/datasets/ds1000

    # Re-evaluate everything (e.g. after fixing a sandbox bug)
    python -m data_generation.ds1000.eval -o ... --rerun

    # Score a subset only
    python -m data_generation.ds1000.eval -o ... --uids ds1000_Pandas_0,ds1000_NumPy_5
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from loguru import logger

# Make project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ._common import DEFAULT_CODE_FILENAME, OUTPUT_FILENAME, write_output_json
from .generation import DEFAULT_SPLIT, HF_DATASET_ID, _build_uid, _load_ds1000

EVAL_SCRIPT_NAME = "eval_run.py"
EVAL_STDERR_KEEP_CHARS = 2000


def _index_dataset_by_uid(split: str) -> dict[str, dict]:
    """Build a uid → row map for fast lookup. Loads the whole dataset
    once (cheap; DS-1000 is ~1k rows)."""
    rows = _load_ds1000(split)
    by_uid: dict[str, dict] = {}
    for r in rows:
        uid = _build_uid(r["library"], r["problem_id"])
        by_uid[uid] = r
    return by_uid


def _build_eval_script(code_context: str, solution_code: str) -> str | None:
    """Compose the eval script: ``code_context`` (defines test_execution /
    test_string) + a tail that calls whichever exists with the agent's
    solution. Returns ``None`` if ``code_context`` defines neither — which
    would mean an unexpected DS-1000 row shape (eval-system error, not a
    model failure).
    """
    has_exec = "def test_execution(" in code_context
    has_str = "def test_string(" in code_context
    if not (has_exec or has_str):
        return None

    # Prefer test_execution (numerical / dataframe / tensor outputs); fall
    # back to test_string (matplotlib code-string checks). Some rows define
    # both — upstream's behaviour is to require BOTH to pass when both
    # exist, so we mirror that.
    calls = []
    if has_exec:
        calls.append("    test_execution(_solution)")
    if has_str:
        calls.append("    test_string(_solution)")

    tail = (
        "\n\n# ---- harness injected by data_generation.ds1000.eval ----\n"
        "import sys, traceback\n"
        "_solution = " + repr(solution_code) + "\n"
        "try:\n" + "\n".join(calls) + "\n"
        "    print('DS1000_EVAL_PASS')\n"
        "except Exception:\n"
        "    traceback.print_exc()\n"
        "    sys.exit(1)\n"
    )
    return code_context + tail


def _run_eval(workspace: Path, eval_script: str, timeout: int) -> tuple[bool, str]:
    """Run the substituted test script in a subprocess. Returns
    ``(passed, stderr_tail)``. ``stderr_tail`` is truncated to
    ``EVAL_STDERR_KEEP_CHARS`` for log noise control."""
    eval_path = workspace / EVAL_SCRIPT_NAME
    eval_path.write_text(eval_script, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, EVAL_SCRIPT_NAME],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"<timeout after {timeout}s>"
    except Exception as e:
        return False, f"<subprocess error: {e}>"

    passed = proc.returncode == 0
    if passed:
        # Don't keep eval_run.py + a successful run wastes disk; drop it.
        try:
            eval_path.unlink()
        except OSError:
            pass
        return True, ""

    # On failure keep eval_run.py for debugging; trim noisy stderr.
    err = (proc.stderr or "").strip()
    if not err:
        err = (proc.stdout or "").strip()
    if len(err) > EVAL_STDERR_KEEP_CHARS:
        err = err[:EVAL_STDERR_KEEP_CHARS] + f"\n... <truncated, was {len(err)} chars>"
    return False, err


def _eval_one(
    *,
    workspace: Path,
    row: dict,
    timeout: int,
    code_filename: str,
) -> dict:
    """Evaluate a single workspace; return the updated output.json record."""
    uid = workspace.name
    out_path = workspace / OUTPUT_FILENAME
    code_path = workspace / code_filename

    if not out_path.is_file():
        logger.warning("[{}] no output.json — skipping (was the workspace ever run?)", uid)
        return {"uid": uid, "skipped": True, "reason": "no output.json"}

    try:
        record = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("[{}] output.json malformed — skipping", uid)
        return {"uid": uid, "skipped": True, "reason": "malformed output.json"}

    if not record.get("ok"):
        logger.info("[{}] generation failed (ok=false) — recording passed=false", uid)
        record["passed"] = False
        record["eval_stderr"] = "<no code.py to evaluate>"
        write_output_json(workspace, record)
        return record

    if not code_path.is_file():
        logger.warning("[{}] code.py missing despite ok=true — marking failed", uid)
        record["passed"] = False
        record["eval_stderr"] = "<code.py missing>"
        write_output_json(workspace, record)
        return record

    code_text = code_path.read_text(encoding="utf-8")
    code_context = row.get("code_context") or ""
    eval_script = _build_eval_script(code_context, code_text)
    if eval_script is None:
        logger.warning("[{}] code_context defines no test_execution / test_string", uid)
        record["passed"] = False
        record["eval_stderr"] = "<eval-system error: no test_execution/test_string in code_context>"
        write_output_json(workspace, record)
        return record

    passed, stderr_tail = _run_eval(workspace, eval_script, timeout)
    record["passed"] = passed
    if passed:
        record.pop("eval_stderr", None)
    else:
        record["eval_stderr"] = stderr_tail
    write_output_json(workspace, record)
    logger.info("[{}] passed={} stderr_chars={}", uid, passed, len(stderr_tail))
    return record


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _iter_workspaces(output_root: Path) -> Iterable[Path]:
    if not output_root.is_dir():
        return
    for ws in sorted(output_root.iterdir()):
        if ws.is_dir() and ws.name.startswith("ds1000_"):
            yield ws


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DS-1000 trajectories generated by "
            "data_generation.ds1000.generation. Updates output.json in each "
            "workspace with passed=true|false (and eval_stderr on failure)."
        ),
        prog="python -m data_generation.ds1000.eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        "-o",
        required=True,
        help="Same --output-root used for data_generation.ds1000.generation.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"DS-1000 split (default: {DEFAULT_SPLIT}). Must match the split "
        "used for generation, otherwise uid → row lookup will miss.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Per-task subprocess timeout in seconds (default 15).",
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
    parser.add_argument(
        "--code-filename",
        default=DEFAULT_CODE_FILENAME,
        help=f"Solution filename to read from each workspace (default {DEFAULT_CODE_FILENAME}).",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    if not output_root.is_dir():
        raise SystemExit(f"output-root not a directory: {output_root}")

    logger.info("Indexing {} (split={})", HF_DATASET_ID, args.split)
    by_uid = _index_dataset_by_uid(args.split)
    logger.info("Indexed {} DS-1000 rows", len(by_uid))

    # Filter workspaces.
    workspaces = list(_iter_workspaces(output_root))
    if args.uids:
        wanted = {u.strip() for u in args.uids.split(",") if u.strip()}
        workspaces = [w for w in workspaces if w.name in wanted]
        missing = wanted - {w.name for w in workspaces}
        if missing:
            logger.warning("Requested uids not found in output_root: {}", sorted(missing))

    if not args.rerun:
        before = len(workspaces)
        skipped: list[str] = []
        kept: list[Path] = []
        for ws in workspaces:
            try:
                rec = json.loads((ws / OUTPUT_FILENAME).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                rec = {}
            if "passed" in rec:
                skipped.append(ws.name)
            else:
                kept.append(ws)
        workspaces = kept
        if skipped:
            logger.info(
                "Skipping {} workspace(s) already evaluated; use --rerun to redo", len(skipped)
            )

    logger.info("Evaluating {} workspace(s)", len(workspaces))
    summary: list[dict] = []
    for ws in workspaces:
        row = by_uid.get(ws.name)
        if row is None:
            logger.warning("[{}] no DS-1000 row matches this uid — skipping", ws.name)
            summary.append({"uid": ws.name, "skipped": True, "reason": "uid not in dataset"})
            continue
        rec = _eval_one(
            workspace=ws,
            row=row,
            timeout=args.timeout,
            code_filename=args.code_filename,
        )
        summary.append(rec)

    # Final tally.
    n_eval = sum(1 for r in summary if "passed" in r)
    n_pass = sum(1 for r in summary if r.get("passed"))
    n_skip = sum(1 for r in summary if r.get("skipped"))
    logger.info(
        "Done. {}/{} passed, {} skipped (results in each workspace's output.json)",
        n_pass,
        n_eval,
        n_skip,
    )


if __name__ == "__main__":
    _main()
