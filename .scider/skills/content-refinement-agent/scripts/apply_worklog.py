#!/usr/bin/env python3
"""
apply_worklog.py — Append a timestamped iteration entry to worklog.json.

The worklog is the canonical history of the refinement loop: every
iteration's review, score, decision, and actions taken. The orchestrator
reads it at the end to identify the best snapshot to promote.

Usage:
    python apply_worklog.py \\
        --worklog workspace/refinement/worklog.json \\
        --iter 2 \\
        --review iter2/review.json \\
        --score iter2/score.json \\
        --decision ACCEPT_IMPROVED \\
        --actions iter2/worklog_entry.json   # the agent's emitted worklog block

The script creates worklog.json if it doesn't exist.
"""
import argparse
import datetime as dt
import json
import os
import sys


def load_json(path: str | None) -> dict | list | None:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worklog", required=True, help="path to worklog.json")
    p.add_argument("--iter", type=int, required=True, help="iteration number (0-indexed)")
    p.add_argument("--review", help="path to review.json for this iteration")
    p.add_argument("--score", help="path to score.json for this iteration")
    p.add_argument(
        "--decision",
        required=True,
        help="ACCEPT_IMPROVED / ACCEPT_TIED_NON_NEGATIVE / "
        "REVERT_OVERALL_DECREASED / REVERT_TIED_NEGATIVE_SUBAXIS",
    )
    p.add_argument(
        "--actions",
        help="path to the agent's worklog block JSON "
        "(addressed_weaknesses, integrated_answers, actions_taken)",
    )
    p.add_argument("--halted-because", help="reason if this iteration triggers a halt")
    args = p.parse_args()

    if os.path.exists(args.worklog):
        with open(args.worklog) as f:
            wl = json.load(f)
    else:
        wl = {"iterations": [], "halted_because": None, "best_iter": None}

    entry = {
        "iter": args.iter,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": args.decision,
        "review": load_json(args.review),
        "score": load_json(args.score),
        "actions": load_json(args.actions),
    }
    wl["iterations"].append(entry)

    if args.halted_because:
        wl["halted_because"] = args.halted_because

    # Re-compute best_iter: highest accepted overall_score
    accepted = [
        it
        for it in wl["iterations"]
        if it.get("decision", "").startswith("ACCEPT") and it.get("score")
    ]
    if accepted:
        best = max(accepted, key=lambda it: it["score"].get("overall_score", 0))
        wl["best_iter"] = best["iter"]

    os.makedirs(os.path.dirname(os.path.abspath(args.worklog)) or ".", exist_ok=True)
    with open(args.worklog, "w") as f:
        json.dump(wl, f, indent=2, ensure_ascii=False)

    print(f"OK: appended iter {args.iter} ({args.decision}) to {args.worklog}")
    if wl["best_iter"] is not None:
        print(f"    current best_iter: {wl['best_iter']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
