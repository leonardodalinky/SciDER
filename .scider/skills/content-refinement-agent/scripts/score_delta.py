#!/usr/bin/env python3
"""
score_delta.py — Apply the PaperOrchestra refinement halt rules from two
score JSONs.

Encodes the halt rules from arXiv:2604.05018 §4 Step 5:

  - ACCEPT if curr.overall > prev.overall
  - ACCEPT if curr.overall == prev.overall AND net sub-axis delta >= 0
  - REVERT (overall_decreased) if curr.overall < prev.overall
  - REVERT (tied_negative_subaxis) if curr.overall == prev.overall AND
            net sub-axis delta < 0

Additionally encodes the plateau early-stop rule (not in the original paper
but added to match its cost budget of ~5-7 LLM calls):

  - HALT_PLATEAU if the improvement is accepted but overall_delta is below
    --plateau-threshold for --plateau-streak or more consecutive iterations.
    Exit code 4.  The loop should stop — further iterations are unlikely to
    yield meaningful gains.

Exit codes:
    0  ACCEPT (improved or tied non-negative, and no plateau)
    1  REVERT (overall decreased)
    2  REVERT (tied with negative sub-axis delta)
    3  argument or input error
    4  HALT_PLATEAU (accepted but diminishing returns detected)

Score JSON shape (see references/reviewer-rubric.md):
    {
      "axis_scores": {
        "scientific_depth":     {"score": 65, ...},
        "technical_execution":  {"score": 70, ...},
        "logical_flow":         {"score": 60, ...},
        "writing_clarity":      {"score": 55, ...},
        "evidence_presentation":{"score": 72, ...},
        "academic_style":       {"score": 68, ...}
      },
      "overall_score": 64.5,
      ...
    }

Usage:
    python score_delta.py --prev iter0/score.json --curr iter1/score.json
    python score_delta.py --prev iter2/score.json --curr iter3/score.json \\
        --plateau-threshold 1.0 --plateau-streak 2 --consecutive-small 2
"""
import argparse
import json
import sys

AXES = [
    "scientific_depth",
    "technical_execution",
    "logical_flow",
    "writing_clarity",
    "evidence_presentation",
    "academic_style",
]

DEFAULT_PLATEAU_THRESHOLD = 1.0  # points
DEFAULT_PLATEAU_STREAK = 3  # consecutive iterations below threshold → halt


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prev", required=True, help="Score JSON from previous accepted iteration")
    p.add_argument("--curr", required=True, help="Score JSON from just-completed iteration")
    p.add_argument(
        "--plateau-threshold",
        type=float,
        default=DEFAULT_PLATEAU_THRESHOLD,
        metavar="POINTS",
        help=f"Minimum overall_delta to not count as a 'small' improvement "
        f"(default: {DEFAULT_PLATEAU_THRESHOLD})",
    )
    p.add_argument(
        "--plateau-streak",
        type=int,
        default=DEFAULT_PLATEAU_STREAK,
        metavar="N",
        help=f"Number of consecutive small improvements before HALT_PLATEAU "
        f"(default: {DEFAULT_PLATEAU_STREAK})",
    )
    p.add_argument(
        "--consecutive-small",
        type=int,
        default=0,
        metavar="N",
        help="Number of consecutive small-delta accepted iterations so far "
        "(maintained by the calling loop; default: 0)",
    )
    args = p.parse_args()

    try:
        prev = load(args.prev)
        curr = load(args.curr)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: failed to load score JSONs: {e}", file=sys.stderr)
        return 3

    p_overall = float(prev.get("overall_score", 0))
    c_overall = float(curr.get("overall_score", 0))
    overall_delta = c_overall - p_overall

    p_axes = prev.get("axis_scores") or {}
    c_axes = curr.get("axis_scores") or {}
    deltas: dict[str, float] = {}
    for ax in AXES:
        ps = float((p_axes.get(ax) or {}).get("score", 0))
        cs = float((c_axes.get(ax) or {}).get("score", 0))
        deltas[ax] = cs - ps
    net_subaxis = sum(deltas.values())

    # --- Primary accept/revert decision ---
    if c_overall > p_overall:
        decision = "ACCEPT_IMPROVED"
        exit_code = 0
    elif c_overall == p_overall:
        if net_subaxis >= 0:
            decision = "ACCEPT_TIED_NON_NEGATIVE"
            exit_code = 0
        else:
            decision = "REVERT_TIED_NEGATIVE_SUBAXIS"
            exit_code = 2
    else:
        decision = "REVERT_OVERALL_DECREASED"
        exit_code = 1

    # --- Plateau early-stop (only applies to accepted iterations) ---
    is_small_delta = overall_delta < args.plateau_threshold
    new_consecutive_small = (args.consecutive_small + 1) if is_small_delta else 0
    plateau_triggered = False

    if exit_code == 0 and new_consecutive_small >= args.plateau_streak:
        decision = "HALT_PLATEAU"
        exit_code = 4
        plateau_triggered = True

    out = {
        "decision": decision,
        "exit_code": exit_code,
        "overall_prev": p_overall,
        "overall_curr": c_overall,
        "overall_delta": overall_delta,
        "subaxis_deltas": deltas,
        "net_subaxis": net_subaxis,
        "is_small_delta": is_small_delta,
        "consecutive_small": new_consecutive_small,
        "plateau_threshold": args.plateau_threshold,
        "plateau_streak": args.plateau_streak,
        "plateau_triggered": plateau_triggered,
    }
    print(json.dumps(out, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
