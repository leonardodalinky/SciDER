"""
AIRS-Bench normalized-score evaluator.

Reads a ``results.json`` produced by ``airsbench_workflow.py`` and computes
each task's *normalized score* (NS) per the AIRS-Bench paper, plus the
average across tasks present in the results file.

Normalized score (definition from the paper)::

    phi(s)  = -log10(|s - s_opt|)
    NS(s)   = (phi(s) - phi(s_min)) / (phi(s_sota) - phi(s_min))

where for each task ``t``:
    s        : our score on task t (from results.json)
    s_opt    : metadata.logging_info.optimal_score
    s_sota   : best metadata.logging_info.sota[*].sota_score (direction-aware)
    s_min    : metadata.logging_info.estimated_worst_score

Lower-is-better metrics (``metric_lower_is_better: true``, e.g. MAE) use
min(...) for ``s_sota``; higher-is-better use max(...). The ``phi`` operator
itself is direction-agnostic — it measures distance to the optimum, so the
direction only matters when picking the best SOTA from a list.

Usage::

    uv run python benchmarks/airsbench/airsbench_eval.py \\
        --results /sciclone/proj-ds/ai4scientist/kelin/SciDER/airs-bench/workspace/results.json
    # default --tasks-root: <this_script_dir>/airs-bench/airsbench/tasks/rad
"""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml
from loguru import logger

_HERE = Path(__file__).parent.resolve()
DEFAULT_TASKS_ROOT = _HERE / "airs-bench" / "airsbench" / "tasks" / "rad"
# Floor for |s - s_opt| so phi doesn't blow up to infinity when our score
# matches the optimum exactly. 1e-12 is small enough not to perturb honest
# numbers, large enough to keep the log finite.
_PHI_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Core math                                                                   #
# --------------------------------------------------------------------------- #


def phi(s: float, s_opt: float) -> float:
    """phi(s) = -log10(max(|s - s_opt|, eps)). Higher = closer to optimum."""
    diff = max(abs(float(s) - float(s_opt)), _PHI_EPS)
    return -math.log10(diff)


def best_sota(sota_entries: list[dict], lower_is_better: bool) -> float | None:
    vals = [
        float(e["sota_score"])
        for e in sota_entries or []
        if isinstance(e.get("sota_score"), (int, float))
    ]
    if not vals:
        return None
    return min(vals) if lower_is_better else max(vals)


def normalized_score(
    *,
    our: float,
    s_opt: float,
    s_min: float,
    s_sota: float,
) -> tuple[float, dict[str, float]]:
    """Return (NS, components) where ``components`` exposes phi(min/sota/ours)."""
    phi_ours = phi(our, s_opt)
    phi_min = phi(s_min, s_opt)
    phi_sota = phi(s_sota, s_opt)
    denom = phi_sota - phi_min
    if denom == 0:
        ns = float("nan")
    else:
        ns = (phi_ours - phi_min) / denom
    return ns, {"phi_ours": phi_ours, "phi_min": phi_min, "phi_sota": phi_sota}


# --------------------------------------------------------------------------- #
# IO + per-task evaluation                                                    #
# --------------------------------------------------------------------------- #


def _read_metadata(task_dir: Path) -> dict:
    with (task_dir / "metadata.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _our_score_from_record(record: dict) -> float | None:
    """Pull our score out of a result record. Prefers explicit ``our_score``,
    falls back to the first numeric value in ``score``."""
    if isinstance(record.get("our_score"), (int, float)):
        return float(record["our_score"])
    score = record.get("score") or {}
    if isinstance(score, dict):
        for v in score.values():
            if isinstance(v, (int, float)):
                return float(v)
    return None


def evaluate_one(record: dict, task_dir: Path) -> dict:
    """Compute NS + components for a single result record."""
    out: dict = {"task": record.get("task")}
    if not record.get("ok"):
        out.update(skipped=True, reason=f"record ok={record.get('ok')}: {record.get('error')}")
        return out

    our = _our_score_from_record(record)
    if our is None:
        out.update(skipped=True, reason="no numeric score in record")
        return out

    metadata = _read_metadata(task_dir)
    info = metadata.get("logging_info") or {}
    s_opt = info.get("optimal_score")
    s_min = info.get("estimated_worst_score")
    lower = bool(metadata.get("metric_lower_is_better"))
    s_sota = best_sota(info.get("sota") or [], lower)

    missing = [
        n
        for n, v in (("optimal_score", s_opt), ("estimated_worst_score", s_min), ("sota", s_sota))
        if v is None
    ]
    if missing:
        out.update(skipped=True, reason=f"missing metadata fields: {missing}")
        return out

    ns, comps = normalized_score(our=our, s_opt=s_opt, s_min=s_min, s_sota=s_sota)
    out.update(
        skipped=False,
        our_score=our,
        s_opt=float(s_opt),
        s_min=float(s_min),
        s_sota=float(s_sota),
        lower_is_better=lower,
        normalized_score=ns,
        **comps,
    )
    return out


def evaluate_all(results: list[dict], tasks_root: Path) -> dict:
    rows: list[dict] = []
    for r in results:
        name = r.get("task")
        if not name:
            continue
        td = tasks_root / name
        if not (td / "metadata.yaml").is_file():
            rows.append({"task": name, "skipped": True, "reason": f"task dir not found: {td}"})
            continue
        rows.append(evaluate_one(r, td))

    scored = [
        r
        for r in rows
        if not r.get("skipped") and not math.isnan(r.get("normalized_score", float("nan")))
    ]
    avg = (sum(r["normalized_score"] for r in scored) / len(scored)) if scored else float("nan")

    return {
        "n_total": len(rows),
        "n_scored": len(scored),
        "n_skipped": len(rows) - len(scored),
        "average_normalized_score": avg,
        "tasks": rows,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _print_table(report: dict) -> None:
    print()
    print(f"  {'task':55s} {'ours':>10s}   {'sota':>10s}   {'min':>10s}   {'opt':>6s}   {'NS':>7s}")
    print(f"  {'-'*55} {'-'*10}   {'-'*10}   {'-'*10}   {'-'*6}   {'-'*7}")
    for r in sorted(report["tasks"], key=lambda x: x.get("task", "")):
        if r.get("skipped"):
            print(f"  {r['task']:55s}   skipped: {r.get('reason','?')}")
            continue

        def f(x: float, w: int = 10, prec: int = 4) -> str:
            return f"{x:>{w}.{prec}f}"

        print(
            f"  {r['task']:55s} "
            f"{f(r['our_score'])}   {f(r['s_sota'])}   {f(r['s_min'])}   "
            f"{f(r['s_opt'], w=6, prec=2)}   {f(r['normalized_score'], w=7, prec=4)}"
        )
    print()
    print(
        f"  scored {report['n_scored']}/{report['n_total']} tasks "
        f"({report['n_skipped']} skipped) — "
        f"average normalized score: {report['average_normalized_score']:.4f}"
    )
    print()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute AIRS-Bench normalized scores from a results.json.",
        prog="python benchmarks/airsbench/airsbench_eval.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to results.json produced by airsbench_workflow.",
    )
    parser.add_argument(
        "--tasks-root",
        default=str(DEFAULT_TASKS_ROOT),
        help=f"Path to airs-bench rad/ tasks dir (default: {DEFAULT_TASKS_ROOT}).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional: write the full per-task report as JSON to this path.",
    )

    args = parser.parse_args()

    results_path = Path(args.results).expanduser()
    if not results_path.is_file():
        logger.error("results file not found: {}", results_path)
        sys.exit(1)
    tasks_root = Path(args.tasks_root).expanduser().resolve()
    if not tasks_root.is_dir():
        logger.error("tasks_root is not a directory: {}", tasks_root)
        sys.exit(1)

    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(results, list):
        logger.error("results.json must be a list of records, got {}", type(results).__name__)
        sys.exit(1)

    report = evaluate_all(results, tasks_root)
    _print_table(report)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  wrote full report → {args.out}")


if __name__ == "__main__":
    _main()
