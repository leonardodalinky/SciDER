#!/usr/bin/env python3
"""
validate_pool.py — Validate and auto-fix citation_pool.json before it is
passed to bibtex_format.py or the Section Writing Agent.

Catches the two most common schema errors produced by the Literature Review
Agent and fixes them in place with --fix.

Error 1 — Authors as plain strings (WRONG format for bibtex_format.py):
    WRONG:   "authors": ["Alice Smith", "Bob Jones"]
    CORRECT: "authors": [{"name": "Alice Smith"}, {"name": "Bob Jones"}]

Error 2 — Missing required fields (title, year). These cause bibtex_format.py
    to emit incomplete entries. Reported as errors, not auto-fixed.

Also checks that the pool has the top-level keys that downstream scripts
expect: "papers", "min_cite_paper_count".

Exit codes:
    0  Pool is valid (or was fully fixed with --fix)
    1  Unrecoverable errors remain (missing required fields, no papers)

Usage:
    python validate_pool.py --pool workspace/citation_pool.json
    python validate_pool.py --pool workspace/citation_pool.json --fix
"""
import argparse
import json
import sys

REQUIRED_PAPER_FIELDS = ["title", "year"]
RECOMMENDED_PAPER_FIELDS = ["paperId", "abstract", "venue", "authors"]
REQUIRED_TOP_FIELDS = ["papers", "min_cite_paper_count"]


def validate_and_fix(pool: dict, fix: bool) -> tuple[list[str], list[str], int]:
    """
    Returns (errors, warnings, n_fixed).
    If fix=True, mutates pool in place where possible.
    """
    errors: list[str] = []
    warnings: list[str] = []
    n_fixed = 0

    # Top-level structure
    for field in REQUIRED_TOP_FIELDS:
        if field not in pool:
            warnings.append(f"top-level field '{field}' missing — was dedupe_by_id.py run?")

    papers = pool.get("papers", [])
    if not papers:
        errors.append("pool['papers'] is empty or missing")
        return errors, warnings, n_fixed

    for i, paper in enumerate(papers):
        label = paper.get("title") or f"paper #{i}"

        # --- Authors format check ---
        authors = paper.get("authors")
        if authors is not None:
            if not isinstance(authors, list):
                errors.append(f"[{label}] 'authors' must be a list, got {type(authors).__name__}")
            elif authors:
                if isinstance(authors[0], str):
                    if fix:
                        paper["authors"] = [{"name": a} for a in authors]
                        n_fixed += 1
                    else:
                        errors.append(
                            f"[{label}] authors are plain strings "
                            f'(e.g. "{authors[0]}") — run with --fix to auto-convert'
                        )
                elif not isinstance(authors[0], dict):
                    errors.append(
                        f"[{label}] authors[0] is {type(authors[0]).__name__}, "
                        f"expected dict with 'name' key"
                    )

        # --- Required fields ---
        for field in REQUIRED_PAPER_FIELDS:
            if not paper.get(field):
                errors.append(f"[{label}] missing required field '{field}'")

        # --- Recommended fields ---
        for field in RECOMMENDED_PAPER_FIELDS:
            if not paper.get(field):
                warnings.append(f"[{label}] missing recommended field '{field}'")

    return errors, warnings, n_fixed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True, help="citation_pool.json path")
    p.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix recoverable errors (authors format) and write back",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress warnings, only show errors")
    args = p.parse_args()

    with open(args.pool) as f:
        pool = json.load(f)

    errors, warnings, n_fixed = validate_and_fix(pool, fix=args.fix)

    if not args.quiet:
        for w in warnings:
            print(f"WARN: {w}")

    had_errors = bool(errors)
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)

    if had_errors and not args.fix:
        print(
            "\nTip: re-run with --fix to auto-correct recoverable issues (authors format).",
            file=sys.stderr,
        )
        return 1

    if n_fixed > 0:
        with open(args.pool, "w") as f:
            json.dump(pool, f, indent=2, ensure_ascii=False)
        print(f"OK: {n_fixed} paper(s) auto-fixed and written back to {args.pool}")

    n = len(pool.get("papers", []))
    if not had_errors and n_fixed == 0:
        print(f"OK: {n} papers validated — no errors")
    elif n_fixed > 0 and not errors:
        print(f"OK: {n} papers validated after auto-fix")

    return (
        0
        if (
            not errors
            or (args.fix and n_fixed > 0 and not [e for e in errors if "missing required" in e])
        )
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
