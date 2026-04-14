#!/usr/bin/env python3
"""
citation_coverage.py — Enforce the paper's ≥90% citation integration rule
(App. D.3).

Greps a generated .tex file for all citation commands, counts the unique
keys actually cited, and compares against the verified citation pool.
Exits non-zero if coverage < 90%.

Usage:
    python citation_coverage.py --tex intro_relwork.tex --pool citation_pool.json
    python citation_coverage.py --tex intro_relwork.tex --pool citation_pool.json --threshold 0.85
"""
import argparse
import json
import re
import sys

CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citeauthor|citeyear|autocite|parencite|textcite)"
    r"(?:\[[^\]]*\])?"
    r"\{([^}]+)\}"
)


def extract_cited_keys(tex: str) -> set[str]:
    keys = set()
    for m in CITE_RE.finditer(tex):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                keys.add(k)
    return keys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tex", required=True, help="LaTeX file to inspect")
    p.add_argument("--pool", required=True, help="citation_pool.json")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Minimum integration ratio (default 0.90 per paper)",
    )
    args = p.parse_args()

    with open(args.tex) as f:
        tex = f.read()
    with open(args.pool) as f:
        pool = json.load(f)

    pool_papers = pool.get("papers", [])
    pool_keys = {p.get("bibtex_key") for p in pool_papers if p.get("bibtex_key")}
    if not pool_keys:
        print("ERROR: pool has no bibtex_keys. Run bibtex_format.py first.", file=sys.stderr)
        return 1

    cited = extract_cited_keys(tex)
    cited_in_pool = cited & pool_keys
    n_pool = len(pool_keys)
    n_cited = len(cited_in_pool)
    ratio = n_cited / n_pool if n_pool else 0.0
    threshold_n = int(args.threshold * n_pool)

    print(
        f"Coverage: {n_cited}/{n_pool} = {ratio*100:.1f}% "
        f"(threshold {args.threshold*100:.0f}% = {threshold_n})"
    )

    # report keys cited but NOT in pool — those are forbidden by the prompt
    foreign = cited - pool_keys
    if foreign:
        print(
            f"\nWARNING: {len(foreign)} cited keys NOT in citation pool "
            f"(violates 'cite ONLY collected_papers' rule):"
        )
        for k in sorted(foreign):
            print(f"  - {k}")

    if n_cited < threshold_n:
        uncited = pool_keys - cited
        print(f"\nFAIL: missing {len(uncited)} pool papers from .tex:")
        # show with title for actionable re-prompting
        title_by_key = {
            p.get("bibtex_key"): p.get("title", "") for p in pool_papers if p.get("bibtex_key")
        }
        discovered_by_key = {
            p.get("bibtex_key"): p.get("discovered_for", [])
            for p in pool_papers
            if p.get("bibtex_key")
        }
        for k in sorted(uncited):
            tag = ",".join(discovered_by_key.get(k, [])) or "?"
            t = title_by_key.get(k, "")
            print(f"  - {k:40s}  [{tag}]  {t[:60]}")
        return 1

    print("OK: citation coverage meets threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
