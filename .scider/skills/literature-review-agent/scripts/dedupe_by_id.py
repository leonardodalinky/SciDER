#!/usr/bin/env python3
"""
dedupe_by_id.py — Deduplicate a verified citation pool using Semantic Scholar
unique paperId, with DOI / ArXiv / normalized-title fallbacks.

Implements the paper's Rule 4 (App. D.3): "gathered citations are
deduplicated using unique paper ID keys".

Also computes `min_cite_paper_count = floor(0.9 * len(papers))` for the
≥90% citation integration rule.

Usage:
    python dedupe_by_id.py --in raw_pool.json --out citation_pool.json [--cutoff 2024-10-01]
"""
import argparse
import json
import math
import re
import sys


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def make_key(paper: dict) -> str:
    if paper.get("paperId"):
        return f"s2:{paper['paperId']}"
    ext = paper.get("externalIds") or {}
    if ext.get("DOI"):
        return f"doi:{ext['DOI'].lower()}"
    if ext.get("ArXiv"):
        # strip version suffix if any
        a = ext["ArXiv"].split("v")[0] if "v" in ext["ArXiv"][-3:] else ext["ArXiv"]
        return f"arxiv:{a.lower()}"
    title = paper.get("title", "")
    return f"title:{norm_title(title)}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", required=True, help="Raw verified pool JSON")
    p.add_argument("--out", required=True, help="Deduped citation_pool.json")
    p.add_argument("--cutoff", help="Cutoff date YYYY-MM-DD (recorded in output)")
    args = p.parse_args()

    with open(args.inp) as f:
        raw = json.load(f)

    candidates = raw.get("papers") or raw.get("candidates") or []
    if not candidates:
        print("ERROR: input has neither 'papers' nor 'candidates' key", file=sys.stderr)
        return 1

    by_key: dict[str, dict] = {}
    collisions: list[tuple[str, str]] = []
    for c in candidates:
        key = make_key(c)
        if key in by_key:
            existing = by_key[key]
            score_new = c.get("match_score", 0)
            score_old = existing.get("match_score", 0)
            if score_new > score_old:
                # merge discovered_for
                merged = existing.get("discovered_for", []) + c.get("discovered_for", [])
                c["discovered_for"] = list(dict.fromkeys(merged))  # preserve order, dedupe
                by_key[key] = c
            else:
                merged = existing.get("discovered_for", []) + c.get("discovered_for", [])
                existing["discovered_for"] = list(dict.fromkeys(merged))
            collisions.append((key, c.get("title", "")))
        else:
            by_key[key] = c

    deduped = list(by_key.values())
    n = len(deduped)
    min_cite = math.floor(0.9 * n)

    out = {
        "papers": deduped,
        "min_cite_paper_count": min_cite,
        "n_total": n,
        "n_collisions_merged": len(collisions),
    }
    if args.cutoff:
        out["cutoff_date"] = args.cutoff

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"OK: {len(candidates)} candidates → {n} unique papers")
    print(f"    {len(collisions)} duplicates merged")
    print(f"    min_cite_paper_count (≥90%): {min_cite}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
