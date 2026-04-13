#!/usr/bin/env python3
"""
pre_dedup_candidates.py — Deduplicate Phase 1 raw candidates by normalized
title before Phase 2 Semantic Scholar verification.

Multiple search queries in Phase 1 often return the same papers. Verifying
duplicates wastes S2 quota (1 QPS hard cap) and adds 30-40% unnecessary
wall-time. This script removes obvious duplicates — same paper found via
multiple queries — before the sequential verification loop begins.

Dedup strategy (in order of preference):
1. Exact arXiv ID match extracted from source URL or snippet.
2. Levenshtein ratio >= 92 on normalized titles (high threshold to avoid
   false collisions between similarly-named papers).

When two candidates are considered the same, we keep the one that appeared
earlier in the list and merge their `discovered_for` attribution tags so
the surviving entry is credited to all originating queries.

Usage:
    python pre_dedup_candidates.py \\
        --in workspace/raw_candidates.json \\
        --out workspace/deduped_candidates.json

Input JSON shape:
    {"candidates": [{"title": "...", "url": "...", "snippet": "...",
                     "discovered_for": ["intro.1"]}, ...]}
    OR a bare list.
"""
import argparse
import json
import re
import sys

ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)


def norm_title(t: str) -> str:
    t = re.sub(r"[^a-z0-9 ]", " ", t.lower())
    return " ".join(t.split())


def levenshtein_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    dist = prev[lb]
    return (1.0 - dist / max(la, lb)) * 100.0


def extract_arxiv_id(candidate: dict) -> str | None:
    for text in (candidate.get("url", ""), candidate.get("snippet", "")):
        m = ARXIV_RE.search(text)
        if m:
            return m.group(1)
    return None


def make_exact_key(candidate: dict) -> str:
    """Canonical key: arXiv ID if extractable, else normalized title."""
    aid = extract_arxiv_id(candidate)
    if aid:
        return f"arxiv:{aid}"
    return f"title:{norm_title(candidate.get('title', ''))}"


def merge_discovered_for(a: dict, b: dict) -> list:
    df_a = a.get("discovered_for") or []
    df_b = b.get("discovered_for") or []
    return list(dict.fromkeys(df_a + df_b))


def dedup(candidates: list[dict], title_ratio_threshold: float = 92.0) -> list[dict]:
    # Pass 1: exact key dedup (arXiv ID or identical normalized title)
    by_key: dict[str, dict] = {}
    for c in candidates:
        key = make_exact_key(c)
        if key in by_key:
            by_key[key]["discovered_for"] = merge_discovered_for(by_key[key], c)
        else:
            by_key[key] = dict(c)

    deduped = list(by_key.values())

    # Pass 2: fuzzy title dedup — O(n²) but n is ~50-100 candidates max
    normed = [norm_title(c.get("title", "")) for c in deduped]
    drop: set[int] = set()
    for i in range(len(deduped)):
        if i in drop:
            continue
        for j in range(i + 1, len(deduped)):
            if j in drop:
                continue
            if levenshtein_ratio(normed[i], normed[j]) >= title_ratio_threshold:
                deduped[i]["discovered_for"] = merge_discovered_for(deduped[i], deduped[j])
                drop.add(j)

    return [c for idx, c in enumerate(deduped) if idx not in drop]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", required=True, help="Raw Phase 1 candidates JSON")
    p.add_argument("--out", required=True, help="Deduped candidates JSON")
    p.add_argument(
        "--title-ratio",
        type=float,
        default=92.0,
        help="Levenshtein ratio threshold for fuzzy title match (default: 92)",
    )
    args = p.parse_args()

    with open(args.inp) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        candidates = raw
    else:
        candidates = raw.get("candidates") or raw.get("papers") or []

    if not isinstance(candidates, list):
        print("ERROR: input must be a JSON array or object with 'candidates' key", file=sys.stderr)
        return 1

    before = len(candidates)
    result = dedup(candidates, title_ratio_threshold=args.title_ratio)
    after = len(result)
    removed = before - after

    out_obj = {
        "candidates": result,
        "n_before_dedup": before,
        "n_after_dedup": after,
        "n_removed": removed,
    }
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print(f"OK: {before} candidates → {after} unique ({removed} duplicates removed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
