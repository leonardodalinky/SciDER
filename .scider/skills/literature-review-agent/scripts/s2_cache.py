#!/usr/bin/env python3
"""
s2_cache.py — Persistent Semantic Scholar verification cache.

Problem: Phase 2 verification is throttled to 1 QPS. If a pipeline run
fails partway through (gate error, network timeout, interrupted session),
re-running wastes the full S2 wait time again on already-verified papers.

Solution: a flat JSON cache at workspace/cache/s2_cache.json. On a cache
HIT the script emits the stored response and exits 0 so the caller can skip
the live S2 request. On a cache MISS it exits 1. After a live request the
caller stores the result with --store.

The cache key is derived from the normalized query title (lowercase,
alphanumeric only) so minor whitespace differences still hit.

Usage:

  CHECK mode — exits 0 + prints JSON if cached, else exits 1:
    python s2_cache.py --cache workspace/cache/s2_cache.json \\
        --check "Attention Is All You Need"

  STORE mode — write a response into the cache:
    python s2_cache.py --cache workspace/cache/s2_cache.json \\
        --store "Attention Is All You Need" \\
        --response '{"paperId": "...", "title": "..."}'

  STATS mode — print cache size and hit rate summary:
    python s2_cache.py --cache workspace/cache/s2_cache.json --stats
"""
import argparse
import json
import os
import re
import sys


def norm_key(title: str) -> str:
    """Lowercase, alphanumeric-only cache key."""
    return re.sub(r"[^a-z0-9]", "", title.lower())


def load_cache(path: str) -> dict:
    if os.path.isfile(path):
        with open(path) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_cache(path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, help="Path to cache JSON file")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        metavar="TITLE",
        help="Check for title; exit 0 + print JSON if found, else exit 1",
    )
    mode.add_argument(
        "--store", metavar="TITLE", help="Store a response for TITLE (requires --response)"
    )
    mode.add_argument("--stats", action="store_true", help="Print cache statistics")

    p.add_argument(
        "--response", metavar="JSON", help="S2 response JSON to store (used with --store)"
    )
    args = p.parse_args()

    cache = load_cache(args.cache)

    if args.stats:
        print(f"Cache file : {args.cache}")
        print(f"Entries    : {len(cache)}")
        if cache:
            print("Sample keys:", list(cache.keys())[:5])
        return 0

    if args.check:
        key = norm_key(args.check)
        if key in cache:
            print(json.dumps(cache[key]))
            return 0  # HIT
        return 1  # MISS

    # --store mode
    if not args.response:
        print("ERROR: --store requires --response", file=sys.stderr)
        return 2
    try:
        response = json.loads(args.response)
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON in --response: {e}", file=sys.stderr)
        return 2

    key = norm_key(args.store)
    cache[key] = response
    save_cache(args.cache, cache)
    print(f"OK: cached '{args.store}' → key '{key}' ({len(cache)} total entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
