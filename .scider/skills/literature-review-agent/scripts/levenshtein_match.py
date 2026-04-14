#!/usr/bin/env python3
"""
levenshtein_match.py — Fuzzy title match for citation verification.

Implements the paper's Rule 1 (App. D.3): a candidate paper passes only if
its title's Levenshtein ratio against the Semantic Scholar hit's title is
strictly greater than 70.

Includes a substring-bypass safety net for short candidate titles (the
Linformer false-negative case): if the candidate is < 4 words and is
contained as a substring in the S2 hit's title, return 100.

Exit code is always 0; the integer ratio is printed to stdout. The caller
parses it and decides whether to discard.

Usage:
    python levenshtein_match.py --candidate "..." --found "..."
    python levenshtein_match.py --candidate "..." --found "..." --substring-bypass
"""
import argparse
import re
import sys

try:
    import Levenshtein
except ImportError:
    print(
        "ERROR: python-Levenshtein required. Install with: pip install python-Levenshtein",
        file=sys.stderr,
    )
    sys.exit(2)


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def ratio(a: str, b: str, substring_bypass: bool = False) -> int:
    na, nb = normalize(a), normalize(b)
    r = int(round(Levenshtein.ratio(na, nb) * 100))
    if substring_bypass and len(na.split()) < 4:
        if na in nb:
            return max(r, 95)
    return r


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidate", required=True, help="The original candidate title (from web search)"
    )
    p.add_argument("--found", required=True, help="The title returned by Semantic Scholar")
    p.add_argument(
        "--substring-bypass",
        action="store_true",
        help="Bump short-candidate substring matches to 95",
    )
    p.add_argument(
        "--threshold", type=int, default=70, help="Print PASS/FAIL alongside the ratio (default 70)"
    )
    args = p.parse_args()

    r = ratio(args.candidate, args.found, args.substring_bypass)
    verdict = "PASS" if r > args.threshold else "FAIL"
    print(f"{r} {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
