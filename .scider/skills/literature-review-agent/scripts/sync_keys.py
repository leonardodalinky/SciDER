#!/usr/bin/env python3
r"""
sync_keys.py — Synchronize citation keys in a .tex file with the canonical
bibtex_key values stored in citation_pool.json.

Problem: The Literature Review Agent writes cite keys in its own format
(e.g. 'lewis2020rag'), while bibtex_format.py generates canonical keys from
author + year + first-significant-title-word (e.g. 'lewis2020retrievalaugmented').
After running bibtex_format.py these two sources are out of sync, causing the
citation_coverage gate to fail (it looks for \cite{canonical_key} in the .tex).

This script reads the 'key' -> 'bibtex_key' mapping from citation_pool.json
and performs a targeted substitution inside \cite{}, \citep{}, \citet{}
commands in the target .tex file. It handles multi-key citations like
\cite{a,b,c} correctly.

Run this immediately after bibtex_format.py, before Step 4 (Section Writing).

Usage:
    python sync_keys.py \
        --pool workspace/citation_pool.json \
        --tex  workspace/drafts/intro_relwork.tex \
        --inplace

    # Without --inplace: prints updated content to stdout (safe preview mode).
"""
import argparse
import json
import re
import sys

# Matches \cite, \citep, \citet, \citealt, \citealp, \citeauthor, \citeyear,
# starred variants like \cite*, and the optional [prenote][postnote] args.
CITE_RE = re.compile(
    r"(\\cite[a-zA-Z*]*)"  # command
    r"(?:\[[^\]]*\])*"  # optional bracket args (prenote/postnote)
    r"\{([^}]+)\}"  # required brace arg with keys
)


def build_key_map(pool: dict) -> dict[str, str]:
    """Return {agent_key: bibtex_key} for every paper where they differ."""
    key_map: dict[str, str] = {}
    for paper in pool.get("papers", []):
        old = paper.get("key")
        new = paper.get("bibtex_key")
        if old and new and old != new:
            key_map[old] = new
    return key_map


def replace_keys(content: str, key_map: dict[str, str]) -> tuple[str, int]:
    if not key_map:
        return content, 0

    n_replaced = 0

    def replacer(m: re.Match) -> str:
        nonlocal n_replaced
        cmd = m.group(1)
        keys_str = m.group(2)
        keys = [k.strip() for k in keys_str.split(",")]
        new_keys: list[str] = []
        for k in keys:
            if k in key_map:
                new_keys.append(key_map[k])
                n_replaced += 1
            else:
                new_keys.append(k)
        # Reconstruct original bracket args (they were consumed by the regex
        # but we don't need to preserve them specially — re-emit as matched)
        full_match = m.group(0)
        # Rebuild: command + everything between command and { + new keys
        bracket_part = full_match[len(cmd) : full_match.index("{")]
        return f"{cmd}{bracket_part}{{{', '.join(new_keys)}}}"

    updated = CITE_RE.sub(replacer, content)
    return updated, n_replaced


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True, help="citation_pool.json")
    p.add_argument("--tex", required=True, help="Target .tex file to update")
    p.add_argument(
        "--inplace", action="store_true", help="Overwrite --tex in place (default: print to stdout)"
    )
    args = p.parse_args()

    with open(args.pool) as f:
        pool = json.load(f)
    key_map = build_key_map(pool)

    if not key_map:
        print("OK: no key differences in citation_pool.json — nothing to sync")
        return 0

    print(f"Key map ({len(key_map)} substitutions):")
    for old, new in key_map.items():
        print(f"  {old} → {new}")

    with open(args.tex) as f:
        content = f.read()

    updated, n = replace_keys(content, key_map)

    if args.inplace:
        with open(args.tex, "w") as f:
            f.write(updated)
        print(f"OK: {n} citation key(s) updated in {args.tex}")
    else:
        sys.stdout.write(updated)
        print(f"\n# sync_keys: {n} substitution(s) would be made", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
