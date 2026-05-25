#!/usr/bin/env python3
"""
orphan_cite_gate.py — Verify every \\cite{KEY} in a LaTeX file resolves to
an entry in refs.bib.

The Section Writing Agent prompt mandates "use ONLY the keys found in
citation_map.json". This script enforces it deterministically.

Exit codes:
    0  every cite key resolves
    1  one or more orphan cite keys, or refs.bib contains fabricated entries
       (keys absent from the verified citation_pool.json)

Usage:
    python orphan_cite_gate.py paper.tex refs.bib [citation_pool.json]

If the optional citation_pool.json is given, the gate ALSO fails when refs.bib
contains any key that is not present in the verified pool — this catches the
agent hand-writing/fabricating bibliography entries instead of using the
pre-built, verified pool.
"""
import json
import re
import sys

CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citeauthor|citeyear|autocite|parencite|textcite)"
    r"(?:\[[^\]]*\])?"
    r"\{([^}]+)\}"
)
BIB_KEY_RE = re.compile(r"^@\w+\{\s*([^,\s]+)", re.M)


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2

    tex_path, bib_path = sys.argv[1], sys.argv[2]
    pool_path = sys.argv[3] if len(sys.argv) == 4 else None
    tex = open(tex_path).read()
    bib = open(bib_path).read()

    bib_keys = set(BIB_KEY_RE.findall(bib))
    if not bib_keys:
        print(f"ERROR: no @entry keys found in {bib_path}", file=sys.stderr)
        return 1

    # Anti-fabrication check: every refs.bib key must come from the verified pool.
    if pool_path:
        pool = json.load(open(pool_path))
        pool_keys = {
            p.get("bibtex_key")
            for p in pool.get("papers", [])
            if p.get("bibtex_key")
        }
        fabricated = sorted(bib_keys - pool_keys) if pool_keys else []
        if fabricated:
            print(
                f"\nFAIL: {len(fabricated)} refs.bib entr(y/ies) NOT in the verified "
                f"citation pool — looks fabricated. Rebuild refs.bib from the pool "
                f"with bibtex_format.py; never hand-write entries:",
                file=sys.stderr,
            )
            for k in fabricated:
                print(f"  - {k}", file=sys.stderr)
            return 1

    cite_keys: set[str] = set()
    for m in CITE_RE.finditer(tex):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                cite_keys.add(k)

    orphans = sorted(cite_keys - bib_keys)
    unused = sorted(bib_keys - cite_keys)

    print(f"refs.bib has {len(bib_keys)} entries; {tex_path} cites {len(cite_keys)} unique keys")

    if orphans:
        print(f"\nFAIL: {len(orphans)} orphan \\cite key(s) (not in refs.bib):", file=sys.stderr)
        for k in orphans:
            print(f"  - {k}", file=sys.stderr)
        return 1

    if unused:
        # Just informational. The literature-review-agent's citation_coverage.py
        # is the gate that enforces ≥90% integration.
        print(f"INFO: {len(unused)} bib entries not yet cited (informational)")

    print("OK: no orphan cite keys")
    return 0


if __name__ == "__main__":
    sys.exit(main())
