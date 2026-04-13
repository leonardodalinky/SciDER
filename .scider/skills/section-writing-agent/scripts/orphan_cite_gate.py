#!/usr/bin/env python3
"""
orphan_cite_gate.py — Verify every \\cite{KEY} in a LaTeX file resolves to
an entry in refs.bib.

The Section Writing Agent prompt mandates "use ONLY the keys found in
citation_map.json". This script enforces it deterministically.

Exit codes:
    0  every cite key resolves
    1  one or more orphan cite keys

Usage:
    python orphan_cite_gate.py paper.tex refs.bib
"""
import re
import sys

CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citeauthor|citeyear|autocite|parencite|textcite)"
    r"(?:\[[^\]]*\])?"
    r"\{([^}]+)\}"
)
BIB_KEY_RE = re.compile(r"^@\w+\{\s*([^,\s]+)", re.M)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    tex_path, bib_path = sys.argv[1], sys.argv[2]
    tex = open(tex_path).read()
    bib = open(bib_path).read()

    bib_keys = set(BIB_KEY_RE.findall(bib))
    if not bib_keys:
        print(f"ERROR: no @entry keys found in {bib_path}", file=sys.stderr)
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
