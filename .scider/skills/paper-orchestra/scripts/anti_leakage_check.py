#!/usr/bin/env python3
"""
anti_leakage_check.py — Deterministic post-hoc check that the LLM obeyed the
Universal Anti-Leakage Prompt (arXiv:2604.05018, App. D.4).

Greps a generated LaTeX paper for forbidden artifacts:

  - Email addresses (a-z0-9.+_-@a-z0-9.-)
  - "corresponding author" / "Correspondence to" phrases
  - Common affiliation tokens (Google, OpenAI, Microsoft, DeepMind, FAIR, etc.)
  - Author-list patterns (e.g., "FirstName LastName, FirstName LastName, ...")
    in the title-block region (before \\begin{document} or first \\section)

Exit codes:
    0  no leaks found
    1  leaks found (the orchestrator should reject the draft and re-prompt)

This is a safety net, NOT a substitute for the prompt itself.

Usage:
    python anti_leakage_check.py path/to/paper.tex
"""
import re
import sys

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+")
CORR_AUTHOR_RE = re.compile(r"corresponding\s+author|correspondence\s+to", re.I)

# Affiliation tokens that appear in real papers' title blocks. These are
# heuristics; false positives are possible if a paper legitimately discusses
# one of these as a topic. The check fires only on the title-block region.
AFFILIATION_TOKENS = [
    "Google",
    "OpenAI",
    "Microsoft",
    "DeepMind",
    "Meta AI",
    "FAIR",
    "Anthropic",
    "Stanford",
    "MIT",
    "Berkeley",
    "CMU",
    "Tsinghua",
    "Peking University",
    "Cornell",
]

# An author-list pattern: "Name1, Name2, Name3 and Name4" or with superscripts.
AUTHOR_LIST_RE = re.compile(
    r"(?:[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*\^\d+|\s*\$\^\{\d+\}\$)?(?:\s*,\s*|\s+and\s+)){2,}"
    r"[A-Z][a-z]+\s+[A-Z][a-z]+"
)


def get_titleblock(text: str) -> str:
    """Return the region between \\begin{document} (or top) and the first \\section."""
    start_m = re.search(r"\\begin\{document\}", text)
    start = start_m.end() if start_m else 0
    end_m = re.search(r"\\section\b", text[start:])
    end = start + end_m.start() if end_m else min(start + 4000, len(text))
    return text[start:end]


def check(path: str) -> int:
    text = open(path).read()
    leaks: list[str] = []

    for m in EMAIL_RE.finditer(text):
        leaks.append(f"  email: {m.group()}")

    for m in CORR_AUTHOR_RE.finditer(text):
        ctx = text[max(0, m.start() - 30) : m.end() + 30].replace("\n", " ")
        leaks.append(f"  corresponding-author phrase: ...{ctx}...")

    titleblock = get_titleblock(text)
    for tok in AFFILIATION_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\b", titleblock):
            leaks.append(f"  affiliation token in title block: {tok}")

    m = AUTHOR_LIST_RE.search(titleblock)
    if m:
        leaks.append(f"  author-list-like pattern in title block: {m.group()[:80]}...")

    if leaks:
        print(f"FAIL: anti-leakage violations in {path}:", file=sys.stderr)
        for line in leaks:
            print(line, file=sys.stderr)
        print("\nThe Anti-Leakage Prompt (App. D.4) forbids these. Re-prompt the", file=sys.stderr)
        print("writing step with explicit instructions to remove these artifacts.", file=sys.stderr)
        return 1

    print(f"OK: no anti-leakage violations in {path}")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    return check(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
