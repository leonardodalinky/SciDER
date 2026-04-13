#!/usr/bin/env python3
"""
latex_sanity.py — Deterministic structural checks on a generated LaTeX file.

Catches the most common ways the Section Writing Agent's output can fail to
compile, before invoking latexmk:

  1. Unmatched braces (counts \\{ and \\} but ignores escaped ones)
  2. Mismatched \\begin{X} / \\end{X} environments
  3. Unescaped special characters (& % _ outside math/verbatim contexts)
     — heuristic only; common false positives in tabular cells
  4. Duplicate \\label{...}
  5. Missing \\documentclass

Exit codes:
    0  no errors found
    1  one or more errors

Usage:
    python latex_sanity.py path/to/paper.tex
"""
import re
import sys


def check_braces(text: str) -> list[str]:
    # Strip escaped braces and comments
    stripped = re.sub(r"%[^\n]*", "", text)
    stripped = stripped.replace("\\{", "").replace("\\}", "")
    n_open = stripped.count("{")
    n_close = stripped.count("}")
    if n_open != n_close:
        return [f"unmatched braces: {{ × {n_open}, }} × {n_close} (delta {n_open - n_close})"]
    return []


def check_environments(text: str) -> list[str]:
    starred = lambda s: s.replace("*", r"\*")  # noqa: E731
    starts = re.findall(r"\\begin\{([^}]+)\}", text)
    ends = re.findall(r"\\end\{([^}]+)\}", text)
    errors: list[str] = []

    stack: list[str] = []
    pos = 0
    # Walk in order, push starts, pop on ends
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            if not stack:
                errors.append(f"\\end{{{env}}} with no matching \\begin")
                continue
            top = stack.pop()
            if top != env:
                errors.append(f"\\begin{{{top}}} closed by \\end{{{env}}}")

    if stack:
        errors.append(f"unclosed environments: {stack}")
    return errors


def check_documentclass(text: str) -> list[str]:
    if not re.search(r"\\documentclass", text):
        return ["missing \\documentclass — not a complete LaTeX document"]
    return []


def check_duplicate_labels(text: str) -> list[str]:
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    seen: dict[str, int] = {}
    for l in labels:
        seen[l] = seen.get(l, 0) + 1
    dupes = [l for l, n in seen.items() if n > 1]
    if dupes:
        return [f"duplicate labels: {dupes}"]
    return []


def check_unescaped_specials(text: str) -> list[str]:
    """Heuristic: look for & % _ that appear OUTSIDE tabular/math/verbatim
    environments. False positives are common; we only emit WARNINGS, not errors."""
    warnings: list[str] = []
    # Strip math, tabular, verbatim, comments
    s = re.sub(r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}", "", text, flags=re.S)
    s = re.sub(
        r"\\begin\{(equation|align|array|matrix|verbatim|lstlisting)\*?\}.*?\\end\{\1\*?\}",
        "",
        s,
        flags=re.S,
    )
    s = re.sub(r"\$[^$]*\$", "", s)
    s = re.sub(r"%[^\n]*", "", s)
    s = re.sub(r"\\[%&_#$]", "", s)  # remove already-escaped
    bad = re.findall(r"[%&_]", s)
    if bad:
        warnings.append(f"WARN: {len(bad)} potentially unescaped %, &, or _ outside math/tabular")
    return warnings


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = sys.argv[1]
    text = open(path).read()

    errors: list[str] = []
    errors += check_documentclass(text)
    errors += check_braces(text)
    errors += check_environments(text)
    errors += check_duplicate_labels(text)
    warnings = check_unescaped_specials(text)

    for w in warnings:
        print(w)

    if errors:
        print(f"\nFAIL: {len(errors)} latex sanity error(s) in {path}", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"OK: {path} passes structural sanity checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
