#!/usr/bin/env python3
"""
validate_inputs.py — Verify that a paper-orchestra workspace has the four
required input files in the correct place and minimally well-formed.

This is a deterministic structural check. It does NOT call an LLM, does NOT
talk to the network, and does NOT validate semantic content — that is the
job of the Outline Agent itself.

Exit codes:
    0  all checks passed
    1  one or more required inputs missing or malformed

Usage:
    python validate_inputs.py --workspace /path/to/workspace/
"""
import argparse
import os
import re
import sys

REQUIRED_INPUTS = [
    "idea.md",
    "experimental_log.md",
    "template.tex",
    "conference_guidelines.md",
]


def check_file_exists(path: str) -> list[str]:
    if not os.path.isfile(path):
        return [f"MISSING: {path}"]
    if os.path.getsize(path) == 0:
        return [f"EMPTY: {path}"]
    return []


def check_idea_md(path: str) -> list[str]:
    errors = check_file_exists(path)
    if errors:
        return errors
    text = open(path).read()
    required_headings = ["Problem Statement", "Core Hypothesis"]
    missing = [h for h in required_headings if not re.search(rf"^#+\s*{re.escape(h)}", text, re.M)]
    if missing:
        return [f"WARN: idea.md missing recommended headings: {missing}"]
    return []


def check_experimental_log(path: str) -> list[str]:
    errors = check_file_exists(path)
    if errors:
        return errors
    text = open(path).read()
    if not re.search(r"^##\s+1\.?\s*Experimental Setup", text, re.M):
        return ["WARN: experimental_log.md missing '## 1. Experimental Setup' heading"]
    if not re.search(r"^##\s+2\.?\s*Raw Numeric Data", text, re.M):
        return ["WARN: experimental_log.md missing '## 2. Raw Numeric Data' heading"]
    # Anti-leakage check on log itself: should not reference Figure N or Table N
    leaks = re.findall(r"(?:see|in|from)\s+(?:Figure|Fig\.|Table|Tab\.)\s*\d+", text, re.I)
    if leaks:
        return [
            f"ERROR: experimental_log.md contains figure/table references "
            f"({leaks[:3]}...). Per App. F.2 the log must be self-contained."
        ]
    return []


def check_template(path: str) -> list[str]:
    errors = check_file_exists(path)
    if errors:
        return errors
    text = open(path).read()
    if "\\documentclass" not in text:
        return [f"ERROR: {path} missing \\documentclass — not a LaTeX document"]
    if not re.search(r"\\section\s*\{", text):
        return [
            f"WARN: {path} has no \\section{{...}} commands — outline agent "
            f"will have no skeleton to fill"
        ]
    return []


def check_guidelines(path: str) -> list[str]:
    errors = check_file_exists(path)
    if errors:
        return errors
    text = open(path).read().lower()
    out = []
    if "page" not in text:
        out.append("WARN: conference_guidelines.md does not mention 'page' — " "page limit unclear")
    if "deadline" not in text and "cutoff" not in text and "submission" not in text:
        out.append(
            "WARN: conference_guidelines.md does not mention a deadline / "
            "cutoff — literature review agent will not be able to scope citations"
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True, help="Path to the workspace directory")
    args = p.parse_args()

    ws = os.path.abspath(args.workspace)
    inputs = os.path.join(ws, "inputs")
    if not os.path.isdir(inputs):
        print(f"ERROR: {inputs} does not exist. Run init_workspace.py first.", file=sys.stderr)
        return 1

    all_problems: list[str] = []
    checks = {
        "idea.md": check_idea_md,
        "experimental_log.md": check_experimental_log,
        "template.tex": check_template,
        "conference_guidelines.md": check_guidelines,
    }
    for fname, fn in checks.items():
        problems = fn(os.path.join(inputs, fname))
        for p_ in problems:
            all_problems.append(p_)

    figs = os.path.join(inputs, "figures")
    if os.path.isdir(figs):
        n_figs = len(
            [f for f in os.listdir(figs) if f.lower().endswith((".png", ".pdf", ".jpg", ".jpeg"))]
        )
        print(f"INFO: {n_figs} pre-existing figure(s) in inputs/figures/")
    else:
        print("INFO: no inputs/figures/ — plotting agent will generate everything")

    if not all_problems:
        print("OK: all 4 required inputs present and well-formed.")
        return 0

    fatal = [
        p
        for p in all_problems
        if p.startswith("ERROR") or p.startswith("MISSING") or p.startswith("EMPTY")
    ]
    warn = [p for p in all_problems if p.startswith("WARN")]
    for p_ in fatal:
        print(p_, file=sys.stderr)
    for p_ in warn:
        print(p_)
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
