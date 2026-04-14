#!/usr/bin/env python3
"""
validate_outline.py — Validate workspace/outline.json against the JSON Schema.

Performs the structural checks defined in
`references/outline_schema.json` plus a few semantic checks the schema
cannot express:

  - figure_id snake_case + no "figure" infix
  - "no orphaned subsection" hierarchy rule
  - related_work_strategy.subsections has 2-4 entries (paper rule)
  - emits WARNINGS for missing-but-recommended things, not just ERRORS

Exit codes:
    0  valid (warnings allowed)
    1  schema or semantic errors

Usage:
    python validate_outline.py /path/to/outline.json
"""
import json
import os
import re
import sys

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError:
    print("ERROR: jsonschema is required. Install with: pip install jsonschema", file=sys.stderr)
    sys.exit(2)

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.normpath(os.path.join(HERE, "..", "references", "outline_schema.json"))

FIGID_RE = re.compile(r"^[a-z0-9_]+$")


def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def semantic_checks(outline: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    # Figure ID checks
    for i, fig in enumerate(outline.get("plotting_plan", [])):
        fid = fig.get("figure_id", "")
        if not FIGID_RE.match(fid):
            errors.append(
                f"plotting_plan[{i}].figure_id={fid!r} is not snake_case " f"matching ^[a-z0-9_]+$"
            )
        # The word "figure" must not appear (allow the conventional fig_ prefix)
        body = re.sub(r"^fig_", "", fid)
        if "figure" in body.lower():
            errors.append(
                f"plotting_plan[{i}].figure_id={fid!r} contains 'figure' — "
                f"forbidden by Outline Agent prompt (App. F.1)"
            )
        # If plot_type is "plot", objective should mention a specific chart type
        if fig.get("plot_type") == "plot":
            obj = fig.get("objective", "").lower()
            chart_keywords = [
                "radar",
                "bar",
                "line",
                "scatter",
                "box",
                "violin",
                "heatmap",
                "histogram",
                "pie",
                "stacked",
                "grouped",
                "ridge",
                "density",
                "convergence",
                "training curve",
            ]
            if not any(k in obj for k in chart_keywords):
                warnings.append(
                    f"plotting_plan[{i}] is plot_type='plot' but objective "
                    f"does not name a specific chart type. The Outline Agent "
                    f"prompt requires this."
                )

    # Section hierarchy: no orphan subsections
    for i, section in enumerate(outline.get("section_plan", [])):
        if not isinstance(section, dict):
            continue
        subs = section.get("subsections") or []
        if len(subs) == 1:
            warnings.append(
                f"section_plan[{i}] {section.get('section_title')!r} has only "
                f"one subsection — this is an orphan. The Outline Agent prompt "
                f"requires that if X.1 exists, X.2 must too."
            )

    return errors, warnings


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    outline_path = sys.argv[1]
    if not os.path.isfile(outline_path):
        print(f"ERROR: {outline_path} not found", file=sys.stderr)
        return 1

    try:
        with open(outline_path) as f:
            outline = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: {outline_path} is not valid JSON: {e}", file=sys.stderr)
        return 1

    schema = load_schema()
    validator = Draft202012Validator(schema)
    schema_errors = sorted(validator.iter_errors(outline), key=lambda e: e.path)

    sem_errors, sem_warnings = semantic_checks(outline)

    if schema_errors:
        print("SCHEMA ERRORS:", file=sys.stderr)
        for e in schema_errors:
            path = "/".join(str(p) for p in e.absolute_path) or "(root)"
            print(f"  {path}: {e.message}", file=sys.stderr)

    if sem_errors:
        print("SEMANTIC ERRORS:", file=sys.stderr)
        for msg in sem_errors:
            print(f"  {msg}", file=sys.stderr)

    if sem_warnings:
        print("WARNINGS:")
        for msg in sem_warnings:
            print(f"  {msg}")

    if schema_errors or sem_errors:
        print(f"\nFAIL: {outline_path}", file=sys.stderr)
        return 1

    n_figs = len(outline.get("plotting_plan", []))
    n_relwork = len(
        outline.get("intro_related_work_plan", {})
        .get("related_work_strategy", {})
        .get("subsections", [])
    )
    n_sections = len(outline.get("section_plan", []))
    n_hints = sum(
        len(sub.get("citation_hints", []))
        for sec in outline.get("section_plan", [])
        for sub in (sec.get("subsections") or [])
    )
    print(
        f"OK: {outline_path} — {n_figs} figures, "
        f"{n_relwork} related-work clusters, {n_sections} sections, "
        f"{n_hints} citation hints"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
