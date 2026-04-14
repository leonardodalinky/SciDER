#!/usr/bin/env python3
"""
extract_metrics.py — Parse markdown tables out of experimental_log.md's
"## 2. Raw Numeric Data" section into structured JSON.

The Section Writing Agent uses this to construct LaTeX booktabs tables
without re-deriving numeric values from raw markdown text. Per the App. F.1
prompt, "do not hallucinate numbers; use the exact values provided in the
log" — this script makes that mechanical.

Output JSON shape:
    {
      "tables": [
        {
          "label": "Performance comparison on Dataset X",
          "headers": ["Method", "Accuracy", "F1", "Latency (ms)"],
          "rows": [
            ["Baseline", "78.2", "0.79", "12.3"],
            ...
          ]
        },
        ...
      ]
    }

Usage:
    python extract_metrics.py --log experimental_log.md --out metrics.json
"""
import argparse
import json
import re
import sys


def find_raw_data_section(text: str) -> str:
    """Return the slice of text from '## 2. Raw Numeric Data' to the next H2."""
    m = re.search(r"^##\s+2\.?\s*Raw Numeric Data\s*$", text, re.M)
    if not m:
        return ""
    start = m.end()
    next_h2 = re.search(r"^##\s+", text[start:], re.M)
    end = start + next_h2.start() if next_h2 else len(text)
    return text[start:end]


def parse_markdown_tables(section: str) -> list[dict]:
    """Walk the section, extracting markdown tables and their preceding labels."""
    lines = section.split("\n")
    tables: list[dict] = []
    current_label: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Track table labels: ### Table N: Foo  /  ### Table: Foo  /  **Table 1: Foo**
        m = re.match(r"^#+\s*Table[^:]*:\s*(.+?)\s*$", line)
        if m:
            current_label = m.group(1).strip()
            i += 1
            continue
        m = re.match(r"^\*\*Table[^:]*:\s*(.+?)\*\*\s*$", line)
        if m:
            current_label = m.group(1).strip()
            i += 1
            continue

        # Detect table start: a header row followed by a separator row.
        if "|" in line and i + 1 < len(lines):
            sep = lines[i + 1].strip()
            if re.fullmatch(r"\|?\s*[:\-]+\s*(\|\s*[:\-]+\s*)+\|?", sep):
                headers = [c.strip() for c in line.strip("|").split("|")]
                rows: list[list[str]] = []
                j = i + 2
                while j < len(lines) and "|" in lines[j].strip():
                    cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                    if len(cells) >= 2:
                        rows.append(cells)
                    j += 1
                tables.append(
                    {
                        "label": current_label or f"Table {len(tables) + 1}",
                        "headers": headers,
                        "rows": rows,
                    }
                )
                current_label = None
                i = j
                continue

        i += 1

    return tables


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True, help="experimental_log.md path")
    p.add_argument("--out", required=True, help="metrics.json output path")
    args = p.parse_args()

    text = open(args.log).read()
    section = find_raw_data_section(text)
    if not section:
        print("WARN: no '## 2. Raw Numeric Data' section found", file=sys.stderr)
        with open(args.out, "w") as f:
            json.dump({"tables": []}, f, indent=2)
        return 0

    tables = parse_markdown_tables(section)
    out = {"tables": tables}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"OK: extracted {len(tables)} table(s) → {args.out}")
    for t in tables:
        print(f"  - {t['label']}: {len(t['headers'])} cols × {len(t['rows'])} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
