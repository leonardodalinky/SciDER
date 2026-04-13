#!/usr/bin/env python3
"""
render_diagram.py — Render a simple boxes-and-arrows conceptual diagram from
a JSON spec using matplotlib patches. 300 DPI PNG output.

For complex Fig-1-style overview diagrams (multi-row branches, grouped
sub-systems), the host agent should write matplotlib patches code directly
following references/diagram-patterns.md. This script handles the common
case: nodes positioned on a grid, edges between named nodes.

Spec format (JSON):
    {
        "aspect_ratio": "16:9",
        "nodes": [
            {"id": "input",  "x": 0.5, "y": 3.0, "w": 1.5, "h": 0.7,
             "label": "Idea (I)",  "kind": "input"},
            {"id": "outline","x": 3.0, "y": 3.0, "w": 1.6, "h": 0.9,
             "label": "Outline\\nAgent", "kind": "agent"}
        ],
        "edges": [
            {"from": "input", "to": "outline"}
        ]
    }

Coordinates are in arbitrary units; the script auto-scales to fill the
figure. `kind` ∈ {"input", "agent", "output", "control"} drives the color.

Usage:
    python render_diagram.py --spec diagram.json --out figure.png
"""
import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ASPECT_TO_SIZE = {
    "1:1": (3.4, 3.4),
    "1:4": (1.8, 7.2),
    "2:3": (3.4, 5.1),
    "3:2": (5.1, 3.4),
    "3:4": (3.0, 4.0),
    "4:1": (7.0, 1.75),
    "4:3": (4.0, 3.0),
    "4:5": (3.2, 4.0),
    "5:4": (4.5, 3.6),
    "9:16": (2.8, 4.97),
    "16:9": (5.5, 3.09),
    "21:9": (7.0, 3.0),
}

KIND_COLORS = {
    "input": "#cfe2f3",
    "agent": "#9fc5e8",
    "output": "#b6d7a8",
    "control": "#ead1dc",
    "default": "#e8e8f0",
}
BORDER = "#2060cc"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True, help="path to JSON spec")
    p.add_argument("--out", required=True, help="path to output PNG")
    args = p.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    size = ASPECT_TO_SIZE.get(spec.get("aspect_ratio", "16:9"))
    if size is None:
        print(f"ERROR: unknown aspect_ratio {spec.get('aspect_ratio')}", file=sys.stderr)
        return 1

    nodes = {n["id"]: n for n in spec["nodes"]}
    if not nodes:
        print("ERROR: spec has no nodes", file=sys.stderr)
        return 1

    # auto-bounds
    xs = [n["x"] for n in nodes.values()] + [n["x"] + n.get("w", 1) for n in nodes.values()]
    ys = [n["y"] for n in nodes.values()] + [n["y"] + n.get("h", 1) for n in nodes.values()]
    pad = 0.4
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad

    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("auto")
    ax.axis("off")

    # nodes
    for n in nodes.values():
        color = KIND_COLORS.get(n.get("kind", "default"), KIND_COLORS["default"])
        bb = FancyBboxPatch(
            (n["x"], n["y"]),
            n.get("w", 1.5),
            n.get("h", 0.8),
            boxstyle="round,pad=0.06,rounding_size=0.12",
            ec=BORDER,
            fc=color,
            lw=0.8,
        )
        ax.add_patch(bb)
        cx = n["x"] + n.get("w", 1.5) / 2
        cy = n["y"] + n.get("h", 0.8) / 2
        ax.text(
            cx,
            cy,
            n["label"],
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            fontfamily="serif",
        )

    # edges
    for e in spec.get("edges", []):
        n1, n2 = nodes[e["from"]], nodes[e["to"]]
        x1 = n1["x"] + n1.get("w", 1.5)
        y1 = n1["y"] + n1.get("h", 0.8) / 2
        x2 = n2["x"]
        y2 = n2["y"] + n2.get("h", 0.8) / 2
        a = FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="->", mutation_scale=10, color=BORDER, lw=0.9
        )
        ax.add_patch(a)

    if spec.get("title"):
        fig.suptitle(spec["title"], fontsize=9, fontweight="bold", y=0.98)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(
        f"OK: rendered diagram {args.out} ({len(nodes)} nodes, "
        f"{len(spec.get('edges', []))} edges)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
