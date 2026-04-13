#!/usr/bin/env python3
"""
render_matplotlib.py — Render a JSON plot spec to a 300-DPI PNG using the
academic-paper matplotlib style.

This is a deterministic helper for the plotting agent. It handles the
mechanical parts (figsize from aspect ratio, style application, save) so the
host agent can focus on high-level visual decisions in the LLM.

Spec format (JSON):
    {
        "type": "line" | "bar" | "grouped_bar" | "stacked_bar" | "radar" | "scatter" | "heatmap",
        "aspect_ratio": "16:9",
        "title": "Optional title",
        "xlabel": "...",
        "ylabel": "...",
        "series": [
            {"name": "Method A", "x": [...], "y": [...]},
            {"name": "Method B", "x": [...], "y": [...]}
        ],
        "x_labels": [...],     // for bar/grouped_bar/stacked_bar
        "legend_loc": "upper right"
    }

For chart types not covered here, write the matplotlib code yourself directly.

Usage:
    python render_matplotlib.py --spec spec.json --out figure.png
"""
import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── style ──
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "legend.fontsize": 7,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.3,
    }
)

PALETTE = ["#2060cc", "#cc3030", "#208040", "#cc7020", "#8040cc", "#b08020", "#666666"]

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


def make_axes(spec):
    size = ASPECT_TO_SIZE.get(spec.get("aspect_ratio", "16:9"))
    if size is None:
        raise SystemExit(f"unknown aspect_ratio: {spec.get('aspect_ratio')}")
    is_polar = spec.get("type") == "radar"
    fig, ax = plt.subplots(figsize=size, subplot_kw=dict(polar=True) if is_polar else {})
    return fig, ax


def render_line(ax, spec):
    for i, s in enumerate(spec["series"]):
        ax.plot(s["x"], s["y"], color=PALETTE[i % len(PALETTE)], label=s.get("name"))
    if spec.get("xlabel"):
        ax.set_xlabel(spec["xlabel"])
    if spec.get("ylabel"):
        ax.set_ylabel(spec["ylabel"])
    if any("name" in s for s in spec["series"]):
        ax.legend(loc=spec.get("legend_loc", "best"))
    ax.grid(True)


def render_bar(ax, spec):
    s = spec["series"][0]
    x = np.arange(len(s["y"]))
    ax.bar(x, s["y"], color=PALETTE[0], edgecolor="white", linewidth=0.4)
    if spec.get("x_labels"):
        ax.set_xticks(x)
        ax.set_xticklabels(spec["x_labels"])
    if spec.get("xlabel"):
        ax.set_xlabel(spec["xlabel"])
    if spec.get("ylabel"):
        ax.set_ylabel(spec["ylabel"])
    ax.grid(axis="y", alpha=0.2)


def render_grouped_bar(ax, spec):
    n_groups = len(spec["series"][0]["y"])
    n_series = len(spec["series"])
    x = np.arange(n_groups)
    width = 0.8 / n_series
    for i, s in enumerate(spec["series"]):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(
            x + offset,
            s["y"],
            width,
            color=PALETTE[i % len(PALETTE)],
            label=s.get("name"),
            edgecolor="white",
            linewidth=0.4,
        )
    if spec.get("x_labels"):
        ax.set_xticks(x)
        ax.set_xticklabels(spec["x_labels"])
    if spec.get("xlabel"):
        ax.set_xlabel(spec["xlabel"])
    if spec.get("ylabel"):
        ax.set_ylabel(spec["ylabel"])
    ax.legend(loc=spec.get("legend_loc", "best"))
    ax.grid(axis="y", alpha=0.2)


def render_stacked_bar(ax, spec):
    x = np.arange(len(spec["x_labels"]))
    bottom = np.zeros(len(spec["x_labels"]))
    for i, s in enumerate(spec["series"]):
        y = np.array(s["y"])
        ax.bar(
            x,
            y,
            bottom=bottom,
            color=PALETTE[i % len(PALETTE)],
            label=s.get("name"),
            edgecolor="white",
            linewidth=0.4,
        )
        bottom += y
    ax.set_xticks(x)
    ax.set_xticklabels(spec["x_labels"])
    if spec.get("ylabel"):
        ax.set_ylabel(spec["ylabel"])
    ax.legend(loc=spec.get("legend_loc", "best"))


def render_radar(ax, spec):
    labels = spec["x_labels"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    for i, s in enumerate(spec["series"]):
        vals = list(s["y"]) + [s["y"][0]]
        ax.plot(angles, vals, color=PALETTE[i % len(PALETTE)], linewidth=1.5, label=s.get("name"))
        ax.fill(angles, vals, color=PALETTE[i % len(PALETTE)], alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))


def render_scatter(ax, spec):
    for i, s in enumerate(spec["series"]):
        ax.scatter(
            s["x"], s["y"], color=PALETTE[i % len(PALETTE)], s=20, alpha=0.7, label=s.get("name")
        )
    if spec.get("xlabel"):
        ax.set_xlabel(spec["xlabel"])
    if spec.get("ylabel"):
        ax.set_ylabel(spec["ylabel"])
    if any("name" in s for s in spec["series"]):
        ax.legend(loc=spec.get("legend_loc", "best"))
    ax.grid(True)


def render_heatmap(ax, spec):
    data = np.array(spec["matrix"])
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    if spec.get("x_labels"):
        ax.set_xticks(range(len(spec["x_labels"])))
        ax.set_xticklabels(spec["x_labels"])
    if spec.get("y_labels"):
        ax.set_yticks(range(len(spec["y_labels"])))
        ax.set_yticklabels(spec["y_labels"])
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)


RENDERERS = {
    "line": render_line,
    "bar": render_bar,
    "grouped_bar": render_grouped_bar,
    "stacked_bar": render_stacked_bar,
    "radar": render_radar,
    "scatter": render_scatter,
    "heatmap": render_heatmap,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True, help="path to JSON spec")
    p.add_argument("--out", required=True, help="path to output PNG")
    args = p.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    chart_type = spec.get("type")
    renderer = RENDERERS.get(chart_type)
    if renderer is None:
        print(
            f"ERROR: unknown chart type {chart_type!r}. " f"Allowed: {sorted(RENDERERS)}",
            file=sys.stderr,
        )
        return 1

    fig, ax = make_axes(spec)
    renderer(ax, spec)

    if spec.get("title"):
        ax.set_title(spec["title"])

    if chart_type != "radar":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print(f"OK: rendered {args.out} ({spec.get('aspect_ratio')}, {chart_type})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
