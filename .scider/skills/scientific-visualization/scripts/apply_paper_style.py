"""
Publication-quality matplotlib style for scientific papers.

Usage:
    from apply_paper_style import apply_paper_style, OKABE_ITO_COLORS, save_figure
    apply_paper_style()
    # ... your plotting code ...
    save_figure(fig, "results/figure1.pdf")
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Okabe-Ito colorblind-friendly palette (8 colors)
OKABE_ITO_COLORS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# Wong palette (alternative 8-color colorblind-safe palette)
WONG_COLORS = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]


def apply_paper_style(
    font_size: int = 10,
    tick_size: int = 9,
    legend_size: int = 9,
    title_size: int = 12,
    line_width: float = 1.5,
    use_latex: bool = False,
    spine_style: str = "clean",
):
    """Apply publication-quality matplotlib rcParams.

    Args:
        font_size: Base font size for axis labels (pt). Min 8pt for most journals.
        tick_size: Font size for tick labels.
        legend_size: Font size for legend text.
        title_size: Font size for figure/axes titles.
        line_width: Default line width for plots.
        use_latex: Enable LaTeX rendering (requires LaTeX installation).
        spine_style: 'clean' (remove top/right spines) or 'box' (keep all spines).
    """
    params = {
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": title_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "legend.title_fontsize": font_size,
        # Lines
        "lines.linewidth": line_width,
        "lines.markersize": 5,
        "patch.linewidth": line_width,
        # Axes
        "axes.linewidth": 0.8,
        "axes.labelpad": 4,
        "axes.titlepad": 6,
        "axes.prop_cycle": plt.cycler("color", OKABE_ITO_COLORS),
        # Grid (subtle)
        "axes.grid": False,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        # Ticks
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,
        # Figure
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "figure.autolayout": False,
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.5,
        # LaTeX
        "text.usetex": use_latex,
    }
    plt.rcParams.update(params)

    if spine_style == "clean":
        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )


def save_figure(fig, path: str, dpi: int = 300, fmt: str = None):
    """Save a figure at publication quality.

    Args:
        fig: matplotlib Figure object.
        path: Output path. Extension determines format if fmt is None.
        dpi: Resolution (dots per inch). 300 minimum for raster (PNG/TIFF).
        fmt: Override format ('pdf', 'png', 'svg', 'tiff').
    """
    import pathlib

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(dpi=dpi, bbox_inches="tight", facecolor="white")
    if fmt:
        kwargs["format"] = fmt

    fig.savefig(path, **kwargs)
    print(f"Saved: {path} ({p.suffix or fmt}, dpi={dpi})")


def add_significance_annotation(
    ax,
    x1: float,
    x2: float,
    y: float,
    h: float = 0.02,
    text: str = "*",
    color: str = "black",
    lw: float = 1.0,
):
    """Draw a significance bracket between two x positions.

    Args:
        ax: matplotlib Axes object.
        x1, x2: x-positions of the two groups being compared.
        y: y-position of the bracket base.
        h: Height of the bracket arms.
        text: Annotation text ('ns', '*', '**', '***').
        color: Line and text color.
        lw: Line width.
    """
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c=color)
    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="bottom",
        color=color,
        fontsize=plt.rcParams["font.size"],
    )


def pvalue_to_stars(p: float) -> str:
    """Convert p-value to significance stars."""
    if p > 0.05:
        return "ns"
    elif p > 0.01:
        return "*"
    elif p > 0.001:
        return "**"
    else:
        return "***"


def single_column_figsize(height_ratio: float = 0.75) -> tuple:
    """Return figure size for single-column paper figure (3.5 inches wide)."""
    return (3.5, 3.5 * height_ratio)


def double_column_figsize(height_ratio: float = 0.4) -> tuple:
    """Return figure size for double-column paper figure (7 inches wide)."""
    return (7.0, 7.0 * height_ratio)


if __name__ == "__main__":
    # Demo: publication-quality figure with the paper style
    apply_paper_style(font_size=10)

    x = np.linspace(0, 2 * np.pi, 100)
    fig, axes = plt.subplots(1, 2, figsize=double_column_figsize(0.45))

    # Left: line plot with confidence band
    ax = axes[0]
    for i, (label, phase) in enumerate(
        [("Method A", 0), ("Method B", np.pi / 4), ("Baseline", np.pi / 2)]
    ):
        y = np.sin(x + phase)
        noise = 0.1 * np.random.randn(len(x))
        ax.plot(x, y, label=label, color=OKABE_ITO_COLORS[i])
        ax.fill_between(x, y - 0.1, y + 0.1, alpha=0.15, color=OKABE_ITO_COLORS[i])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal amplitude")
    ax.set_title("(a) Time series comparison")
    ax.legend(loc="upper right")

    # Right: bar chart with error bars and significance
    ax = axes[1]
    methods = ["Baseline", "Method A", "Method B"]
    means = [0.71, 0.82, 0.85]
    stds = [0.03, 0.02, 0.025]
    colors = OKABE_ITO_COLORS[:3]
    bars = ax.bar(methods, means, yerr=stds, capsize=4, color=colors, width=0.5)
    ax.set_ylabel("F1 Score")
    ax.set_title("(b) Method comparison")
    ax.set_ylim(0.6, 0.95)
    add_significance_annotation(ax, 0, 2, 0.90, h=0.01, text="**")

    plt.tight_layout()
    save_figure(fig, "/tmp/paper_style_demo.pdf")
    print("Demo figure saved to /tmp/paper_style_demo.pdf")
