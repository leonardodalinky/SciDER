---
name: scientific-visualization
description: Publication-quality figure guidelines for scientific papers — DPI, font sizes, colorblind-friendly palettes, plot type selection, multi-panel layout, uncertainty representation, and statistical annotation. Use when creating figures for papers or reports.
allowed_agents: [experiment, native_coding, data]
---

# Scientific Visualization

## Overview

Publication figures are permanent artifacts of a paper. A poorly formatted figure with unlabeled axes, inaccessible colors, or missing uncertainty estimates can cause rejection or mislead readers. This skill specifies the exact standards for producing figures that satisfy journal requirements, remain readable in black-and-white and for colorblind readers, and communicate uncertainty honestly.

## When to Use This Skill

Use this skill when:
- Creating any figure that will appear in a paper, preprint, or technical report
- Choosing between plot types for a dataset
- Setting up matplotlib style for a project
- Adding error bars, confidence bands, or statistical significance annotations
- Saving figures in the correct format and resolution
- Designing multi-panel layouts for a results section
- Verifying that figures are accessible to colorblind readers

---

## Figure Specifications for Publication

### Resolution and Format

| Use Case | Format | DPI | Notes |
|----------|--------|-----|-------|
| Primary figures | PDF or SVG | Vector (infinite) | Preferred for line plots, bar charts, scatter — scales losslessly |
| Photographs / microscopy | TIFF or PNG | ≥ 300 dpi | TIFF preferred for medical/bio journals |
| Web / preprint only | PNG | ≥ 150 dpi | Acceptable but not for camera-ready submission |

**Rule**: always produce a vector version (PDF/SVG) alongside any raster export.

### Figure Width

Match journal column width exactly. Common standards:

| Layout | Width |
|--------|-------|
| Single column | 3.3–3.5 inches (84–89 mm) |
| 1.5 column | 5.0–5.5 inches |
| Double column / full width | 6.5–7.0 inches (165–178 mm) |

Check the author guide for the target journal; many specify mm.

### Font Sizes (Minimum)

| Element | Minimum Size |
|---------|-------------|
| Tick labels | 8 pt |
| Axis labels | 10 pt |
| Legend text | 8 pt |
| Subplot titles | 10 pt |
| Figure title | 12 pt |
| Annotations / significance stars | 8 pt |

These are minimums for print. Aim for 10–11 pt for body text elements.

---

## Colorblind-Friendly Palettes

Approximately 8% of males have color vision deficiency. Use palettes that are distinguishable under deuteranopia and protanopia.

### Okabe-Ito (Wong) 8-Color Palette — Recommended Default

This palette is the standard recommendation of Nature Methods and many scientific style guides.

```python
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
```

### Continuous / Sequential Data

```python
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Good: perceptually uniform, works in grayscale, accessible
# viridis  — default recommendation
# plasma   — higher contrast
# cividis  — designed for color-vision deficiency

plt.imshow(data, cmap='viridis')   # sequential
plt.imshow(data, cmap='cividis')   # alternative, explicitly CVD-safe
```

### Diverging Data (e.g., correlation matrices, residuals)

```python
# RdBu or RdYlBu — both are CVD-safe
plt.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
```

### Applying Okabe-Ito as Default Color Cycle

```python
import matplotlib.pyplot as plt

OKABE_ITO_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OKABE_ITO_COLORS)
```

---

## Plot Type Decision Guide

Choose the plot type that reveals the most information with the least chartjunk.

### Distribution Comparisons

| Situation | Plot Type | Why |
|-----------|-----------|-----|
| n > 30 per group, outliers matter | Box plot | Quartiles + outliers visible |
| n > 30, shape of distribution matters | Violin plot | Density visible; use `cut=0` to avoid extending beyond data range |
| n ≤ 30 (small samples) | Strip / swarm plot | Show every data point; no hidden structure |
| Large n (> 1000), compare shapes | Ridge / KDE plot | Box plots obscure multimodality |

```python
import matplotlib.pyplot as plt
import numpy as np

# Box plot with overlaid strip plot (best practice for moderate n)
fig, ax = plt.subplots(figsize=(5, 4))
groups = {'Control': np.random.normal(50, 10, 40),
          'Treatment A': np.random.normal(60, 12, 40),
          'Treatment B': np.random.normal(55, 8, 40)}

positions = range(len(groups))
bp = ax.boxplot(list(groups.values()), positions=positions,
                patch_artist=True, widths=0.4, showfliers=False)

# Okabe-Ito colors
colors = ["#56B4E9", "#E69F00", "#009E73"]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Overlay individual points
for i, (name, data) in enumerate(groups.items()):
    jitter = np.random.uniform(-0.1, 0.1, len(data))
    ax.scatter(i + jitter, data, alpha=0.5, s=20, color=colors[i], zorder=3)

ax.set_xticks(positions)
ax.set_xticklabels(list(groups.keys()))
ax.set_ylabel('Score', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

### Trends Over Time / Ordered Axis

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y_mean = np.sin(x)
y_std = 0.2 * np.abs(np.cos(x))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, y_mean, color="#0072B2", linewidth=1.5, label='Model')
ax.fill_between(x, y_mean - 1.96 * y_std, y_mean + 1.96 * y_std,
                color="#0072B2", alpha=0.2, label='95% CI')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Validation Loss', fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

### Relationships: Scatter

```python
from scipy import stats

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(x_vals, y_vals, alpha=0.6, s=30, color="#0072B2", edgecolors='white', linewidth=0.3)

# Regression line with CI
slope, intercept, r, p, se = stats.linregress(x_vals, y_vals)
x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, color="#D55E00", linewidth=1.5)
ax.text(0.05, 0.92, f'r = {r:.2f}, p = {p:.3f}', transform=ax.transAxes, fontsize=9)

ax.set_xlabel('X Variable (units)', fontsize=11)
ax.set_ylabel('Y Variable (units)', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Hexbin for large n (> 1000 points)
# ax.hexbin(x_vals, y_vals, gridsize=30, cmap='viridis')
```

### Categorical Comparisons: Bar Charts

```python
import matplotlib.pyplot as plt
import numpy as np

categories = ['Model A', 'Model B', 'Model C', 'Baseline']
means = [0.82, 0.78, 0.85, 0.65]
errors = [0.03, 0.04, 0.02, 0.01]
colors = ["#56B4E9", "#E69F00", "#009E73", "#999999"]

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(categories, means, yerr=errors, capsize=4,
              color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_ylabel('F1 Score (macro)', fontsize=11)
ax.set_ylim(0, 1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# For many categories → horizontal bar chart
# ax.barh(categories, means, xerr=errors, ...)

# NEVER use 3D bar charts — they distort magnitude perception
```

### Correlation Matrices: Heatmap

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Pearson r')

# Annotate each cell
for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        val = corr_matrix[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=8, color=color)

ax.set_xticks(range(len(feature_names)))
ax.set_yticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(feature_names, fontsize=8)
ax.set_title('Feature Correlation Matrix', fontsize=11, pad=10)
```

### Proportions: Stacked Bar (Never Pie)

```python
# Pie charts make angle/area comparison difficult — use stacked bars
fig, ax = plt.subplots(figsize=(6, 3))
categories = ['Train', 'Validation', 'Test']
sizes = [0.70, 0.15, 0.15]
colors = ["#0072B2", "#56B4E9", "#E69F00"]

left = 0
for size, color, label in zip(sizes, colors, ['70% Train', '15% Val', '15% Test']):
    ax.barh(['Split'], [size], left=left, color=color, label=label, edgecolor='white')
    ax.text(left + size / 2, 0, label, ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    left += size

ax.set_xlim(0, 1)
ax.set_xlabel('Proportion', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
```

---

## Applying Paper Style with matplotlib rcParams

The `apply_paper_style.py` script (in `scripts/`) encapsulates these settings. Use it as follows:

```python
from scripts.apply_paper_style import apply_paper_style, OKABE_ITO_COLORS, save_figure

apply_paper_style(font_size=10)
# All subsequent figures will use publication-quality defaults
```

To apply manually:

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

OKABE_ITO_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]

mpl.rcParams.update({
    # Figure
    "figure.figsize":        (5.5, 4.0),
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.facecolor":     "white",

    # Font
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.labelsize":        10,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.fontsize":       9,

    # Lines and markers
    "lines.linewidth":       1.5,
    "lines.markersize":      5,
    "patch.linewidth":       0.5,

    # Axes
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.prop_cycle":       mpl.cycler(color=OKABE_ITO_COLORS),
    "axes.grid":             True,
    "axes.axisbelow":        True,

    # Grid (subtle)
    "grid.color":            "#DDDDDD",
    "grid.linewidth":        0.6,
    "grid.linestyle":        "--",
    "grid.alpha":            0.7,

    # Ticks
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.major.size":      4,
    "ytick.major.size":      4,
    "xtick.minor.size":      2,
    "ytick.minor.size":      2,
})
```

---

## Multi-Panel Layouts

### 2×2 Layout

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

axes[0, 0].plot([1, 2, 3], [1, 4, 9])
axes[0, 0].set_title('(A) Loss over epochs')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')

# ... populate other panels ...

# Panel labels (A, B, C, D)
for ax, label in zip(axes.flat, ['A', 'B', 'C', 'D']):
    ax.text(-0.15, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

plt.tight_layout(pad=1.5)
plt.savefig('results_overview.pdf', dpi=300, bbox_inches='tight', facecolor='white')
```

### 1×3 Layout (horizontal)

```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.8))
# ...
fig.subplots_adjust(wspace=0.35)
```

### GridSpec (unequal panels)

```python
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(7.0, 5.0))
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax_main  = fig.add_subplot(gs[0, :])   # full-width top row
ax_bl    = fig.add_subplot(gs[1, 0])   # bottom left
ax_bm    = fig.add_subplot(gs[1, 1])   # bottom middle
ax_br    = fig.add_subplot(gs[1, 2])   # bottom right
```

---

## Saving Figures Correctly

```python
import matplotlib.pyplot as plt

# Preferred: vector PDF
fig.savefig('figure1.pdf', dpi=300, bbox_inches='tight', facecolor='white')

# Raster fallback: PNG at 300 dpi
fig.savefig('figure1.png', dpi=300, bbox_inches='tight', facecolor='white')

# TIFF for bio/medical journals (large file)
fig.savefig('figure1.tiff', dpi=600, bbox_inches='tight', facecolor='white')

# SVG for web / editable vector
fig.savefig('figure1.svg', bbox_inches='tight', facecolor='white')
```

Use the helper from `scripts/apply_paper_style.py`:

```python
from scripts.apply_paper_style import save_figure
save_figure(fig, 'figure1', dpi=300, fmt='pdf')   # saves figure1.pdf
save_figure(fig, 'figure1', dpi=300, fmt='png')   # also saves figure1.png
```

---

## Error Bars and Uncertainty Representation

Choose the right uncertainty measure for the claim you are making:

| Measure | Formula | Use When |
|---------|---------|----------|
| ±SD | σ | Describing spread of raw data |
| ±SEM | σ / √n | Estimating precision of the sample mean (NOT data spread) |
| 95% CI | mean ± 1.96 × SEM | Inferential: range plausibly containing the true mean |
| Bootstrap 95% CI | percentile method | When distribution is non-normal or n is small |

```python
import numpy as np
import matplotlib.pyplot as plt

n = 40
data_a = np.random.normal(75, 10, n)
data_b = np.random.normal(65, 12, n)

means = [data_a.mean(), data_b.mean()]
sems  = [data_a.std() / np.sqrt(n), data_b.std() / np.sqrt(n)]
cis   = [1.96 * sem for sem in sems]

fig, ax = plt.subplots(figsize=(4, 4))
ax.bar([0, 1], means, yerr=cis, capsize=5,
       color=["#56B4E9", "#E69F00"], alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Group A', 'Group B'])
ax.set_ylabel('Score (mean ± 95% CI)', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

**Important**: If you use SEM as error bars, state that explicitly in the caption. Never label SEM bars as "error bars" without specifying the measure.

---

## Statistical Significance Annotations

### Standard Notation

| Symbol | p-value |
|--------|---------|
| ns | p > 0.05 |
| * | p ≤ 0.05 |
| ** | p ≤ 0.01 |
| *** | p ≤ 0.001 |
| **** | p ≤ 0.0001 |

### Manual Bracket Annotation

```python
def add_significance_bracket(ax, x1, x2, y, h, text, fontsize=10, linewidth=1.0):
    """
    Draw a significance bracket between positions x1 and x2.

    Parameters
    ----------
    ax      : matplotlib Axes
    x1, x2  : x-positions of the two groups being compared
    y       : y-position for the bracket (should be above the bars/data)
    h       : height of the bracket arms
    text    : annotation string, e.g. '**' or 'ns'
    """
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=linewidth, color='black')
    ax.text((x1 + x2) / 2, y + h, text,
            ha='center', va='bottom', fontsize=fontsize)

# Example usage
add_significance_bracket(ax, x1=0, x2=1, y=90, h=2, text='**')
```

### Using statannotations (automated)

```python
# pip install statannotations
from statannotations.Annotator import Annotator
import seaborn as sns

ax = sns.boxplot(data=df, x='group', y='score')
pairs = [('Control', 'Treatment A'), ('Control', 'Treatment B')]
annotator = Annotator(ax, pairs, data=df, x='group', y='score')
annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
annotator.apply_and_annotate()
```

---

## Common Mistakes to Avoid

| Mistake | Why It's Wrong | Fix |
|---------|---------------|-----|
| Chartjunk (gridlines, 3D effects, shadows) | Distracts from data | Use minimal style; remove top/right spines |
| Truncated y-axis (not starting at zero for bar charts) | Exaggerates differences | Start bars at zero; use line plots if showing change |
| Missing axis labels | Reader cannot interpret units | Always label both axes with units in parentheses |
| Unlabeled axes | Same as above | All axes must have labels before saving |
| Pie charts | Angle/area comparisons are inaccurate | Use bar charts or stacked bars |
| Using red/green alone to encode categories | Invisible to deuteranopes | Use Okabe-Ito palette |
| Mismatched scales in multi-panel figures | Implies false relative differences | Use `sharex` / `sharey` when scales should match |
| Reporting SEM without saying so | Misleading — SEM is much smaller than SD | Always state "mean ± SEM" or "mean ± SD" in caption |
| Font below 8pt | Unreadable in print | Enforce minimum via rcParams |
| Saving at 72 dpi | Pixelated in print | Always use ≥ 300 dpi or vector format |
| No units | Ambiguous axes | "Accuracy" → "Accuracy (%)" or "Normalized Score" |

---

## Figure Caption Checklist

Every figure must have a caption that includes:

- [ ] What is shown (not just "Figure 1")
- [ ] What each panel shows (A, B, C, D)
- [ ] Definition of error bars (±SD, ±SEM, 95% CI, and n)
- [ ] Sample size per group
- [ ] Statistical test used for any significance annotation
- [ ] Color encoding explained if not obvious from axis labels

Example:

> **Figure 2. Model comparison across five datasets.** (A) F1 score (macro) for each model; error bars represent 95% CI across 5-fold cross-validation (n = 5 folds). (B) Learning curves showing training and validation F1 as a function of training set size; shaded region = ±SD. *, p ≤ 0.05; **, p ≤ 0.01 by paired t-test across folds with Bonferroni correction.

---

## Resources

### Scripts Directory

- **apply_paper_style.py**: Import `apply_paper_style()`, `OKABE_ITO_COLORS`, `save_figure()`, and `add_significance_annotation()` to instantly apply publication-quality defaults to any matplotlib figure.
