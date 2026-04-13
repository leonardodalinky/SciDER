# Chart Patterns

Matplotlib recipes for academic-paper figures. Adapted from
`~/.all-skills/academic-paper/references/chart-patterns.md` and tuned for the
PaperOrchestra plotting agent's aspect-ratio constraints.

## Global style config

Apply this at the top of every plotting script.

```python
import matplotlib
matplotlib.use('Agg')             # headless
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         8,
    'axes.titlesize':    9,
    'axes.titleweight':  'bold',
    'axes.labelsize':    8,
    'axes.linewidth':    0.6,
    'legend.fontsize':   7,
    'legend.framealpha': 0.95,
    'legend.edgecolor':  '#cccccc',
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.08,
    'grid.alpha':        0.15,
    'grid.linewidth':    0.5,
    'lines.linewidth':   1.3,
})

# Muted academic palette (print-safe)
BLUE   = '#2060cc'
RED    = '#cc3030'
GREEN  = '#208040'
ORANGE = '#cc7020'
PURPLE = '#8040cc'
GOLD   = '#b08020'
GRAY   = '#666666'
PALETTE = [BLUE, RED, GREEN, ORANGE, PURPLE, GOLD, GRAY]
```

## Per-pattern recipes

### Line chart (training curves, scaling laws)

```python
fig, ax = plt.subplots(figsize=fig_size_for("16:9"))
for i, (name, ys) in enumerate(series.items()):
    ax.plot(xs, ys, color=PALETTE[i], label=name)
ax.set_xlabel('Training step')
ax.set_ylabel('Validation loss')
ax.legend(loc='upper right', frameon=True)
ax.grid(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(out_path)
plt.close()
```

For noisy time-series, smooth with a rolling mean before plotting:
```python
def rolling(y, w=7):
    return np.convolve(y, np.ones(w)/w, mode='valid')
```

### Grouped bar chart (method comparison across metrics)

```python
methods = ['Baseline', 'Ours-S', 'Ours-L']
metrics = ['Acc', 'F1', 'AUC']
data = np.array([[78, 79, 0.83],
                 [82, 84, 0.87],
                 [85, 87, 0.91]])  # rows=methods, cols=metrics

fig, ax = plt.subplots(figsize=fig_size_for("5:4"))
x = np.arange(len(metrics))
w = 0.25
for i, m in enumerate(methods):
    ax.bar(x + (i-1)*w, data[i], w, color=PALETTE[i], label=m,
           edgecolor='white', linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Score')
ax.legend(loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2)
fig.tight_layout()
fig.savefig(out_path)
plt.close()
```

### Radar chart (multi-axis SOTA comparison — paper's Fig 2 pattern)

```python
labels = ['Originality', 'Quality', 'Clarity', 'Significance', 'Soundness']
ours    = [3.0, 3.4, 3.6, 3.0, 3.4]
baseline = [2.5, 2.7, 3.0, 2.3, 2.8]

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]
ours    = ours + ours[:1]
baseline = baseline + baseline[:1]

fig, ax = plt.subplots(figsize=fig_size_for("1:1"), subplot_kw=dict(polar=True))
ax.plot(angles, ours, color=BLUE, linewidth=1.5, label='Ours')
ax.fill(angles, ours, color=BLUE, alpha=0.15)
ax.plot(angles, baseline, color=RED, linewidth=1.5, label='Baseline')
ax.fill(angles, baseline, color=RED, alpha=0.10)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=7)
ax.set_yticks([1, 2, 3, 4])
ax.set_ylim(0, 4)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
fig.tight_layout()
fig.savefig(out_path)
plt.close()
```

### Stacked bar / win-rate plot (paper's Fig 2/3 SxS pattern)

```python
methods = ['Single Agent', 'AI-Sci-v2', 'PaperOrchestra']
wins = [33, 22, 65]
ties = [22, 18, 25]
losses = [45, 60, 10]

fig, ax = plt.subplots(figsize=fig_size_for("4:3"))
x = np.arange(len(methods))
ax.bar(x, losses, color=RED, label='Baseline win', edgecolor='white', linewidth=0.4)
ax.bar(x, ties, bottom=losses, color=GRAY, label='Tie', edgecolor='white', linewidth=0.4)
ax.bar(x, wins, bottom=np.array(losses)+np.array(ties),
       color=BLUE, label='Our win', edgecolor='white', linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel('Percentage (%)')
ax.set_ylim(0, 100)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(out_path)
plt.close()
```

### Heatmap (ablation grid)

```python
import matplotlib.colors as mcolors
data = np.random.rand(5, 6) * 0.4 + 0.5  # placeholder
xticks = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
yticks = ['A=0.1', 'A=0.2', 'A=0.5', 'A=1.0', 'A=2.0']

fig, ax = plt.subplots(figsize=fig_size_for("4:3"))
im = ax.imshow(data, cmap='Blues', vmin=0.4, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(xticks)))
ax.set_xticklabels(xticks)
ax.set_yticks(range(len(yticks)))
ax.set_yticklabels(yticks)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                color='white' if data[i,j] > 0.7 else 'black', fontsize=6)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
fig.tight_layout()
fig.savefig(out_path)
plt.close()
```

### Multi-panel (side-by-side ablation/case study)

```python
fig, axes = plt.subplots(1, 3, figsize=fig_size_for("21:9"))
for ax, (label, data) in zip(axes, panels.items()):
    ax.plot(data['x'], data['y'], color=BLUE)
    ax.set_title(label, fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(out_path)
plt.close()
```

## Aspect ratio helper

```python
def fig_size_for(ratio: str) -> tuple[float, float]:
    """Return (width_inches, height_inches) at a fixed width target.
    Width target: 5.5in for single-column, 7.0in for full-page wide."""
    w_to_h = {
        "1:1":  (3.4, 3.4),
        "1:4":  (1.8, 7.2),
        "2:3":  (3.4, 5.1),
        "3:2":  (5.1, 3.4),
        "3:4":  (3.0, 4.0),
        "4:1":  (7.0, 1.75),
        "4:3":  (4.0, 3.0),
        "4:5":  (3.2, 4.0),
        "5:4":  (4.5, 3.6),
        "9:16": (2.8, 4.97),
        "16:9": (5.5, 3.09),
        "21:9": (7.0, 3.0),
    }
    return w_to_h[ratio]
```

## Anti-patterns

- Never use 3D charts, pie charts, or decorative visuals.
- Never use default matplotlib colors (too saturated for print).
- Never skip axis labels or units.
- Never place legend outside the axes area without `bbox_to_anchor` (causes overflow).
- Always `plt.close()` after `savefig()` to prevent memory leaks across many figures.
- Never put a `Figure N:` text into the chart itself — captions handle that.
