# Aspect Ratios

The Outline Agent's prompt enumerates exactly 12 allowed aspect ratios (App.
F.1, page 41). Each one maps to a specific figure size in inches at 300 DPI.
The plotting agent MUST use these — anything else fails the outline schema
validator.

| Ratio | Inches (W × H) | Pixels @ 300 DPI | Use case |
|---|---|---|---|
| `1:1`  | 3.4 × 3.4    | 1020 × 1020 | Square radar chart, 1×1 ablation grid |
| `1:4`  | 1.8 × 7.2    | 540 × 2160  | Tall vertical strip (rare; e.g., per-class bars) |
| `2:3`  | 3.4 × 5.1    | 1020 × 1530 | Portrait single-column figure |
| `3:2`  | 5.1 × 3.4    | 1530 × 1020 | Landscape single-column figure |
| `3:4`  | 3.0 × 4.0    | 900 × 1200  | Tall stacked subplots |
| `4:1`  | 7.0 × 1.75   | 2100 × 525  | Banner / timeline / strip-plot |
| `4:3`  | 4.0 × 3.0    | 1200 × 900  | Standard single-column rectangle |
| `4:5`  | 3.2 × 4.0    | 960 × 1200  | Slight portrait, 5-row heatmap |
| `5:4`  | 4.5 × 3.6    | 1350 × 1080 | Wide single-column |
| `9:16` | 2.8 × 4.97   | 840 × 1491  | Mobile / tall portrait |
| `16:9` | 5.5 × 3.09   | 1650 × 927  | Cross-column wide chart |
| `21:9` | 7.0 × 3.0    | 2100 × 900  | Full-page-width banner |

## Width budget

The widths are chosen to slot into the most common LaTeX layouts:

- **2-column conference template** (CVPR, ICCV, NeurIPS):
  single-column = 3.3in, double-column = 7.0in. Use widths ≤ 3.5in for
  `\begin{figure}` and ≤ 7.0in for `\begin{figure*}`.
- **1-column template** (ICLR, plain article): page text width is ~6.5in.

The `fig_size_for(ratio)` helper in `chart-patterns.md` returns the
matching `(width, height)` tuple. The bundled `render_matplotlib.py` script
uses the same table.

## DPI

Always 300. Conference templates reject ≤150 DPI raster figures.

## Strict mapping in JSON spec

The `render_matplotlib.py` and `render_diagram.py` helpers accept a JSON
spec with `"aspect_ratio"` set to one of the 12 strings above. They look
up the size, set `figsize`, set `dpi=300`, and call `tight_layout()` +
`bbox_inches='tight'` automatically.
