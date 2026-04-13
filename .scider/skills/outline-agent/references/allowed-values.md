# Allowed enumerated values

Source: arXiv:2604.05018, App. F.1 Outline Agent prompt.

These are the only values the validator will accept for each enum field.
Anything else will fail `validate_outline.py`.

## `plot_type`

| Value | Use for |
|---|---|
| `"plot"` | Statistical/quantitative charts (line, bar, radar, scatter, box, histogram, heatmap, etc.). The `objective` field MUST also name the specific chart type. |
| `"diagram"` | Conceptual / architectural diagrams (block diagram, flowchart, system overview). Rendered with graphviz or matplotlib patches. |

## `data_source`

| Value | Meaning |
|---|---|
| `"idea.md"` | The figure visualizes a concept or architecture from the idea (typically diagrams). |
| `"experimental_log.md"` | The figure plots numeric values from the experimental log. |
| `"both"` | The figure combines conceptual context with empirical data. |

## `aspect_ratio`

Exactly one of these 12 strings:

```
1:1   1:4   2:3   3:2   3:4   4:1   4:3   4:5   5:4   9:16   16:9   21:9
```

Common conventions:

| Use | Ratio |
|---|---|
| Square ablation grid | `1:1` |
| Single column figure (2-col layout) | `4:3` or `5:4` |
| Wide cross-column figure (2-col layout) | `16:9` or `21:9` |
| Vertical mobile / tall comparison | `9:16` or `2:3` |
| Teaser banner | `21:9` |
| Tall stacked subplots | `3:4` or `4:5` |

The Plotting Agent's `render_matplotlib.py` translates the ratio into pixel
dimensions at 300 DPI.

## `figure_id` rules

- Snake_case: `^[a-z0-9_]+$`
- Convention: prefix with `fig_` (e.g., `fig_main_results`,
  `fig_ablation_temperature_sensitivity`).
- MUST be semantically meaningful (no `fig_1`, `fig_a`, `chart`).
- MUST NOT contain the word `figure` as a substring after the `fig_` prefix.

## Citation hint format

Two acceptable forms (per the prompt):

| You know | Use |
|---|---|
| Exact author and title | `"Author (Exact Paper Title)"` — e.g., `"Vaswani et al. (Attention Is All You Need)"` |
| You don't know either | `"research paper or technical report introducing '[Exact Model/Dataset/Metric Name]'"` — e.g., `"research paper or technical report introducing 'AdamW optimizer'"` |

**Never guess authors.** The Literature Review Agent will use these hints as
search queries — wrong authors lead to wrong citations.
