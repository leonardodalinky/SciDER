# Plotting Pipeline (per-figure loop)

Source: arXiv:2604.05018, §4 Step 2 ("Plotting Agent"), App. B (cost: ~20-30
LLM calls including few-shot retrieval, visual planning, image generation,
VLM-guided critique-and-redraw cycles, and context-aware captioning).

The paper outsources visual generation to PaperBanana (Zhu et al., 2026)
internally. We can't redistribute PaperBanana, so this skill expresses the
*loop structure* in terms a host coding agent can execute with its own LLM
and matplotlib.

## The 5-stage per-figure loop

```
                ┌─────────────────────────────────────┐
                │  outline.json  →  figure spec        │
                │  { figure_id, plot_type, objective,  │
                │    aspect_ratio, data_source }       │
                └─────────────────┬───────────────────┘
                                  │
                                  ▼
       ┌──────────────────────────────────────────────────┐
       │  STAGE 1 — Few-shot retrieval                     │
       │  Pick the matching pattern from chart-patterns.md │
       │  (plot) or diagram-patterns.md (diagram). Identify│
       │  variables you'll need from the data sources.     │
       └─────────────────┬────────────────────────────────┘
                         │
                         ▼
       ┌──────────────────────────────────────────────────┐
       │  STAGE 2 — Visual planning                        │
       │  Sketch (in your head / in JSON) the chart layout:│
       │  axes, series, colors, legend placement, title.   │
       │  Resolve aspect_ratio → pixel dimensions          │
       │  via aspect-ratios.md.                            │
       └─────────────────┬────────────────────────────────┘
                         │
                         ▼
       ┌──────────────────────────────────────────────────┐
       │  STAGE 3 — Image generation                       │
       │  Write matplotlib code. Apply the global style    │
       │  from chart-patterns.md. Save to                  │
       │  workspace/figures/<figure_id>.png at 300 DPI.    │
       │  Run via your Bash tool, OR call                  │
       │  scripts/render_matplotlib.py with a spec JSON.   │
       └─────────────────┬────────────────────────────────┘
                         │
                         ▼
       ┌──────────────────────────────────────────────────┐
       │  STAGE 4 — VLM critique loop  (skip if no vision) │
       │  for iter in 1..3:                                │
       │      load PNG as multimodal input to your LLM     │
       │      critique against `objective` field           │
       │      if no issues: break                          │
       │      else: regenerate matplotlib code, re-render  │
       └─────────────────┬────────────────────────────────┘
                         │
                         ▼
       ┌──────────────────────────────────────────────────┐
       │  STAGE 5 — Caption                                │
       │  Use the verbatim caption-prompt.md.              │
       │  Save into figures/captions.json.                 │
       └──────────────────────────────────────────────────┘
```

## What to look for in the VLM critique

When you (the host agent) inspect a rendered figure with vision, score it
against this checklist (derived from the paper's plotting failure modes and
the academic-paper skill's review-guide.md):

| Issue | Signal |
|---|---|
| Mislabeled or missing axes | Numeric tick marks with no axis label |
| Illegible text | Font size <6pt; tick labels overlapping |
| Color clash | Two adjacent series indistinguishable in print |
| Missing legend | More than one series, no legend |
| Overlapping legend | Legend covers a data point |
| Misleading scaling | Bar chart Y-axis doesn't start at 0; log scale unannounced |
| Cropped content | Title or labels cut off at edge |
| Missing units | Numeric axes without units in label |
| Decorative noise | Drop shadows, gradients, 3D effects |
| Unsupported claim | The plot shows a trend not present in the source data |

If any of these fire, regenerate the matplotlib script with the fix and
re-render. Cap at 3 iterations.

## Wall-time budget

The paper allocates ~20-30 LLM calls per Plotting Agent invocation. With
~5-10 figures per typical paper, that's ~3 calls per figure: roughly
1 (planning) + 1 (initial generation) + 1 (critique pass) + 0-1 (redraw) +
1 (caption). Stay within this budget by using deterministic helpers for the
mechanical parts (`render_matplotlib.py` does pixel sizing, style, save) and
reserving LLM calls for actual visual judgment.
