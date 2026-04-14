# Outline JSON Schema (prose)

The Outline Agent emits a single JSON object with three top-level keys:
`plotting_plan`, `intro_related_work_plan`, `section_plan`. Source:
arXiv:2604.05018, App. F.1, pp. 43–44.

The machine-readable JSON Schema is in `outline_schema.json` and is enforced
by `scripts/validate_outline.py`. This document is the human-readable view.

## Top-level structure

```json
{
  "plotting_plan": [ <FigureSpec>, ... ],
  "intro_related_work_plan": <IntroRelWorkPlan>,
  "section_plan": [ <SectionSpec>, ... ]
}
```

## FigureSpec

```json
{
  "figure_id":     "fig_framework_overview",            // snake_case, no "Figure"
  "title":         "Framework Overview",
  "plot_type":     "diagram",                           // "plot" or "diagram"
  "data_source":   "idea.md",                           // "idea.md" | "experimental_log.md" | "both"
  "objective":     "Visual summary (Block Diagram) of the proposed pipeline.",
  "aspect_ratio":  "16:9"                               // see allowed-values.md
}
```

If `plot_type` is `"plot"`, the `objective` MUST name the specific chart type
(e.g., "Visual summary (Radar Chart) demonstrating that our method achieves
SOTA balance across 5 metrics.").

## IntroRelWorkPlan

```json
{
  "introduction_strategy": {
    "hook_hypothesis":        "Video-LLMs are currently the dominant paradigm for short clips.",
    "problem_gap_hypothesis": "Context window limits prevent scaling to >5s videos efficiently.",
    "search_directions": [
      "Find highly cited papers establishing the real-world impact of context limits in video generation",
      "Search for published 'long-context video generation' surveys",
      "Identify foundational papers establishing causal video generation"
    ]
  },
  "related_work_strategy": {
    "overview": "Investigate three specific paradigms to build a graph proving the necessity of our Sliding-Window approach.",
    "subsections": [
      {
        "subsection_title":          "2.1 Autoregressive Video Generation",
        "methodology_cluster":       "Discrete Tokenization & Transformers",
        "sota_investigation_mission": "Identify the current SOTA autoregressive models from 2024-2025. Determine their maximum stable generation length.",
        "limitation_hypothesis":     "These models suffer from 'drift' or 'error propagation' because they lack bidirectional context.",
        "limitation_search_queries": [
          "Autoregressive video generation error propagation metrics",
          "Causal masking limitations in temporal video transformers"
        ],
        "bridge_to_our_method": "Our method introduces bidirectional blocks to fix the hypothesized drift issue."
      }
    ]
  }
}
```

## SectionSpec

```json
{
  "section_title": "3. Methodology",
  "subsections": [
    {
      "subsection_title": "3.1 Temporal-Aware Attention Mechanism",
      "content_bullets": [
        "Define the query-key matching logic",
        "Explain the masking strategy"
      ],
      "citation_hints": [
        "Vaswani et al. (Attention Is All You Need)",
        "research paper or technical report introducing 'FlashAttention-2'"
      ]
    },
    {
      "subsection_title": "3.2 Optimization Objective",
      "content_bullets": [
        "Detail the loss function",
        "Discuss regularization terms"
      ],
      "citation_hints": []
    }
  ]
}
```

## Validation rules (enforced by `validate_outline.py`)

1. The top-level object MUST contain exactly the keys `plotting_plan`,
   `intro_related_work_plan`, `section_plan` (extras allowed but reported).
2. `plotting_plan` is a non-empty array.
3. Each FigureSpec has all six fields with correct enum values.
4. `figure_id` matches `^[a-z0-9_]+$` and does not contain `figure` as a
   substring after `fig_`.
5. `intro_related_work_plan.introduction_strategy` and `.related_work_strategy`
   both present.
6. `related_work_strategy.subsections` has 2–4 entries (the paper's
   "2-4 distinct methodology clusters" rule).
7. `section_plan` is a non-empty array. Each SectionSpec has `section_title`
   and `subsections`.
8. **Hierarchy rule**: if a section has any subsection, it must have ≥2
   (no orphans). The validator emits a WARNING for sections with exactly 1
   subsection.
9. `citation_hints` is always an array (may be empty).
