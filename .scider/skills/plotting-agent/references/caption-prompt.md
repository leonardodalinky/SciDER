# Plotting Agent — Caption Generation prompt

**Source: arXiv:2604.05018, Appendix F.1, page 45 (verbatim).**

The paper uses PaperBanana (Zhu et al., 2026) for the entire visual generation
pipeline; it then appends a single caption regeneration step at the end using
the prompt below. We reproduce only this final caption step verbatim because
it is the part the paper authors actually wrote (PaperBanana's internal
prompts are external and not reproduced here).

---

```
Input Data
  - Task Type:               {task_name}
  - Contextual Section:      {raw_content}
  - Overall Figure Intent:   {description}
  - Detailed Figure Description: {figure_desc}

Please provide the final caption for this figure based on the system
instructions.

Requirements

  - The caption should be concise and informative, and can be directly used
    as a caption for academic papers.
  - The caption MUST NOT contain a "Figure X:" or "Caption X:" prefix, as
    the latex template will add it automatically.
  - The caption MUST NOT contain any markdown formatting (like bold,
    italics, etc), it should be plain text.

Respond with the plain text caption only.
```

---

## Field substitution guide

| Field | Source |
|---|---|
| `{task_name}` | The section the figure belongs to. Look it up in `outline.json` — find which section's `content_bullets` (or surrounding text in `drafts/paper.tex` if Step 4 has run) reference this `figure_id`. Common values: `"Methodology"`, `"Experiments"`, `"Ablation Studies"`. |
| `{raw_content}` | The text of the section the figure appears in. If Step 4 has run, paste the relevant paragraph from `drafts/paper.tex`. If not, paste the joined `content_bullets` from the corresponding subsection in `outline.json`. |
| `{description}` | The `objective` field from the figure spec in `outline.json`. |
| `{figure_desc}` | A 1-2 sentence factual description of what the rendered PNG actually contains. If your host has vision, generate this by inspecting the rendered PNG. If not, derive it from the matplotlib script you wrote (e.g., "Grouped bar chart comparing methods A, B, C across metrics M1 and M2."). |

## Output

Plain text. One caption. No prefix, no markdown, no quotes around it.
Save the result into `workspace/figures/captions.json` keyed by `figure_id`:

```json
{
  "fig_main_results": "Comparison of three temporal-attention variants on the Ref-AVS Seen split. Bars show Jaccard index; error bars are 95% confidence intervals over five seeds.",
  "fig_framework_overview": "Overview of the proposed pipeline. Raw video frames flow into the frozen SAM encoder; aligned audio cues are projected through the temporal modality fusion layer before being injected into the mask decoder."
}
```
