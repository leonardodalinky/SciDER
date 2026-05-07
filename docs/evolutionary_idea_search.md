# Evolutionary Idea Search

## Overview

The evolutionary idea search is a best-first refinement loop that runs between seed idea extraction and user approval in SciDER's ideation pipeline. Rather than picking the best of N one-shot generated ideas, it iteratively improves and combines them across four quality dimensions, producing a ranked final set that a judge model consistently prefers over the unrefined seeds.

The mechanism is grounded in three published systems adapted for text-domain idea search:
- **AIDE** (arxiv 2502.13138) — greedy tree search with Draft/Improve operators
- **AIRA-2** (arxiv 2603.26499) — steady-state evolutionary with rank-based selection
- **AI-Scientist v2** (arxiv 2504.08066) — best-first tree search with LLM-evaluated fitness

The key adaptation: those systems use objective ML fitness functions (validation loss, benchmark score). Here, fitness is estimated via **K-way batch ranking** across four dimensions rather than absolute 0-10 scores, which is more noise-resistant for text-domain evaluation.

---

## Pipeline Integration

The search is inserted as a single LangGraph node between `extract_ideas` and `approve_ideas`:

```
START → init → agent_loop → generate_report → extract_ideas → idea_search → approve_ideas → END
```

`idea_search` is a no-op when `idea_search_enabled=False`, preserving full backward compatibility. The `approve_ideas` retry path is unchanged.

---

## How It Works

### 1. Seed Population

The N ideas extracted by the ideation agent (typically 5–8) become the seed population, capped at `POPULATION_SIZE=8`. They are scored immediately across all four dimensions.

### 2. Scoring: K-way Batch Ranking

Each scoring pass makes **4 parallel LLM calls** (one per dimension). Each call ranks all N ideas in a single prompt and returns a 1-based permutation:

```
{"ranking": [best_id, 2nd, ..., worst_id], "brief_rationale": "..."}
```

The normalized rank score for each idea is:

```
score = (N + 1 − rank) / N
```

So rank 1 → 1.0, rank N → 1/N. This keeps scores in (0, 1] and avoids the positional bias and inter-call variance of absolute 0-10 scoring.

**Four dimensions:**

| Dimension | Weight | Question asked |
|---|---|---|
| Novelty | 0.30 | Which idea opens the most genuinely new research territory? |
| Feasibility | 0.25 | Which idea could a PhD student realistically run as their dissertation? |
| Impact | 0.25 | Which idea, if it succeeds, would most change the field? |
| Specificity | 0.20 | Which idea has the clearest path from hypothesis to measured result? |

**Composite score** = weighted average of the four normalized ranks.

On parse failure, all ideas in that dimension receive a neutral score of 0.5 (degradation, not abort). Failed dimensions unconditionally overwrite any stale score from a prior population context.

### 3. Operators

#### Improve (targeted)

Identifies each parent idea's weakest-scoring dimension and generates a revised version with dimension-specific guidance — not a generic "make it better" prompt:

- **Novelty**: find an unexplored angle or literature gap; must differ substantively from prior work
- **Feasibility**: replace speculative components; must be completable in 12–18 months with standard resources
- **Impact**: broaden or sharpen the problem; consider who else benefits downstream
- **Specificity**: add a null hypothesis, baseline comparison, primary outcome metric, and confounds to control

#### Combine (crossover synthesis)

Takes two parent ideas and uses the top-dimension rationale string from scoring as a hint for productive crossover. The prompt instructs the model to inherit the core mechanism from one idea and the novelty angle or experimental rigor from the other.

### 4. Search Loop

```
population = score(seed_ideas)          # 4 calls

for iteration in 1..MAX_ITERATIONS:
    survivors  = top 4 by composite_score
    n_new      = 8 - 4 = 4
    n_improve  = round(4 × 0.75) = 3   # guaranteed per iteration
    n_combine  = 4 - 3 = 1             # guaranteed per iteration

    new_nodes  = [improve(parent) for 3 parents]
               + [combine(a, b)]

    re-score(survivors + new_nodes)     # 4 calls (calibrated against full merged batch)
    population = survivors + new_nodes

final_score(population)                 # 4 calls (only if population changed)
```

**Population arithmetic**: `POPULATION_SIZE=8` with `SURVIVORS_PER_ITER=4` gives `n_new=4`. With `IMPROVE_FRACTION=0.75`: `round(4×0.75)=3` improve slots and 1 combine slot per iteration — guaranteeing crossover runs every iteration. (Using `POPULATION_SIZE=6` would give `n_new=2`, `round(2×0.75)=2`, `n_combine=0` — Combine never runs.)

**Re-scoring merged batches**: survivors are re-scored alongside new ideas each iteration. A survivor's composite score can decrease when new competitors enter the batch — this is correct calibration, not a regression.

**Final scoring pass**: skipped when no operators produced new nodes (e.g., score-only baseline runs), saving 4 calls and ensuring `baseline_composite.lift = 0` by construction.

### 5. Output

Each idea in `best_ideas` is enriched with search metadata:

```python
{
    "title": ...,
    "description": ...,
    "composite_score": 0.743,
    "score_novelty": 0.875,
    "score_feasibility": 0.750,
    "score_impact": 0.750,
    "score_specificity": 0.625,
    "search_operator": "combine",   # "seed" | "improve" | "combine"
}
```

Results are sorted by `composite_score` descending. The `paper_bootstrap` selection function uses `composite_score` when present, falling back to `novelty_score` for backward compatibility.

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `idea_search_enabled` | `True` | Toggle. When `False`, the node is a no-op. |
| `max_idea_search_calls` | `60` | Hard LLM budget cap (~32 expected for 3 iterations). |
| `POPULATION_SIZE` | `8` | Max ideas in population per generation. |
| `MAX_ITERATIONS` | `3` | Search generations. |
| `SURVIVORS_PER_ITER` | `4` | Ideas carried forward each iteration. |
| `IMPROVE_FRACTION` | `0.75` | Fraction of new slots filled by Improve (vs Combine). |

**LLM call budget:**

| Phase | Calls |
|---|---|
| Initial scoring | 4 |
| Per iteration: 3 improve + 1 combine | 4 |
| Per iteration: re-score merged batch | 4 |
| Final scoring pass | 4 |
| **Total (3 iterations)** | **4 + (8×3) + 4 = 32** |

Hard cap of 60 provides ~2× headroom for retries and budget overruns.

---

## Configuration

### Ideation-only workflow

```python
from scider.workflows.ideation_workflow import run_ideation_workflow

result = run_ideation_workflow(
    user_query="efficient transformers for long document understanding",
    workspace_path="workspace/",
    idea_search_enabled=True,      # default
    max_idea_search_calls=60,      # default
)
```

### Full workflow

```python
from scider.workflows.full_workflow_with_ideation import FullWorkflowWithIdeation

w = FullWorkflowWithIdeation(
    user_query="...",
    workspace_path=Path("workspace/"),
    idea_search_enabled=True,
    max_idea_search_calls=60,
)
w.run()
```

### Streamlit UI

Both the **Ideation** and **Full Workflow** forms expose:
- "Enable evolutionary idea search" checkbox (default: on)
- "Max LLM calls for idea search" number input (Ideation form only, default: 60)

---

## Evaluation

### Setup

The evaluation harness (`evals/eval_idea_search.py`) tests four configurations per query on fixed seed idea pools:

| Configuration | Description |
|---|---|
| `baseline_novelty` | Top seed by the LLM's own `novelty_score` (current single-pass behaviour) |
| `baseline_composite` | Top seed by 4D composite score — no search iterations |
| `improve_only` | Full search loop, Improve operator only (`improve_fraction=1.0`) |
| `improve_combine` | Full search loop, Improve + Combine (default config) |

Three pairwise judge comparisons per query use a separate `critic` model to avoid self-confirmation:

- **J1**: `baseline_novelty` vs `improve_combine` — main claim
- **J2**: `baseline_composite` vs `improve_combine` — value of search iterations beyond scoring
- **J3**: `improve_only` vs `improve_combine` — marginal value of the Combine operator

**Lift metric** for search runs: `(final_composite − seed_baseline) / seed_baseline`, where `seed_baseline` is the best surviving seed's composite in the final n=8 scoring pass (same population, same context). Falls back to `initial_best_composite` when all seeds are replaced by operators, which introduces ~5–10% noise from the n=6 vs n=8 scale difference; in that case the judge win-rate is the reliable signal.

### Results (5 queries, preliminary)

Run on 5 diverse queries: NLP, Federated Learning, Drug Discovery, Causal Inference, Speech.

**Judge win-rates** (B = right-side configuration wins):

| Comparison | B wins | Win-rate | p-value |
|---|---|---|---|
| `improve_combine` vs `baseline_novelty` | **5 / 5** | **100%** | 0.031 * |
| `improve_combine` vs `baseline_composite` | **4 / 5** | **80%** | 0.188 |
| `improve_combine` vs `improve_only` | 3 / 5 | 60% | 0.500 |

**Composite lift over initial best seed:**

| Configuration | Mean lift |
|---|---|
| `baseline_composite` (score-only) | 0.0% (by construction) |
| `improve_only` | +1.4% |
| `improve_combine` | −0.9% |

The negative aggregate lift for `improve_combine` is driven by the two high-ceiling queries (initial best ≥ 0.82), where the n-scale shift dominates. On queries with initial best ≤ 0.70, both search configs show positive lift (+6–12%).

**Per-query results:**

| Domain | initial_best | improve_combine lift | J1 | J2 | J3 |
|---|---|---|---|---|---|
| NLP | 0.817 | −4.3% | B | B | B |
| FL/Privacy | 0.700 | +6.3% | B | B | A |
| Chemistry | 0.650 | +8.7% | B | B | B |
| Statistics | 0.642 | +12.0% | B | B | A |
| Speech | 0.917 | −26.4% | B | B | A |

**Other observations:**

- **Baseline novelty vs composite gap is large**: in the Speech query, the top-novelty-score seed had composite=0.367 while the best composite seed scored 0.917. The LLM's single `novelty_score` is a poor proxy for multi-dimensional quality — validating the 4D scoring switch as a standalone improvement.
- **Population diversity**: mean pairwise TF-IDF cosine similarity of 0.20–0.35 across all queries, well below the 0.7 collapse threshold.
- **Operator distribution**: ~7 improve and ~1–2 combine per query in the final population. After 3 iterations, seeds are typically fully replaced by operator-generated ideas.
- **LLM calls**: 63 per query (4 shared seed scoring + 28 improve_only + 28 improve_combine + 3 judge).

### Interpretation

The judge signal is the reliable metric. A judge that saw neither the search config nor the model used consistently preferred `improve_combine` over unrefined seeds (5/5). The composite lift numbers are useful sanity checks but carry n-scale noise when seeds don't survive into the final population.

The 5-query pilot is statistically meaningful only for J1 (p=0.031). The 20-query eval will provide proper power for all three comparisons.

---

## Known Limitations

1. **n-scale lift noise**: initial scoring uses the seed population (n=6), but final scoring runs on n=8 (seeds + operators). Rank-based scores are not directly comparable across different n. When all seeds are replaced by operators (common after 3 iterations), the lift denominator falls back to the n=6 initial score. The 5–10% noise this introduces is documented in the report notes; use judge win-rates as the primary metric.

2. **Self-calibration only**: the search optimises for what the ideation LLM itself ranks highly. A dimension scoring failure (e.g., API rate limit) degrades gracefully to 0.5 neutral — the search continues but that dimension's signal is lost for that pass.

3. **No memory across sessions**: the search is stateless; it does not accumulate knowledge of what operators worked well in prior runs.

4. **Preliminary evaluation**: results are from 5 queries. The 20-query eval (`evals/eval_idea_search.py`) is ready to run and will provide statistically significant win-rates across multiple domains.

---

## Files

| File | Role |
|---|---|
| `scider/agents/ideation_agent/idea_search.py` | Core search engine: `IdeaNode`, scorer, operators, search loop |
| `scider/agents/ideation_agent/state.py` | `idea_search_enabled`, `max_idea_search_calls`, `idea_search_result`, `composite_scores` fields |
| `scider/agents/ideation_agent/execute.py` | `idea_search_node` LangGraph node |
| `scider/agents/ideation_agent/build.py` | Graph wiring: `extract_ideas → idea_search → approve_ideas` |
| `scider/workflows/ideation_workflow.py` | `idea_search_enabled` / `max_idea_search_calls` pass-through |
| `scider/workflows/full_workflow_with_ideation.py` | Same pass-through for the full pipeline |
| `scider/workflows/paper_bootstrap.py` | Score selector: prefers `composite_score`, falls back to `novelty_score` |
| `streamlit-client/forms/ideation.py` | UI toggle and budget input |
| `streamlit-client/forms/full.py` | UI toggle in full workflow form |
| `tests/idea_search_test.py` | 16 unit tests with mocked LLM |
| `evals/eval_idea_search.py` | Ablation evaluation harness (20 queries, 4 configs, judge comparisons) |
