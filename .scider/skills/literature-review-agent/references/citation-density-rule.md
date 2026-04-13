# Citation Density Rule

Source: arXiv:2604.05018, App. D.3.

## The 90% rule

> ...the system strictly constrains the model to cite only the provided
> verified papers, explicitly mandating that at least 90% of the gathered
> literature pool must be actively integrated and cited when synthesizing
> the Introduction and Related Work sections.

Why: this is the paper's core defense against citation inflation. The
literature review pool is built once via the rigorous discovery →
verification → dedup pipeline. The writing step must then *use* almost all
of it. This prevents the agent from gathering 50 papers and citing only the
3 most famous ones, which would defeat the entire literature search.

## Implementation

After the Lit Review writing call produces `intro_relwork.tex`:

```bash
python scripts/citation_coverage.py \
    --tex workspace/drafts/intro_relwork.tex \
    --pool workspace/citation_pool.json \
    --threshold 0.90
```

The script:

1. Reads `citation_pool.json` and counts `papers[]` (= N).
2. Computes `min_required = floor(0.90 * N)`.
3. Greps `intro_relwork.tex` for all `\cite{KEY}`, `\citep{KEY}`, `\citet{KEY}`,
   `\autocite{KEY}`, `\citeauthor{KEY}`, etc.
4. Counts the **unique** keys actually cited.
5. Reports `cited / N` and exits non-zero if `cited < min_required`.

## What to do on failure

The script prints the missing keys grouped by `discovered_for` cluster:

```
FAIL: 17/22 papers cited (77.3%, need ≥90%)
Uncited papers (5):
  - vaswani2017attention      [discovered_for: intro]       (Attention Is All You Need)
  - he2016deep                [discovered_for: intro]       (Deep Residual Learning ...)
  - liu2024video              [discovered_for: related_work[2.1]]  (Long Video Generation ...)
  - chen2024sparse            [discovered_for: related_work[2.2]]  (Sparse Attention Surveys ...)
  - kim2024transformer        [discovered_for: related_work[2.2]]  (Transformer Scaling Laws ...)
```

The host agent should then re-call the Lit Review writing step with an
appended instruction:

```
The previous draft cited only 17 out of 22 verified papers (77.3%, threshold
is 90%). You MUST integrate the following 5 papers into the appropriate
sections:
  - vaswani2017attention (intro): foundational attention reference
  - he2016deep (intro): foundational ResNet reference
  - liu2024video (related work 2.1): direct competing approach for long video
  - chen2024sparse (related work 2.2): sparse attention survey, group with [...]
  - kim2024transformer (related work 2.2): scaling-laws context

Do not remove any existing citations. Add new ones where contextually
appropriate. Re-emit the full template.tex with both sections updated.
```

After 2-3 re-prompts, if coverage still falls short, the pipeline should
emit a warning and proceed — the paper does not specify a hard halt on this,
only a strong constraint.
