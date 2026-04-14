# Safe Revision Rules

The Content Refinement Agent prompt (App. F.1 p.50–51) imposes two
anti-reward-hacking constraints. Both must be enforced not just by the
prompt but by deterministic post-revision gates, because LLMs occasionally
forget instructions buried in long prompts.

## Rule 1 — Ignore reviewer requests for new experiments

The simulated reviewer will sometimes ask:

- "What if you ablated the temperature parameter?"
- "How does this compare to baseline X?"
- "Have you tried this on dataset Y?"

The Refinement Agent must **not** fabricate answers to these. The paper:

> If the reviewer asks for new experiments, ablations, or baselines that
> are NOT in experimental_log.md, simply ignore those specific requests.
> Your job is purely presentation refinement of the existing completed
> experiments, not adding or promising to add new experiments.

### Enforcement

There is no fully deterministic way to grep for "fabricated experiments" —
it requires reading the new content and cross-checking against
`experimental_log.md`. The pragmatic check:

1. Run the orphan-citation gate from `section-writing-agent/scripts/orphan_cite_gate.py`.
   New numeric claims often come bundled with new (orphan) citations.
2. Run a numeric-claim grep: extract every `\d+\.\d+%?` from the new draft,
   intersect with `\d+\.\d+%?` in `experimental_log.md`. New numbers in the
   draft that aren't in the log are suspicious. (False positives possible
   for parameter counts and dates; review manually.)

The orchestrator should re-prompt the refinement step if either gate fires
with new fabricated claims.

## Rule 2 — Never explicitly state a limitation

The paper:

> The directive to "never explicitly state a limitation" prevents reward
> hacking. During early testing, the agent exploited the automated
> reviewer's scoring function by superficially listing missing baselines
> as limitations to artificially inflate acceptance scores.

### Enforcement (deterministic)

Grep the revised draft for the substring `limitation` (case-insensitive),
excluding LaTeX comments. If found anywhere in the body, reject the
revision and re-prompt:

```bash
# pseudocode — implement inline in the host agent
grep -in -E '\blimitation' workspace/refinement/iter<N>/paper.tex \
    | grep -v '^\s*%'
```

Allowed contexts (these are NOT violations):

- LaTeX comments: `% address the limitation of ...`
- Citation context: a paper title containing "limitation" cited in
  `\cite{...}`. The grep should ignore the inside of `\cite{...}` braces.
- Quoted prior-work descriptions: "Smith et al. acknowledge the
  limitation..." — context-dependent. The simplest rule is "no instances
  of the word 'limitation' in the running prose at all", and let the host
  agent handle edge cases by re-prompting if a legitimate use is needed.

This is a strict rule. The Refinement Agent should rewrite "we acknowledge
the limitation that our method..." as "our method assumes..." or "the
proposed approach is most effective when...". Reframing, not listing.

## Rule 3 — Numeric ground truth

> All numerical claims (accuracy, parameter count, training hours,
> latency) MUST be verified against experimental_log.md.

The grep heuristic above catches this partially. The host agent should
also instruct the refinement step explicitly: "any numeric value you cite
in your revision must already exist in experimental_log.md or
metrics.json."

## Rule 4 — Citation integrity

The orphan-citation gate from
`section-writing-agent/scripts/orphan_cite_gate.py` must pass after every
refinement iteration. Re-run it as part of the post-revision checks:

```bash
python skills/section-writing-agent/scripts/orphan_cite_gate.py \
    workspace/refinement/iter<N>/paper.tex \
    workspace/refs.bib
```

If the refinement step introduced a new `\cite{KEY}` not in `refs.bib`,
revert the iteration and re-prompt with an explicit instruction to use
only existing keys.

## Rule 5 — LaTeX integrity

Re-run `latex_sanity.py` and `latexmk -pdf` after every revision. If the
revision broke the build, revert.

## Summary checklist for each refinement iteration

```bash
# 1. apply revision → iter<N>/paper.tex
# 2. compile
cd workspace/refinement/iter<N>/ && latexmk -pdf -interaction=nonstopmode paper.tex

# 3. structural sanity
python skills/section-writing-agent/scripts/latex_sanity.py paper.tex || REVERT
python skills/section-writing-agent/scripts/orphan_cite_gate.py paper.tex ../../refs.bib || REVERT

# 4. anti-leakage
python skills/paper-orchestra/scripts/anti_leakage_check.py paper.tex || REVERT

# 5. limitation grep (Rule 2)
grep -in -E '\blimitation' paper.tex | grep -v '^\s*%' && REVERT

# 6. score and decide
python skills/content-refinement-agent/scripts/score_delta.py \
    --prev ../iter<N-1>/score.json --curr score.json
# exit 0 → keep, exit 1/2 → revert
```

If all gates pass and `score_delta.py` returns 0, the iteration is
accepted.
