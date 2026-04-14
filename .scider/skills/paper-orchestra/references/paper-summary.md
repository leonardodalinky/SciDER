# PaperOrchestra — 1-page distillation

**Source:** Song, Y., Song, Y., Pfister, T., Yoon, J. *PaperOrchestra: A
Multi-Agent Framework for Automated AI Research Paper Writing.*
arXiv:2604.05018, 2026. <https://arxiv.org/pdf/2604.05018>

## Problem

Existing autonomous research-paper writers are tightly coupled to specific
experimental loops. They cannot transform unstructured human-provided
materials (an idea, an experimental log, a template) into a submission-ready
manuscript. Survey-only frameworks (AutoSurvey2, LiRA) lack full-paper
synthesis. Single-agent baselines hallucinate citations and produce shallow
literature reviews.

## Approach

A five-agent pipeline that maps `W(I, E, T, G, F) → P = (paper.tex, paper.pdf)`,
where:

- `I` = Idea Summary (Sparse or Dense markdown)
- `E` = Experimental Log (raw data, ablations, observations)
- `T` = LaTeX template
- `G` = Conference guidelines
- `F` = Optional pre-existing figures

The five agents:

1. **Outline Agent (1 LLM call)** — synthesizes the inputs into a JSON outline
   with three sub-plans: a visualization plan, a literature search strategy
   (macro for Intro + micro for Related Work), and a section-level writing
   plan with mandatory citation hints for every dataset / metric / baseline.
2. **Plotting Agent (~20–30 calls)** — executes the visualization plan, using
   PaperBanana-style few-shot retrieval and a VLM critique loop to
   iteratively refine generated figures.
3. **Literature Review Agent (~20–30 calls)** — runs parallel candidate
   discovery (10 web search workers), then sequential Semantic Scholar
   verification (1 QPS, Levenshtein > 70 fuzzy title match), dedupes by
   Semantic Scholar paper ID, and drafts Introduction + Related Work using
   ≥90% of the verified pool.
4. **Section Writing Agent (1 single multimodal call)** — drafts Abstract,
   Methodology, Experiments, Conclusion; extracts numeric values from `E`
   into LaTeX booktabs tables; integrates figures from Step 2.
5. **Content Refinement Agent (~5–7 calls)** — runs an AgentReview-style
   simulated peer review loop; accepts revisions only if overall score
   improves OR ties with non-negative net sub-axis change; reverts on
   decrease or negative tie-break; halts at iteration cap (~3).

Steps 2 and 3 run in parallel.

## Engineering details that matter for fidelity

- **Universal Anti-Leakage Prompt** (App. D.4) prepended to every writing call.
- **Citation cutoff**: research_cutoff aligned to venue submission deadline
  (Nov 2024 for CVPR 2025, Oct 2024 for ICLR 2025). Months default to first
  day of the month for strict-predates comparison.
- **Citation verification**: Levenshtein title ratio > 70, year alignment
  bonus, must have abstract, must strictly predate cutoff, dedup by S2 paperId.
- **Citation density rule**: ≥90% of the gathered pool MUST be cited in
  Intro + Related Work.
- **Refinement safety**: agent must ignore reviewer requests for new
  experiments and must never explicitly write "limitation" — both prevent
  reward hacking against the simulated evaluator.
- **One single Section Writing call** — not chunked per section.

## Results (paper §5)

- Side-by-side LLM evaluation: PaperOrchestra wins 88–99% on literature
  review quality and 39–86% on overall quality vs single-agent and
  AI-Scientist-v2 baselines.
- Simulated acceptance rate (ScholarPeer): 84% (CVPR), 81% (ICLR) — close to
  human GT rates of 86%/94%.
- Citation P1 Recall: 12.59–13.75% absolute over the strongest baseline.
- Refinement loop alone: +19% (CVPR) and +22% (ICLR) acceptance rate gains.

## What this repo implements

This repo is a **host-agent-pluggable skill pack** that lets any coding agent
(Claude Code, Cursor, Antigravity, Cline, Aider, OpenCode) execute the
PaperOrchestra pipeline. There are no API keys, no LLM SDKs, no embedded
network clients. Each skill is a markdown instruction document plus
deterministic local helper scripts. The host agent does all LLM reasoning,
web search, and Semantic Scholar fetching using its own native tools.

See `pipeline.md`, `io-contract.md`, and `host-integration.md` for details.
