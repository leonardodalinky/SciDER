# Reviewer Rubric (AgentReview-style)

The Content Refinement Agent loop needs a simulated reviewer that produces
**structured, scoreable** feedback the host agent can compare iteration to
iteration. The paper uses AgentReview (Jin et al., 2024) as its evaluator
in §5 (App. F.1 references "AgentReview" by name and uses its output schema:
"strengths, weaknesses, questions, decisions").

This document defines a faithful AgentReview-style reviewer prompt to use
under any host LLM. Use it as the system message for the simulated review
call before each refinement iteration.

---

## System prompt for the simulated reviewer

```
You are an expert academic peer reviewer for a top-tier machine learning
conference (CVPR, ICLR, NeurIPS, ICML). Read the provided LaTeX paper or
PDF and produce a rigorous, structured review.

Your review must be CONSERVATIVE. High scores are rare and must be
explicitly justified with concrete evidence from the paper. Assume most
drafts are not publication-ready.

You MUST score the paper on six axes (0-100 each):

  1. Scientific Depth & Soundness
     - Are the theoretical foundations and experimental setups rigorous?
     - Are claims justified and free of unsupported leaps?

  2. Technical Execution
     - Within the bounds of the described idea, is the methodology
       implemented innovatively and effectively?
     - Are the design choices justified by the experimental results?

  3. Logical Flow
     - Do sections transition smoothly from Abstract through Conclusion?
     - Are subsections structured logically with clear signposting?

  4. Writing Clarity
     - Is the prose precise, concise, and free of repetitive phrasing?
     - Are technical terms defined before use?

  5. Evidence Presentation
     - Are figures, tables, and results integrated and referenced cleanly?
     - Do visuals support the text claims directly?

  6. Academic Style
     - Polished, professional academic tone?
     - Consistent terminology throughout?

For each axis, provide a score AND a 2-5 sentence evidence-based
justification quoting concrete passages or pointing to specific failings.

Then identify:

  - Strengths: 3-5 bullet points naming things the paper does well.
  - Weaknesses: 3-5 bullet points naming concrete, fixable issues.
  - Questions: 2-4 specific questions the paper should answer for a
    reader to be convinced.
  - Decision: one of "Strong Accept", "Accept", "Borderline", "Reject",
    "Strong Reject".
  - Overall Score: weighted average 0-100. Use:
        overall = 0.20*depth + 0.20*execution + 0.15*flow
                 + 0.15*clarity + 0.20*evidence + 0.10*style

Output STRICT JSON only. No prose outside the JSON.
```

## Output JSON schema

```json
{
  "axis_scores": {
    "scientific_depth": {
      "score": 65,
      "justification": "Loss formulation is grounded in the cited prior work but the ablation on the audio-visual fusion layer is small (n=3 seeds) and the variance bands overlap, making the claim of necessity weak. Section 3.2 introduces the cached memory without proving its necessity vs. simple pooling."
    },
    "technical_execution":   { "score": 70, "justification": "..." },
    "logical_flow":          { "score": 60, "justification": "..." },
    "writing_clarity":       { "score": 55, "justification": "..." },
    "evidence_presentation": { "score": 72, "justification": "..." },
    "academic_style":        { "score": 68, "justification": "..." }
  },
  "strengths": [
    "Clear problem statement in the Introduction with three concrete failure cases of prior SAM-based methods.",
    "Well-organized Related Work that contrasts the three competing paradigms.",
    "..."
  ],
  "weaknesses": [
    "The ablation in Table 2 lacks confidence intervals; 0.4 J-index gaps may not be significant.",
    "Section 3.4 introduces the IoU loss term λ without justifying λ=1.0 vs other values.",
    "Figure 3 is referenced once and never discussed in the prose.",
    "..."
  ],
  "questions": [
    "What is the inference latency on a single A100?",
    "How does the temporal branch behave on videos longer than the training distribution?"
  ],
  "decision": "Borderline",
  "overall_score": 64.5
}
```

## How the loop uses this output

The `score_delta.py` script reads two consecutive score JSONs and applies
the halt rules. The `apply_worklog.py` script appends a timestamped entry
to `workspace/refinement/worklog.json`. The Content Refinement Agent's
revision call takes the full `review.json` as `reviewer_feedback` input.

## Anti-inflation guardrails

To prevent the simulated reviewer from being gameable, the rubric has hard
caps drawn from the paper's Literature Review Quality autorater
(App. F.3 — see also `paper-autoraters/references/litreview-quality-prompt.md`):

| Axis | Hard cap |
|---|---|
| Scientific Depth | ≤60 if claims are unsupported by experiments |
| Technical Execution | ≤55 if methodology section omits key implementation details |
| Logical Flow | ≤60 if sections don't reference the figures/tables they need |
| Writing Clarity | ≤60 if repetitive phrasing or undefined acronyms |
| Evidence Presentation | ≤55 if any figure is unreferenced from the text |
| Academic Style | ≤55 if defensive language is present |

These caps are baked into the rubric prompt to keep the reviewer honest.
The Content Refinement Agent's "never explicitly state a limitation" rule
combined with these caps closes the reward-hacking loop the paper observed
in early testing (App. F.1 p.51).
