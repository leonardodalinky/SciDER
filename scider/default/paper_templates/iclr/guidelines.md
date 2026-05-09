# ICLR 2026 Formatting Guidelines

## Key Facts
- **Venue**: International Conference on Learning Representations (ICLR)
- **Format**: Single-column, 12pt, letter paper
- **Page Limit**: 9 pages main content + unlimited pages for references and appendix
- **Submission**: Anonymous (blind review)
- **Style file**: `iclr2026_conference.sty` (auto-downloaded from GitHub ICLR/Master-Template)

## Structure
1. Abstract (≤ 250 words)
2. Introduction
3. Related Work (can be merged with Introduction or placed before Conclusion)
4. Method / Approach
5. Experiments
6. Conclusion
7. References (unlimited)
8. Appendix (optional, unlimited)

## Citations
- Use natbib with `\citet{}` (textual) and `\citep{}` (parenthetical)
- Bibliography style: `iclr2026_conference` (`\bibliographystyle{iclr2026_conference}`)

## Figures and Tables
- Use `\begin{figure}` / `\begin{table}` with `[t]` or `[h]` placement
- All figures must have captions

## Special Requirements
- Reproducibility: include hyperparameters, dataset details, compute resources
- Broader Impact statement encouraged (appendix)
- No coloured boxes, watermarks, or line numbers in the submitted PDF
