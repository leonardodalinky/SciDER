# AAAI 2026 Conference Guidelines

## Style Files

Download `aaai26.sty` and `aaai.bst` from https://aaai.org/authorkit26-1/
(adjust year for your target AAAI conference) and place them alongside
`template.tex` in the template directory. Then:
1. Uncomment `\usepackage{aaai26}` in `template.tex`.
2. Change `\bibliographystyle{abbrvnat}` to `\bibliographystyle{aaai}`.

## Page Limit

- **7 pages** for the main text (technical content).
- **1 additional page** for references only.
- A 2-page appendix is permitted; reviewers are not required to read it.
- Total submission: ≤ 10 pages (7 main + 1 references + 2 appendix).

## Mandatory Sections

- **Abstract** (≤ 250 words; appears verbatim in the program booklet)
- **Introduction**
- **Related Work**
- **Method** (or "Approach", "Model", "Algorithm")
- **Experiments**
- **Conclusion**
- **References**
- **Ethics Statement** (required since AAAI 2023)

## Formatting

- Single-column layout, 10pt Times font, US Letter paper size.
- Figures and tables must be numbered and captioned.
- Use `\bibliographystyle{aaai}` with `aaai.bst` for AAAI-formatted references.
- Citations are numbered `[1]` style; use `\cite{}` in the text.
- Do not modify margins or fonts from the official AAAI style.
- No running headers or footers in the submission version.

## Anonymity

- Submissions are **double-blind**.
- No author names, affiliations, or acknowledgments.
- Self-citations must use third-person phrasing.
- Do not link to non-anonymized code repositories.

## Ethics and Responsible AI

- An **Ethics Statement** is required in the main paper body.
- Address data collection, privacy, bias, misuse potential, and environmental impact as relevant.
- The ethics statement does not count toward the 7-page limit.
