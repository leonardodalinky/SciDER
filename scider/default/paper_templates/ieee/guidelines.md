# IEEE Conference Guidelines

This template uses `IEEEtran.cls`, which ships with standard TeX Live and MiKTeX distributions.
No additional downloads are required.

Adjust the `\documentclass` option for your specific conference:
- `[conference]` — IEEE conference proceedings (two-column, no headers)
- `[journal]` — IEEE Transactions / journal (one-column with headers)
- `[compsoc,conference]` — IEEE Computer Society conference style

## Page Limit

- Typical IEEE conference: **6–8 pages** (varies by conference — check the Call for Papers).
- References are included in the page count.
- An optional 1-page appendix may be permitted; check your conference's rules.

## Mandatory Sections

- **Abstract** (≤ 150 words; appears before Introduction in two-column layout)
- **Keywords** (4–6 IEEE-style keywords)
- **Introduction**
- **Related Work**
- **Proposed Method**
- **Experimental Results**
- **Conclusion**
- **References**

## Formatting

- Two-column layout, 10pt font, US Letter paper size.
- Figures and tables must be numbered and captioned.
- Use `\bibliographystyle{IEEEtran}` for IEEE-formatted references.
- Use `\cite{}` for citations (numeric style, e.g., `[1]`).
- Do not modify margins from the IEEEtran defaults.
- Equations are numbered on the right margin with `(1)`, `(2)`, etc.

## Anonymity

- IEEE conferences vary: some are **double-blind**, others are single-blind.
- Check your specific conference's author guidelines.
- When double-blind, remove author names and mask self-citations.

## Ethics and Reproducibility

- Include a brief statement on broader impact if required by the conference.
- Report key experimental parameters (dataset sizes, hardware, training time).
