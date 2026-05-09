# ICML / ICLR Guidelines

This template covers both ICML and ICLR, which share a similar article-class + venue-sty structure.

## Style Files

**ICML 2026:**
Download `icml2026.sty` from https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip
and place it alongside `template.tex` in the template directory.
Uncomment `\usepackage{icml2026}` in `template.tex`.

**ICLR 2026:**
Download `iclr2026_conference.sty` from https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip
and place it alongside `template.tex` in the template directory.
Uncomment `\usepackage{iclr2026_conference}` in `template.tex`.

## Page Limit

**ICML:**
- **9 pages** for the main text, excluding references and appendix.
- Unlimited references (not counted toward limit).
- Appendix may be included after references; reviewers need not read it.

**ICLR:**
- **9 pages** recommended (no hard limit, but papers significantly longer than 12 pages
  may be desk-rejected; check the current year's rules).
- References and appendix do not count toward the limit.

## Mandatory Sections

- **Abstract** (≤ 250 words)
- **Introduction**
- **Related Work** (may appear as a section or merged into the intro)
- **Method** / **Problem Setup**
- **Experiments** (setup, main results, ablations)
- **Conclusion**
- **References**
- ICLR only: **Reproducibility Statement** (encouraged)

## Formatting

- Single-column layout, 10pt font, US Letter paper size.
- Figures and tables must be numbered and captioned.
- Uses `natbib` with `abbrvnat` or `plainnat` (Author, Year style).
- Do not modify margins or font sizes from the official style file.

## Anonymity

- Both ICML and ICLR use **double-blind** review.
- No author names, affiliations, or acknowledgments in the submission.
- Self-citations must be phrased in third person.
- Do not include links to identifying code repositories.

## Reproducibility

- ICML has a reproducibility checklist (appended to submission).
- ICLR requires a Reproducibility Statement in the main paper.
- Report all hyperparameters, random seeds, compute, and dataset details.
