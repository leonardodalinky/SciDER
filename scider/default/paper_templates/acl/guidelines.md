# ACL / EMNLP / NAACL Guidelines

This template applies to ACL Anthology venues including ACL, EMNLP, NAACL, EACL, and AACL.
Check the specific Call for Papers for your target venue.

## Style File

Download `acl.sty` and `acl_natbib.bst` from https://github.com/acl-org/acl-style-files/archive/refs/heads/master.zip
and place them alongside `template.tex` in the template directory.

## Page Limit

- **8 pages** for the main text (long papers); **4 pages** for short papers.
- **2 additional pages** for references only (not counted toward the main limit).
- Unlimited pages for ethics/limitations/acknowledgments sections (placed after references).
- Optional appendix after references is not counted but reviewers need not read it.

## Mandatory Sections

- **Abstract** (≤ 200 words)
- **Introduction**
- **Related Work**
- **Methodology** (or "Approach", "Model", "System")
- **Experimental Setup**
- **Results**
- **Conclusion**
- **References**
- **Limitations** (required at ACL 2024+; placed after References)
- **Ethics Statement** (encouraged; placed after Limitations)

## Formatting

- Single-column layout, 11pt font, US Letter paper size.
- Figures and tables must be numbered and captioned.
- Citations use `natbib` with the `acl_natbib` style (Author, Year format).
- Use `\citet{}` for in-text citations and `\citep{}` for parenthetical.
- Do not modify margins or font sizes from the official ACL style file.

## Anonymity

- Submissions are **double-blind**: no author names, affiliations, or acknowledgments.
- Remove references to your own prior work that would identify you, or cite in third person.
- Do not include links to personal GitHub repositories or project pages.

## Reproducibility

- ACL venues require a reproducibility checklist.
- Report all hyperparameters, random seeds, evaluation datasets, and compute resources.
- Code and data should be available or described clearly enough for replication.
