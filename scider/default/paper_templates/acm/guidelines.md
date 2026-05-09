# ACM SIGCONF Guidelines

This template uses `acmart.cls`, which ships with standard TeX Live and MiKTeX distributions.
No additional downloads are required.

Common ACM venues using this template: SIGKDD, SIGIR, WWW, CIKM, WSDM, RecSys, SIGMOD, VLDB.

## Document Class Options

- `sigconf` — two-column proceedings format (default for most ACM conferences)
- `review` — adds line numbers for reviewer reference
- `anonymous` — removes author information for blind review
- Remove `review` and `anonymous` for the camera-ready version

## Page Limit

- **10 pages** for the main text (varies — check the Call for Papers).
- References are typically not counted toward the limit.
- Acknowledgments and appendices may be added after references.

## Mandatory Sections

- **Abstract** (≤ 250 words)
- **CCS Concepts** (ACM classification system — required)
- **Keywords**
- **Introduction**
- **Related Work**
- **Method**
- **Evaluation**
- **Conclusion**
- **References**

## Formatting

- Two-column layout, 10pt font, US Letter paper size.
- Figures and tables must be numbered and captioned.
- Use `\bibliographystyle{ACM-Reference-Format}` for ACM-formatted references.
- Citations use numeric style (`[1]`); use `\cite{}` in the text.
- Do not modify margins or fonts from the acmart defaults.

## Anonymity

- ACM conferences are typically **double-blind** (use `anonymous` option).
- Remove author names, affiliations, and acknowledgments for submission.
- Self-citations must refer to the work in third person.

## Reproducibility

- ACM encourages artifact evaluation; include dataset, code, and parameter details.
- Fill in CCS concepts accurately — reviewers use them for matching.
