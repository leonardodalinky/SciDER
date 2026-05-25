"""Paper workspace bootstrap — bridge SciDER outputs into PaperOrchestra inputs.

This module contains pure, side-effect-free (except for disk writes) helpers
that turn SciDER's data / experiment / ideation summaries into the strict
``inputs/{idea.md, experimental_log.md, template.tex, conference_guidelines.md,
figures/}`` layout that PaperOrchestra's skills consume.

It does NOT call LLMs. All conversions are string templating so the same
workflow run is deterministic and idempotent.

The writing agent receives the bootstrapped workspace and then delegates the
real work (outlining, plotting, lit review, section writing, refinement) to
the paper-orchestra skill.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

from loguru import logger

# Bundled default template lives next to this module's sibling package so
# it ships with SciDER and is reachable without a data_files setup.py
# entry. The default is intentionally venue-agnostic ("simple") — a minimal
# article-class skeleton with the sections PaperOrchestra expects. Swap it
# out by passing ``template_dir`` (a directory containing ``template.tex``
# and ``guidelines.md``) or the explicit ``template_tex`` /
# ``conference_guidelines`` string overrides to ``bootstrap_paper_workspace``.
#
# Template directory layout (see ``scider/default/paper_templates/simple/``):
#
#     <template_dir>/
#     ├── template.tex               # required — becomes inputs/template.tex
#     ├── guidelines.md              # required — becomes inputs/conference_guidelines.md
#     ├── neurips_2024.sty           # optional — any support files (.sty, .bst,
#     ├── refs.bst                   #   .cls, .bib, ...) are copied verbatim
#     └── ...                        #   into inputs/ alongside template.tex
#
_DEFAULTS_DIR = Path(__file__).resolve().parent.parent / "default" / "paper_templates"
DEFAULT_TEMPLATE_DIR = _DEFAULTS_DIR / "simple"
# Back-compat aliases — the tests and downstream callers reference these.
DEFAULT_TEMPLATE_TEX = DEFAULT_TEMPLATE_DIR / "template.tex"
DEFAULT_CONFERENCE_GUIDELINES = DEFAULT_TEMPLATE_DIR / "guidelines.md"

# Conventional file names inside a template directory.
_TEMPLATE_TEX_NAME = "template.tex"
_GUIDELINES_NAME = "guidelines.md"

# Copy these image extensions from figures_src into inputs/figures/
_FIGURE_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".webp"}

# Registry of bundled venue templates shipped with SciDER.
# Keys are internal identifiers; values are absolute paths to template directories.
# Each directory must contain template.tex and guidelines.md.
BUNDLED_TEMPLATES: dict[str, Path] = {
    "simple": _DEFAULTS_DIR / "simple",
    "neurips": _DEFAULTS_DIR / "neurips",
    "acl": _DEFAULTS_DIR / "acl",
    "ieee": _DEFAULTS_DIR / "ieee",
    "acm": _DEFAULTS_DIR / "acm",
    "icml": _DEFAULTS_DIR / "icml",
    "iclr": _DEFAULTS_DIR / "iclr",
    "aaai": _DEFAULTS_DIR / "aaai",
}

BUNDLED_TEMPLATE_LABELS: dict[str, str] = {
    "simple": "Simple (generic article)",
    "neurips": "NeurIPS 2026",
    "acl": "ACL / EMNLP / NAACL",
    "ieee": "IEEE Conference",
    "acm": "ACM SIGCONF",
    "icml": "ICML 2026",
    "iclr": "ICLR 2026",
    "aaai": "AAAI 2026",
}


# Extensions considered "LaTeX support files" for auto-extraction from zips.
_STYLE_FILE_EXTS = {".sty", ".bst", ".cls", ".clo", ".cfg"}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def bootstrap_paper_workspace(
    paper_workspace: str | Path,
    *,
    idea_summary: str,
    experimental_log: str,
    template_dir: str | Path | None = None,
    template_tex: str | None = None,
    conference_guidelines: str | None = None,
    figures_src: str | Path | None = None,
) -> Path:
    """Materialize a PaperOrchestra-ready ``inputs/`` layout.

    Writes the four required input files and optionally copies pre-existing
    figures into ``inputs/figures/``. The template comes from a directory:

    - ``template_dir`` — a directory containing ``template.tex``,
      ``guidelines.md``, and any number of auxiliary LaTeX support files
      (``.sty``, ``.bst``, ``.cls``, ``.bib``, ...). All non-convention
      files are copied verbatim into ``inputs/`` so ``\\usepackage{foo}``
      in ``template.tex`` resolves against ``inputs/foo.sty``. Defaults to
      the bundled ``simple`` template directory.
    - ``template_tex`` / ``conference_guidelines`` — explicit string
      overrides that take precedence over whatever was read from
      ``template_dir``. Useful when the caller has already loaded the
      content and just wants to inject it without touching the filesystem.

    Returns the absolute Path to the paper workspace directory.
    """
    paper_workspace = Path(paper_workspace).resolve()
    inputs_dir = paper_workspace / "inputs"
    figures_dir = inputs_dir / "figures"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # idea.md
    (inputs_dir / "idea.md").write_text(idea_summary, encoding="utf-8")

    # experimental_log.md
    (inputs_dir / "experimental_log.md").write_text(experimental_log, encoding="utf-8")

    # Template directory — defaults to the bundled "simple" directory.
    resolved_template_dir = Path(template_dir).resolve() if template_dir else DEFAULT_TEMPLATE_DIR
    if not resolved_template_dir.is_dir():
        raise FileNotFoundError(f"Template directory not found: {resolved_template_dir}")

    # Download and cache venue style files if style_files.json is present.
    # Runs before the copy so freshly downloaded files are included.
    _fetch_and_cache_style_files(resolved_template_dir)

    # Copy template.tex, guidelines.md, and any auxiliary files sitting at
    # the top level of the template directory.
    _copy_template_dir_into_inputs(resolved_template_dir, inputs_dir)

    # Activate any downloaded style files by uncommenting the matching
    # \usepackage line in inputs/template.tex.
    _activate_style_files(inputs_dir)

    # Explicit string overrides win over whatever was copied from disk.
    if template_tex is not None:
        (inputs_dir / "template.tex").write_text(template_tex, encoding="utf-8")
    if conference_guidelines is not None:
        (inputs_dir / "conference_guidelines.md").write_text(
            conference_guidelines, encoding="utf-8"
        )

    # figures_src/*.{png,jpg,jpeg,pdf,webp} → inputs/figures/
    if figures_src is not None:
        figures_src = Path(figures_src)
        if figures_src.is_dir():
            n = 0
            for f in figures_src.iterdir():
                if f.is_file() and f.suffix.lower() in _FIGURE_EXTS:
                    shutil.copy2(f, figures_dir / f.name)
                    n += 1
            logger.info("Copied {} figure(s) from {} to {}", n, figures_src, figures_dir)
        else:
            logger.debug("figures_src {} is not a directory — skipping", figures_src)

    logger.info("Paper workspace bootstrapped at {}", paper_workspace)
    return paper_workspace


def _fetch_and_cache_style_files(template_dir: Path) -> None:
    """Download and cache venue style files declared in ``style_files.json``.

    Reads ``<template_dir>/style_files.json`` (if present), fetches the zip at
    ``url``, and extracts every entry whose extension is in
    ``extract_extensions`` (defaulting to ``.sty .bst .cls .clo .cfg``) into
    ``template_dir``.  Directory nesting inside the zip is stripped — only the
    basename is used — so GitHub archive layouts (e.g.
    ``acl-style-files-master/acl.sty``) work without path configuration.

    Files are cached: if a downloaded file already exists in ``template_dir``
    it is skipped, so repeated bootstrap calls do not re-download.  On any
    network or I/O error a warning is logged and the function returns without
    raising so the rest of the bootstrap can continue with whatever files are
    already present.
    """
    config_path = template_dir / "style_files.json"
    if not config_path.exists():
        return

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read style_files.json in {}: {}", template_dir, exc)
        return

    url = config.get("url", "").strip()
    files: list[str] = config.get("files", [])
    if not url and not files:
        return

    want_exts = set(config.get("extract_extensions", list(_STYLE_FILE_EXTS)))

    _headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def _fetch(fetch_url: str) -> bytes | None:
        try:
            req = urllib.request.Request(fetch_url, headers=_headers)  # noqa: S310
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                return resp.read()
        except Exception as exc:
            logger.warning(
                "Could not download '{}' for template '{}': {}.",
                fetch_url, template_dir.name, exc,
            )
            return None

    # --- zip mode ---
    if url:
        logger.info("Fetching style files for template '{}' from {}", template_dir.name, url)
        raw = _fetch(url)
        if raw is None:
            logger.warning(
                "Proceeding without style files for template '{}' — venue-specific formatting may be absent.",
                template_dir.name,
            )
        else:
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    extracted = 0
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        suffix = Path(info.filename).suffix.lower()
                        if suffix not in want_exts:
                            continue
                        basename = Path(info.filename).name
                        dest = template_dir / basename
                        if dest.exists():
                            logger.debug("Style file {} already cached — skipping", basename)
                            continue
                        dest.write_bytes(zf.read(info.filename))
                        logger.info("Cached style file {} in {}", basename, template_dir)
                        extracted += 1
                    if extracted == 0:
                        logger.debug(
                            "No new style files extracted from zip for template '{}'",
                            template_dir.name,
                        )
            except Exception as exc:
                logger.warning(
                    "Failed to extract style files for template '{}': {}. "
                    "Proceeding without them.",
                    template_dir.name,
                    exc,
                )

    # --- individual files mode ---
    for file_url in files:
        basename = Path(file_url).name
        dest = template_dir / basename
        if dest.exists():
            logger.debug("Style file {} already cached — skipping", basename)
            continue
        suffix = Path(basename).suffix.lower()
        if suffix not in want_exts:
            logger.debug("Skipping {} — extension not in want_exts", basename)
            continue
        logger.info("Fetching style file {} for template '{}'", basename, template_dir.name)
        raw = _fetch(file_url)
        if raw is not None:
            dest.write_bytes(raw)
            logger.info("Cached style file {} in {}", basename, template_dir)


def _activate_style_files(inputs_dir: Path) -> None:
    """Uncomment ``\\usepackage`` lines in ``inputs/template.tex`` for any
    ``.sty`` files that were successfully downloaded into ``inputs/``.

    The bundled venue templates ship with the relevant ``\\usepackage`` call
    commented out (so they compile as plain ``article`` without the style
    file).  Once ``_fetch_and_cache_style_files`` has placed a ``.sty`` file
    alongside ``template.tex``, this function patches ``inputs/template.tex``
    in-place to activate the first matching commented ``\\usepackage`` line.

    Rules:
    - Only the **first** matching line per package stem is uncommented (avoids
      activating both the accepted and submission variants for ICML).
    - Lines already uncommented are left untouched.
    - A line is considered a match when the package name inside ``{…}``
      equals the ``.sty`` file's stem (e.g. ``acl.sty`` → ``acl``).
    """
    template_tex = inputs_dir / "template.tex"
    if not template_tex.exists():
        return

    sty_stems = {f.stem for f in inputs_dir.glob("*.sty")}
    if not sty_stems:
        return

    # Matches a \usepackage call that is the first non-whitespace token on the
    # uncommented line — this excludes prose comments like "% see \usepackage{foo}".
    _pkg_re = re.compile(r'^\s*\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}')
    activated: set[str] = set()

    lines = template_tex.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []
    changed = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("%"):
            # Strip the leading comment marker to get the "uncommented" content.
            uncommented = re.sub(r'^%\s*', '', stripped)
            m = _pkg_re.match(uncommented)
            if m:
                pkg = m.group(1)
                if pkg in sty_stems and pkg not in activated:
                    # Remove the leading comment marker (handles "% " and "%\s*")
                    new_line = re.sub(r'^(\s*)%\s*', r'\1', line, count=1)
                    new_lines.append(new_line)
                    activated.add(pkg)
                    changed = True
                    logger.info(
                        "Activated \\usepackage{{{}}} in inputs/template.tex", pkg
                    )
                    continue
        new_lines.append(line)

    if changed:
        template_tex.write_text("".join(new_lines), encoding="utf-8")


def _copy_template_dir_into_inputs(template_dir: Path, inputs_dir: Path) -> None:
    """Copy a template directory's top-level files into ``inputs/``.

    Conventions:
    - ``template_dir/template.tex``   → ``inputs/template.tex`` (required)
    - ``template_dir/guidelines.md``  → ``inputs/conference_guidelines.md`` (required)
    - Any other top-level file → ``inputs/<same name>`` (verbatim copy, so
      auxiliary ``.sty`` / ``.bst`` / ``.cls`` / ``.bib`` files live next to
      ``template.tex`` where LaTeX's ``\\usepackage`` can find them).
    Subdirectories inside ``template_dir`` are ignored — keep the template
    flat for now.
    """
    tex_src = template_dir / _TEMPLATE_TEX_NAME
    guidelines_src = template_dir / _GUIDELINES_NAME
    if not tex_src.is_file():
        raise FileNotFoundError(
            f"Template directory {template_dir} is missing {_TEMPLATE_TEX_NAME}"
        )
    if not guidelines_src.is_file():
        raise FileNotFoundError(f"Template directory {template_dir} is missing {_GUIDELINES_NAME}")

    # Required files, renamed to match PaperOrchestra's io-contract.
    shutil.copy2(tex_src, inputs_dir / "template.tex")
    shutil.copy2(guidelines_src, inputs_dir / "conference_guidelines.md")

    # Auxiliary top-level files — anything that is not template.tex or
    # guidelines.md gets copied as-is. Subdirectories are skipped.
    aux_count = 0
    for entry in sorted(template_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.name in (_TEMPLATE_TEX_NAME, _GUIDELINES_NAME):
            continue
        shutil.copy2(entry, inputs_dir / entry.name)
        aux_count += 1
    if aux_count:
        logger.info(
            "Copied {} auxiliary template file(s) from {} to {}",
            aux_count,
            template_dir,
            inputs_dir,
        )


def build_sparse_idea_from_query(user_query: str) -> str:
    """Turn a freeform user query into a PaperOrchestra Sparse idea.md.

    Only the Problem Statement is pre-filled; the other three sections are
    left as explicit TODO placeholders. The writing agent / section-writing
    skill will flesh them out using the experimental_log and outline.
    """
    stub = user_query.strip() or "(no user query provided)"
    return f"""## Problem Statement

{stub}

## Core Hypothesis

TODO — derive the core hypothesis from the experimental_log observations and
the user query above. The writing agent will fill this in before Section
Writing (Step 4).

## Proposed Methodology (High-Level Technical Approach)

TODO — reconstruct the methodology from the experimental_log's Experimental
Setup and the SciDER experiment summary.

## Expected Contribution

TODO — extract intended contributions from the SciDER experiment summary's
Conclusions section.
"""


def build_idea_from_ideation(
    research_ideas: list[dict],
    selected_index: int | None,
    user_query: str,
) -> str:
    """Turn ideation output into PaperOrchestra idea.md.

    Picks the user-selected idea if ``selected_index`` is set, otherwise the
    highest-scored idea, otherwise the first idea, otherwise falls back to
    ``build_sparse_idea_from_query`` on ``user_query``.
    """
    if not research_ideas:
        return build_sparse_idea_from_query(user_query)

    def _select_score(i: dict) -> float:
        if i.get("composite_score") is not None:
            return float(i["composite_score"])
        return float(i.get("novelty_score") or 0.0)

    if selected_index is not None and 0 <= selected_index < len(research_ideas):
        idea = research_ideas[selected_index]
    else:
        scored = [(_select_score(i), i) for i in research_ideas if isinstance(i, dict)]
        if scored:
            scored.sort(key=lambda p: p[0], reverse=True)
            idea = scored[0][1]
        else:
            idea = research_ideas[0]

    title = idea.get("title") or "Untitled research idea"
    description = (idea.get("description") or "").strip()
    rationale = (idea.get("rationale") or "").strip()
    experiment_plan = (idea.get("experiment") or idea.get("experiment_plan") or "").strip()
    contribution = (idea.get("contribution") or idea.get("expected_contribution") or "").strip()

    problem = description or user_query.strip() or "(no description available)"
    hypothesis = rationale or "TODO — derive the core hypothesis from the idea description above."
    methodology = (
        experiment_plan
        or "TODO — flesh out the high-level technical approach from the idea description."
    )
    expected = (
        contribution or "TODO — state the intended theoretical or practical value of this work."
    )

    return f"""## Problem Statement

**{title}**

{problem}

## Core Hypothesis

{hypothesis}

## Proposed Methodology (High-Level Technical Approach)

{methodology}

## Expected Contribution

{expected}
"""


def build_experimental_log(
    *,
    data_summary: str,
    experiment_summary: str,
    user_query: str,
) -> str:
    """Produce PaperOrchestra-shaped experimental_log.md via string templating.

    Output strictly conforms to the three-section layout validated by
    paper-orchestra/scripts/validate_inputs.py:

        ## 1. Experimental Setup
        ## 2. Raw Numeric Data
        ## 3. Qualitative Observations

    Each section is guaranteed non-empty (uses "(none recorded)" as a
    placeholder) so the validator warnings don't trip. Anti-leakage: "Figure
    N" / "Table N" references from the source summaries are best-effort
    stripped.
    """
    setup = _extract_setup_section(data_summary, experiment_summary, user_query)
    raw_numeric = _extract_raw_numeric_section(experiment_summary)
    observations = _extract_observations_section(experiment_summary)

    setup = _strip_figure_table_refs(setup)
    raw_numeric = _strip_figure_table_refs(raw_numeric)
    observations = _strip_figure_table_refs(observations)

    return f"""# Experimental Log

## 1. Experimental Setup

{setup or "(none recorded)"}

## 2. Raw Numeric Data

{raw_numeric or "(none recorded)"}

## 3. Qualitative Observations

{observations or "(none recorded)"}
"""


# --------------------------------------------------------------------------- #
# Section extractors — dumb but deterministic
# --------------------------------------------------------------------------- #


def _extract_setup_section(
    data_summary: str,
    experiment_summary: str,
    user_query: str,
) -> str:
    """Assemble the Setup section from whatever SciDER produced.

    We concatenate:
      1. The user query (wrapped in a bullet so it reads like a research
         goal).
      2. The ``## Data Structure`` region of the data summary (if present).
      3. The ``## Methodology`` region of the experiment summary (if present).
    """
    parts: list[str] = []
    if user_query and user_query.strip():
        parts.append(f"* **Research goal:** {user_query.strip()}")

    data_block = _extract_section(data_summary, ("data structure", "overview"))
    if data_block:
        parts.append("**Datasets / Data Structure (from data analysis):**\n\n" + data_block)

    methodology = _extract_section(
        experiment_summary, ("methodology", "implementation", "approach")
    )
    if methodology:
        parts.append("**Implementation / Methodology (from experiment run):**\n\n" + methodology)

    return "\n\n".join(p for p in parts if p)


def _extract_raw_numeric_section(experiment_summary: str) -> str:
    """Pull tables / numeric paragraphs from the experiment summary.

    We prefer the ``## Results`` section. If no explicit Results heading
    exists, we keep every markdown table in the summary as a fallback.
    """
    results = _extract_section(experiment_summary, ("results", "metrics", "evaluation"))
    if results:
        return results
    tables = _extract_markdown_tables(experiment_summary)
    return tables or ""


def _extract_observations_section(experiment_summary: str) -> str:
    """Pull qualitative commentary from Analysis / Conclusions sections."""
    parts: list[str] = []
    for heading in ("analysis", "discussion", "conclusion", "conclusions", "notes"):
        block = _extract_section(experiment_summary, (heading,))
        if block:
            parts.append(block)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Markdown-aware helpers
# --------------------------------------------------------------------------- #


def _extract_section(markdown: str, heading_aliases: tuple[str, ...]) -> str:
    """Return the body of the first ``##`` or ``###`` section whose heading
    (case-insensitive, stripped of punctuation) matches any of the aliases.

    Stops at the next heading of the same or higher level. Returns '' if
    nothing matched.
    """
    if not markdown:
        return ""
    lines = markdown.splitlines()
    aliases = {a.lower() for a in heading_aliases}

    start_idx: int | None = None
    start_level: int | None = None
    for i, line in enumerate(lines):
        m = _heading_match(line)
        if not m:
            continue
        level, title = m
        if _normalize_heading(title) in aliases and level <= 3:
            start_idx = i + 1
            start_level = level
            break

    if start_idx is None:
        return ""

    body: list[str] = []
    for line in lines[start_idx:]:
        m = _heading_match(line)
        if m and m[0] <= start_level:  # type: ignore[operator]
            break
        body.append(line)
    return "\n".join(body).strip()


def _heading_match(line: str) -> tuple[int, str] | None:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    level = len(stripped) - len(stripped.lstrip("#"))
    if level == 0 or level > 6:
        return None
    title = stripped[level:].strip()
    return level, title


def _normalize_heading(title: str) -> str:
    out: list[str] = []
    for ch in title.lower():
        if ch.isalnum() or ch == " ":
            out.append(ch)
    return " ".join("".join(out).split())


def _extract_markdown_tables(markdown: str) -> str:
    """Return all contiguous markdown table blocks concatenated, or ''."""
    if not markdown:
        return ""
    lines = markdown.splitlines()
    tables: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current.append(line)
        else:
            if current:
                tables.append("\n".join(current))
                current = []
    if current:
        tables.append("\n".join(current))
    return "\n\n".join(tables)


def _strip_figure_table_refs(text: str) -> str:
    """Best-effort removal of 'See Figure 1' / 'as shown in Table 2' phrases.

    PaperOrchestra's validator rejects experimental logs that contain such
    references. We replace them with a neutral phrasing so the validator
    doesn't fail, without hallucinating content.
    """
    if not text:
        return text
    import re

    pattern = re.compile(
        r"(?:see|in|from|as shown in|shown in)\s+(?:figure|fig\.|table|tab\.)\s*\d+",
        flags=re.IGNORECASE,
    )
    return pattern.sub("(referenced plot)", text)
