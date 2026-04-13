"""Tests for the paper workspace bootstrap helpers.

These exercise the pure string / file-copy layer that bridges SciDER's
summaries into PaperOrchestra's strict ``inputs/`` layout. No LLM is
involved; everything should be deterministic and fast.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scider.workflows.paper_bootstrap import (
    DEFAULT_CONFERENCE_GUIDELINES,
    DEFAULT_TEMPLATE_DIR,
    DEFAULT_TEMPLATE_TEX,
    bootstrap_paper_workspace,
    build_experimental_log,
    build_idea_from_ideation,
    build_sparse_idea_from_query,
)


class TestBootstrapPaperWorkspace:
    def test_creates_full_inputs_layout_with_defaults(self, tmp_path: Path):
        paper_ws = tmp_path / "paper"
        bootstrap_paper_workspace(
            paper_ws,
            idea_summary="## Problem Statement\nX\n",
            experimental_log="# Experimental Log\n## 1. Experimental Setup\nY\n",
        )
        inputs = paper_ws / "inputs"
        assert (inputs / "idea.md").is_file()
        assert (inputs / "experimental_log.md").is_file()
        assert (inputs / "template.tex").is_file()
        assert (inputs / "conference_guidelines.md").is_file()
        assert (inputs / "figures").is_dir()
        # Defaults came from the bundled files
        assert (
            inputs / "template.tex"
        ).read_text().strip() == DEFAULT_TEMPLATE_TEX.read_text().strip()
        assert (
            inputs / "conference_guidelines.md"
        ).read_text().strip() == DEFAULT_CONFERENCE_GUIDELINES.read_text().strip()

    def test_user_template_override_wins(self, tmp_path: Path):
        paper_ws = tmp_path / "paper"
        bootstrap_paper_workspace(
            paper_ws,
            idea_summary="idea",
            experimental_log="log",
            template_tex="\\documentclass{article}\n\\begin{document}\nCUSTOM\n\\end{document}",
            conference_guidelines="# Custom venue\nPage limit: 4 pages",
        )
        assert "CUSTOM" in (paper_ws / "inputs" / "template.tex").read_text()
        assert "Custom venue" in (paper_ws / "inputs" / "conference_guidelines.md").read_text()

    def test_figures_copied_from_src(self, tmp_path: Path):
        src = tmp_path / "src_imgs"
        src.mkdir()
        # Mix of supported and unsupported extensions
        (src / "plot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)
        (src / "table.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)
        (src / "notes.txt").write_text("not an image")  # should be skipped

        paper_ws = tmp_path / "paper"
        bootstrap_paper_workspace(
            paper_ws,
            idea_summary="idea",
            experimental_log="log",
            figures_src=src,
        )
        figs = paper_ws / "inputs" / "figures"
        assert (figs / "plot.png").is_file()
        assert (figs / "table.jpg").is_file()
        assert not (figs / "notes.txt").exists()

    def test_missing_figures_src_is_not_an_error(self, tmp_path: Path):
        paper_ws = tmp_path / "paper"
        # figures_src pointing at a nonexistent directory must not crash
        bootstrap_paper_workspace(
            paper_ws,
            idea_summary="idea",
            experimental_log="log",
            figures_src=tmp_path / "does_not_exist",
        )
        assert (paper_ws / "inputs" / "figures").is_dir()
        assert not list((paper_ws / "inputs" / "figures").iterdir())

    def test_bundled_simple_dir_has_required_files(self):
        """The bundled default template directory must be well-formed so
        the default fallback path works end-to-end."""
        assert DEFAULT_TEMPLATE_DIR.is_dir()
        assert (DEFAULT_TEMPLATE_DIR / "template.tex").is_file()
        assert (DEFAULT_TEMPLATE_DIR / "guidelines.md").is_file()
        # Back-compat aliases resolve under the directory
        assert DEFAULT_TEMPLATE_TEX.is_file()
        assert DEFAULT_CONFERENCE_GUIDELINES.is_file()


class TestBootstrapTemplateDir:
    """Exercise the ``template_dir`` path — directory with aux files."""

    def _make_template_dir(self, root: Path) -> Path:
        tdir = root / "my_template"
        tdir.mkdir()
        (tdir / "template.tex").write_text(
            "\\documentclass{article}\n\\usepackage{mystyle}\n\\begin{document}MINE\\end{document}"
        )
        (tdir / "guidelines.md").write_text("# My venue\n6 pages.\n")
        # Auxiliary files
        (tdir / "mystyle.sty").write_text("% mystyle.sty — dummy\n")
        (tdir / "refs.bst").write_text("% refs.bst — dummy\n")
        (tdir / "priors.bib").write_text("@misc{foo2025,title={Foo}}\n")
        # A subdirectory that must be ignored
        (tdir / "subdir").mkdir()
        (tdir / "subdir" / "should_not_copy.txt").write_text("ignored")
        return tdir

    def test_template_dir_copies_template_and_guidelines(self, tmp_path: Path):
        tdir = self._make_template_dir(tmp_path)
        paper_ws = tmp_path / "paper"
        bootstrap_paper_workspace(
            paper_ws,
            idea_summary="idea",
            experimental_log="log",
            template_dir=tdir,
        )
        inputs = paper_ws / "inputs"
        assert "MINE" in (inputs / "template.tex").read_text()
        assert "My venue" in (inputs / "conference_guidelines.md").read_text()

    def test_template_dir_copies_aux_files(self, tmp_path: Path):
        tdir = self._make_template_dir(tmp_path)
        paper_ws = tmp_path / "paper"
        bootstrap_paper_workspace(
            paper_ws,
            idea_summary="idea",
            experimental_log="log",
            template_dir=tdir,
        )
        inputs = paper_ws / "inputs"
        assert (inputs / "mystyle.sty").is_file()
        assert (inputs / "refs.bst").is_file()
        assert (inputs / "priors.bib").is_file()
        assert "mystyle.sty — dummy" in (inputs / "mystyle.sty").read_text()

    def test_template_dir_skips_subdirectories(self, tmp_path: Path):
        tdir = self._make_template_dir(tmp_path)
        paper_ws = tmp_path / "paper"
        bootstrap_paper_workspace(
            paper_ws,
            idea_summary="idea",
            experimental_log="log",
            template_dir=tdir,
        )
        # Subdirectory contents must not land in inputs/
        assert not (paper_ws / "inputs" / "subdir").exists()
        assert not (paper_ws / "inputs" / "should_not_copy.txt").exists()

    def test_template_dir_missing_template_tex_raises(self, tmp_path: Path):
        tdir = tmp_path / "bad_template"
        tdir.mkdir()
        (tdir / "guidelines.md").write_text("# venue")
        with pytest.raises(FileNotFoundError, match="template.tex"):
            bootstrap_paper_workspace(
                tmp_path / "paper",
                idea_summary="idea",
                experimental_log="log",
                template_dir=tdir,
            )

    def test_template_dir_missing_guidelines_raises(self, tmp_path: Path):
        tdir = tmp_path / "bad_template"
        tdir.mkdir()
        (tdir / "template.tex").write_text("\\documentclass{article}\n")
        with pytest.raises(FileNotFoundError, match="guidelines.md"):
            bootstrap_paper_workspace(
                tmp_path / "paper",
                idea_summary="idea",
                experimental_log="log",
                template_dir=tdir,
            )

    def test_nonexistent_template_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Template directory not found"):
            bootstrap_paper_workspace(
                tmp_path / "paper",
                idea_summary="idea",
                experimental_log="log",
                template_dir=tmp_path / "does_not_exist",
            )

    def test_string_overrides_win_over_template_dir(self, tmp_path: Path):
        """Explicit ``template_tex`` / ``conference_guidelines`` strings
        should replace whatever was copied from the template directory."""
        tdir = self._make_template_dir(tmp_path)
        paper_ws = tmp_path / "paper"
        bootstrap_paper_workspace(
            paper_ws,
            idea_summary="idea",
            experimental_log="log",
            template_dir=tdir,
            template_tex="\\documentclass{article}\\begin{document}OVERRIDE\\end{document}",
            conference_guidelines="# Override venue\n",
        )
        inputs = paper_ws / "inputs"
        # Strings won
        assert "OVERRIDE" in (inputs / "template.tex").read_text()
        assert "Override venue" in (inputs / "conference_guidelines.md").read_text()
        # But aux files from the directory must still have been copied
        assert (inputs / "mystyle.sty").is_file()
        assert (inputs / "refs.bst").is_file()


class TestBuildSparseIdea:
    def test_contains_required_headings(self):
        md = build_sparse_idea_from_query("make LLMs faster")
        for heading in (
            "Problem Statement",
            "Core Hypothesis",
            "Proposed Methodology",
            "Expected Contribution",
        ):
            assert re.search(
                rf"^##\s+{re.escape(heading)}", md, re.M
            ), f"missing heading: {heading}\n{md}"

    def test_query_appears_in_problem_statement(self):
        md = build_sparse_idea_from_query("make LLMs faster")
        assert "make LLMs faster" in md

    def test_empty_query_does_not_crash(self):
        md = build_sparse_idea_from_query("")
        assert "Problem Statement" in md
        assert "no user query provided" in md


class TestBuildIdeaFromIdeation:
    def test_uses_selected_idea_when_index_given(self):
        ideas = [
            {"title": "First", "description": "desc1", "novelty_score": 3.0},
            {"title": "Second", "description": "desc2", "novelty_score": 9.0},
        ]
        md = build_idea_from_ideation(ideas, selected_index=0, user_query="q")
        assert "**First**" in md
        assert "desc1" in md
        assert "Second" not in md or "desc2" not in md  # not the chosen one

    def test_picks_highest_scored_when_no_selection(self):
        ideas = [
            {"title": "Low", "description": "d1", "novelty_score": 2.0},
            {"title": "High", "description": "d2", "novelty_score": 8.5},
            {"title": "Mid", "description": "d3", "novelty_score": 5.0},
        ]
        md = build_idea_from_ideation(ideas, selected_index=None, user_query="q")
        assert "**High**" in md
        assert "d2" in md

    def test_falls_back_to_sparse_on_empty_ideas(self):
        md = build_idea_from_ideation([], selected_index=None, user_query="alpha beta")
        assert "Problem Statement" in md
        assert "alpha beta" in md

    def test_invalid_selection_index_falls_back_to_top_score(self):
        ideas = [{"title": "Only", "description": "d", "novelty_score": 1.0}]
        md = build_idea_from_ideation(ideas, selected_index=99, user_query="q")
        assert "Only" in md


class TestBuildExperimentalLog:
    def test_three_required_sections_present(self):
        log = build_experimental_log(
            data_summary="## Data Structure\n- 1000 rows, 5 cols",
            experiment_summary="## Methodology\nTrained SVR.\n\n## Results\n| metric | value |\n|---|---|\n| mse | 0.42 |\n\n## Analysis\nConverged.",
            user_query="predict prices",
        )
        assert re.search(r"^##\s+1\.\s+Experimental Setup", log, re.M)
        assert re.search(r"^##\s+2\.\s+Raw Numeric Data", log, re.M)
        assert re.search(r"^##\s+3\.\s+Qualitative Observations", log, re.M)

    def test_setup_section_includes_user_query_and_data_structure(self):
        log = build_experimental_log(
            data_summary="## Data Structure\nCSV with 1000 rows",
            experiment_summary="",
            user_query="predict X from Y",
        )
        assert "predict X from Y" in log
        assert "1000 rows" in log

    def test_raw_numeric_extracts_results_section(self):
        log = build_experimental_log(
            data_summary="",
            experiment_summary="## Results\n| a | b |\n|---|---|\n| 1 | 2 |",
            user_query="q",
        )
        # The Results markdown table must appear verbatim
        assert "| a | b |" in log
        assert "| 1 | 2 |" in log

    def test_raw_numeric_falls_back_to_all_tables_when_no_results_heading(self):
        log = build_experimental_log(
            data_summary="",
            experiment_summary="No heading here but a table:\n| x | y |\n|---|---|\n| 3 | 4 |",
            user_query="q",
        )
        assert "| x | y |" in log

    def test_qualitative_combines_analysis_and_conclusions(self):
        log = build_experimental_log(
            data_summary="",
            experiment_summary=(
                "## Analysis\nThe loss dropped.\n\n## Conclusions\nMethod A outperformed B.\n"
            ),
            user_query="q",
        )
        assert "loss dropped" in log
        assert "Method A outperformed B" in log

    def test_all_sections_nonempty_even_with_blank_inputs(self):
        """PaperOrchestra validator warns on empty Setup/Raw Numeric sections —
        our placeholder ``(none recorded)`` should keep them non-empty."""
        log = build_experimental_log(
            data_summary="",
            experiment_summary="",
            user_query="",
        )
        # Setup still gets a user_query line even if empty → but we pass "", so
        # check that the placeholder kicks in for Raw Numeric and Qualitative.
        assert "(none recorded)" in log

    def test_strips_figure_table_references(self):
        log = build_experimental_log(
            data_summary="",
            experiment_summary="## Analysis\nAs shown in Figure 2, see Table 3 for details.",
            user_query="q",
        )
        # The phrase "Figure 2" must not survive verbatim (anti-leakage).
        assert "Figure 2" not in log
        assert "Table 3" not in log


class TestEndToEndValidator:
    """Bootstrap → run PaperOrchestra's own validator script → expect non-fatal exit.

    We don't require the validator to be available at import time (it's a
    plain Python script under `.scider/skills/`). If it's present we exec it
    with ``subprocess`` and check the exit code.
    """

    @pytest.fixture
    def validator_path(self) -> Path | None:
        candidate = Path(
            "/home/link/github/SciEvo/.scider/skills/paper-orchestra/scripts/validate_inputs.py"
        )
        return candidate if candidate.is_file() else None

    def test_bootstrapped_workspace_passes_validator(
        self, tmp_path: Path, validator_path: Path | None
    ):
        if validator_path is None:
            pytest.skip("paper-orchestra/scripts/validate_inputs.py not present")

        paper_ws = tmp_path / "paper"
        bootstrap_paper_workspace(
            paper_ws,
            idea_summary=build_sparse_idea_from_query("toy problem: predict sinusoids"),
            experimental_log=build_experimental_log(
                data_summary="## Data Structure\n- 500 rows, 4 columns",
                experiment_summary=(
                    "## Methodology\nMLP with 2 hidden layers.\n\n"
                    "## Results\n| model | mse |\n|---|---|\n| MLP | 0.05 |\n\n"
                    "## Analysis\nLoss converged after 100 epochs."
                ),
                user_query="fit a regressor",
            ),
        )

        import subprocess

        result = subprocess.run(
            ["python", str(validator_path), "--workspace", str(paper_ws)],
            capture_output=True,
            text=True,
        )
        # Exit code 0 means no fatal errors. Warnings (stderr) are allowed.
        assert (
            result.returncode == 0
        ), f"validate_inputs.py failed:\nstdout={result.stdout}\nstderr={result.stderr}"
