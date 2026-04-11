"""Tests for SCIDER.md project context loader."""

from __future__ import annotations

import pytest

from scider.core.scider_context import (
    _INSTRUCTION_HEADER,
    MAX_SCIDER_MD_CHARS,
    clear_cache,
    load_scider_md,
)


@pytest.fixture(autouse=True)
def _clear():
    clear_cache()
    yield
    clear_cache()


def test_loads_from_dot_scider_dir(tmp_path):
    (tmp_path / ".scider").mkdir()
    (tmp_path / ".scider" / "SCIDER.md").write_text("hello from .scider")
    result = load_scider_md(tmp_path)
    assert "hello from .scider" in result


def test_loads_from_root_scider_md(tmp_path):
    (tmp_path / "SCIDER.md").write_text("hello from root")
    result = load_scider_md(tmp_path)
    assert "hello from root" in result


def test_dot_scider_and_root_both_loaded(tmp_path):
    """Both .scider/SCIDER.md and SCIDER.md in the same dir are concatenated."""
    (tmp_path / ".scider").mkdir()
    (tmp_path / ".scider" / "SCIDER.md").write_text("from dotdir")
    (tmp_path / "SCIDER.md").write_text("from root")
    result = load_scider_md(tmp_path)
    assert "from dotdir" in result
    assert "from root" in result


def test_walk_up_concatenates(tmp_path):
    """Files from parent dir appear BEFORE child dir files (root-first)."""
    child = tmp_path / "project"
    child.mkdir()
    (tmp_path / "SCIDER.md").write_text("parent instructions")
    (child / "SCIDER.md").write_text("child instructions")
    result = load_scider_md(child)
    assert "parent instructions" in result
    assert "child instructions" in result
    # Parent should appear before child
    assert result.index("parent instructions") < result.index("child instructions")


def test_returns_none_when_missing(tmp_path):
    assert load_scider_md(tmp_path) is None


def test_skips_empty_file(tmp_path):
    (tmp_path / "SCIDER.md").write_text("")
    assert load_scider_md(tmp_path) is None


def test_truncates_large_content(tmp_path):
    content = "x" * (MAX_SCIDER_MD_CHARS + 1000)
    (tmp_path / "SCIDER.md").write_text(content)
    result = load_scider_md(tmp_path)
    assert result.endswith("[Truncated]")


def test_caches_result(tmp_path):
    (tmp_path / "SCIDER.md").write_text("cached")
    result1 = load_scider_md(tmp_path)
    (tmp_path / "SCIDER.md").write_text("changed")
    result2 = load_scider_md(tmp_path)
    assert result1 == result2  # still cached


def test_cache_cleared(tmp_path):
    (tmp_path / "SCIDER.md").write_text("v1")
    load_scider_md(tmp_path)
    clear_cache()
    (tmp_path / "SCIDER.md").write_text("v2")
    result = load_scider_md(tmp_path)
    assert "v2" in result


def test_rules_dir_loaded(tmp_path):
    """All .md files in .scider/rules/ are included."""
    rules = tmp_path / ".scider" / "rules"
    rules.mkdir(parents=True)
    (rules / "coding.md").write_text("use uv")
    (rules / "style.md").write_text("black format")
    result = load_scider_md(tmp_path)
    assert "use uv" in result
    assert "black format" in result


def test_none_workspace_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".scider").mkdir()
    (tmp_path / ".scider" / "SCIDER.md").write_text("from cwd")
    result = load_scider_md(None)
    assert "from cwd" in result


def test_instruction_header_present(tmp_path):
    (tmp_path / "SCIDER.md").write_text("some content")
    result = load_scider_md(tmp_path)
    assert result.startswith(_INSTRUCTION_HEADER)
