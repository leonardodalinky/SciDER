"""Tests for Read tool error output when the target file is missing.

The error message should help the agent recover by either:
- Listing the parent directory's contents (bounded in size), or
- Pointing to the nearest existing ancestor when the parent itself is gone.
"""

from __future__ import annotations

from pathlib import Path

from scider.tools.base import ToolContext
from scider.tools.fs.read_file import ReadFileTool, _parent_dir_hint


def _ctx() -> ToolContext:
    return ToolContext(agent_name="test")


class TestMissingFile:
    def test_lists_parent_when_parent_exists(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.csv").write_text("y")
        (tmp_path / "sub").mkdir()

        result = ReadFileTool().call(_ctx(), file_path=str(tmp_path / "missing.json"))

        assert "does not exist" in result
        assert "a.txt" in result and "b.csv" in result
        assert "sub/" in result  # directories get trailing slash

    def test_flags_missing_parent_directory(self, tmp_path: Path):
        target = tmp_path / "nope" / "deeper" / "file.txt"
        result = ReadFileTool().call(_ctx(), file_path=str(target))
        assert "does not exist" in result
        assert "Parent directory" in result
        # nearest existing ancestor is tmp_path itself
        assert str(tmp_path) in result

    def test_case_mismatch_still_gets_suggestion(self, tmp_path: Path):
        (tmp_path / "Data.CSV").write_text("x")
        result = ReadFileTool().call(_ctx(), file_path=str(tmp_path / "data.csv"))
        assert "Did you mean" in result
        assert "Data.CSV" in result


class TestParentDirHint:
    def test_truncates_large_listings(self, tmp_path: Path):
        # Create 200 files — should cap entries + char count
        for i in range(200):
            (tmp_path / f"file_{i:03d}.txt").write_text("")
        hint = _parent_dir_hint(str(tmp_path / "missing.dat"), max_chars=800, max_entries=30)
        assert len(hint) <= 900  # max_chars + small slack for truncation marker
        assert "more not shown" in hint or "truncated" in hint

    def test_empty_parent(self, tmp_path: Path):
        hint = _parent_dir_hint(str(tmp_path / "nothing.txt"))
        assert "is empty" in hint

    def test_missing_parent_walks_up(self, tmp_path: Path):
        hint = _parent_dir_hint(str(tmp_path / "x" / "y" / "z.txt"))
        assert "does not exist" in hint
        assert str(tmp_path) in hint
