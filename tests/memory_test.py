"""Tests for scider.core.memory — memory index loading and truncation."""

from pathlib import Path
from unittest.mock import patch

from scider.core.memory import (
    MAX_INDEX_BYTES,
    MAX_INDEX_LINES,
    _read_and_truncate,
    build_memory_prompt_section,
    load_memory_index,
)


def _mock_walk(dirs):
    """Return a patch that makes walk_up_dirs() return the given dirs."""
    return patch("scider.core.scider_context.walk_up_dirs", return_value=[Path(d) for d in dirs])


class TestReadAndTruncate:
    def test_normal_index(self, tmp_path):
        index = tmp_path / "MEMORY.md"
        index.write_text("- [Test](test.md) — A test memory\n")
        result = _read_and_truncate(index, str(tmp_path))
        assert "Test" in result
        assert "Memory directory" in result

    def test_empty_index(self, tmp_path):
        index = tmp_path / "MEMORY.md"
        index.write_text("")
        result = _read_and_truncate(index, str(tmp_path))
        assert result == ""

    def test_line_truncation(self, tmp_path):
        index = tmp_path / "MEMORY.md"
        lines = [f"- [Item {i}](item_{i}.md) — Description {i}" for i in range(300)]
        index.write_text("\n".join(lines))
        result = _read_and_truncate(index, str(tmp_path))
        assert "WARNING" in result
        assert "truncated" in result
        content_lines = [l for l in result.split("\n") if l.startswith("- [")]
        assert len(content_lines) <= MAX_INDEX_LINES

    def test_byte_truncation(self, tmp_path):
        index = tmp_path / "MEMORY.md"
        long_line = "- [X](x.md) — " + "A" * 30_000
        index.write_text(long_line)
        result = _read_and_truncate(index, str(tmp_path))
        assert "WARNING" in result
        assert "truncated" in result


class TestLoadMemoryIndex:
    def test_no_memory_dir(self, tmp_path):
        with _mock_walk([tmp_path / "nonexistent"]):
            result = load_memory_index()
            assert result == ""

    def test_project_level(self, tmp_path):
        mem_dir = tmp_path / ".scider" / "memory"
        mem_dir.mkdir(parents=True)
        (mem_dir / "MEMORY.md").write_text("- [Test](test.md) — Project memory\n")

        with _mock_walk([tmp_path]):
            result = load_memory_index()
            assert "Project memory" in result


class TestBuildMemoryPromptSection:
    def test_with_index(self, tmp_path):
        mem_dir = tmp_path / ".scider" / "memory"
        mem_dir.mkdir(parents=True)
        (mem_dir / "MEMORY.md").write_text("- [Pref](pref.md) — User preference\n")

        with _mock_walk([tmp_path]):
            section = build_memory_prompt_section()
            assert "User preference" in section
            assert "Writing memories" in section
            assert "What NOT to save" in section

    def test_without_index(self, tmp_path):
        with _mock_walk([tmp_path / "nope"]):
            section = build_memory_prompt_section()
            assert section == ""
