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
        # Should have at most MAX_INDEX_LINES content lines + header + warning
        content_lines = [l for l in result.split("\n") if l.startswith("- [")]
        assert len(content_lines) <= MAX_INDEX_LINES

    def test_byte_truncation(self, tmp_path):
        index = tmp_path / "MEMORY.md"
        # Create a file with few lines but very long content
        long_line = "- [X](x.md) — " + "A" * 30_000
        index.write_text(long_line)
        result = _read_and_truncate(index, str(tmp_path))
        assert "WARNING" in result
        assert "truncated" in result


class TestLoadMemoryIndex:
    def test_no_memory_dir(self, tmp_path):
        with patch("scider.core.memory._PROJECT_MEMORY_DIR", str(tmp_path / "nonexistent")):
            with patch("scider.core.memory._USER_MEMORY_DIR", tmp_path / "also_nonexistent"):
                result = load_memory_index()
                assert result == ""

    def test_project_level(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("- [Test](test.md) — Project memory\n")

        with patch("scider.core.memory._PROJECT_MEMORY_DIR", str(mem_dir)):
            result = load_memory_index()
            assert "Project memory" in result


class TestBuildMemoryPromptSection:
    def test_with_index(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("- [Pref](pref.md) — User preference\n")

        with patch("scider.core.memory._PROJECT_MEMORY_DIR", str(mem_dir)):
            section = build_memory_prompt_section()
            assert "User preference" in section
            assert "Writing memories" in section  # guidance present
            assert "What NOT to save" in section

    def test_without_index(self, tmp_path):
        with patch("scider.core.memory._PROJECT_MEMORY_DIR", str(tmp_path / "nope")):
            with patch("scider.core.memory._USER_MEMORY_DIR", tmp_path / "also_nope"):
                section = build_memory_prompt_section()
                assert section == ""
