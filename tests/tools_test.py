"""Tests for tool system — registration, validation, prompts."""

from scider.tools import ToolRegistry
from scider.tools.base import BaseTool, ToolContext
from scider.tools.registry import get_tool_prompts


class TestToolRegistry:
    def test_all_tools_registered(self):
        tools = ToolRegistry.get_all_tools()
        names = {t.name for t in tools.values()}
        # Core tools must be present
        for expected in ["Read", "FileEdit", "FileWrite", "Bash", "Glob", "Grep"]:
            assert expected in names, f"{expected} not registered"

    def test_agent_tools_registered(self):
        tools = ToolRegistry.get_all_tools()
        names = {t.name for t in tools.values()}
        for expected in ["Agent", "AskUserQuestion", "TodoWrite"]:
            assert expected in names, f"{expected} not registered"

    def test_task_tools_registered(self):
        tools = ToolRegistry.get_all_tools()
        names = {t.name for t in tools.values()}
        for expected in ["TaskOutput", "TaskStop"]:
            assert expected in names, f"{expected} not registered"

    def test_get_tools_by_names(self):
        tools = ToolRegistry.get_tools_by_names(["Read", "Glob", "NonExistent"])
        assert "Read" in {t.name for t in tools.values()}
        assert "Glob" in {t.name for t in tools.values()}
        assert len(tools) == 2  # NonExistent silently skipped


class TestToolPrompts:
    def test_core_tools_have_prompts(self):
        prompts = get_tool_prompts(["Read", "FileEdit", "FileWrite", "Bash", "Glob", "Grep"])
        assert len(prompts) == 6

    def test_no_prompt_returns_empty(self):
        prompts = get_tool_prompts(["NonExistentTool"])
        assert len(prompts) == 0

    def test_prompt_content(self):
        prompts = get_tool_prompts(["Bash"])
        assert len(prompts) == 1
        assert "run_in_background" in prompts[0]


class TestToolValidation:
    def test_glob_missing_pattern_raises(self):
        """Validation error should raise ValueError (caught by _execute_tool_calls)."""
        wrapper = ToolRegistry.instance().tools["Glob"].func
        ctx = ToolContext(agent_name="test")
        try:
            wrapper(path=".", __tool_context__=ctx)  # missing pattern
            assert False, "Should have raised"
        except ValueError as e:
            assert "pattern" in str(e).lower()

    def test_glob_valid_input(self):
        wrapper = ToolRegistry.instance().tools["Glob"].func
        ctx = ToolContext(agent_name="test")
        result = wrapper(pattern="*.py", path=".", __tool_context__=ctx)
        assert isinstance(result, str)

    def test_read_nonexistent_file(self):
        wrapper = ToolRegistry.instance().tools["Read"].func
        ctx = ToolContext(agent_name="test")
        result = wrapper(file_path="/nonexistent/path.txt", __tool_context__=ctx)
        assert "error" in result.lower() or "not exist" in result.lower()
