"""FileEditTool — exact string replacement in files.

Modeled after Claude Code's FileEditTool. Uses precise old_string → new_string
replacement instead of unified diff. Simpler and more reliable for LLMs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext

DESCRIPTION = """\
Performs exact string replacements in files.

Usage:
- You must use the Read tool at least once before editing a file.
- ALWAYS prefer editing existing files. NEVER write new files unless explicitly required.
- The edit will FAIL if `old_string` is not unique in the file. Either provide \
a larger string with more surrounding context to make it unique, or use \
`replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file.
"""


class FileEditInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to modify")
    old_string: str = Field(description="The text to replace")
    new_string: str = Field(
        description="The text to replace it with (must be different from old_string)"
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_string (default false)",
    )


class FileEditTool(BaseTool):
    name = "FileEdit"
    description = DESCRIPTION
    input_schema = FileEditInput

    def call(
        self,
        context: ToolContext,
        *,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        path = Path(file_path)

        # Empty old_string on nonexistent file = create new file
        if old_string == "" and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_string, encoding="utf-8")
            return f"Created new file: {file_path}"

        # File must exist for non-empty old_string
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."

        # No-op check
        if old_string == new_string:
            return "Error: old_string and new_string are exactly the same. No changes to make."

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file '{file_path}': {e}"

        # Count occurrences
        count = content.count(old_string)

        if count == 0:
            # Try case-insensitive hint
            lower_count = content.lower().count(old_string.lower())
            if lower_count > 0:
                return (
                    f"Error: old_string not found in {file_path} (exact match). "
                    f"Found {lower_count} case-insensitive match(es). "
                    f"Check your casing."
                )
            return (
                f"Error: old_string not found in {file_path}. "
                f"Make sure the string matches exactly, including whitespace and indentation. "
                f"If you've tried multiple times, consider using FileWrite to rewrite the entire file."
            )

        if count > 1 and not replace_all:
            return (
                f"Error: old_string appears {count} times in {file_path}. "
                f"Either provide a larger string with more surrounding context "
                f"to make it unique, or set replace_all=true to replace all occurrences."
            )

        # Perform replacement
        if replace_all:
            new_content = content.replace(old_string, new_string)
            replacements = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            replacements = 1

        # Write back
        try:
            path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return f"Error writing file '{file_path}': {e}"

        return (
            f"Successfully edited {file_path} "
            f"({replacements} replacement{'s' if replacements > 1 else ''} made)."
        )
