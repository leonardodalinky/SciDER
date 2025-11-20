import subprocess
from pathlib import Path

from pydantic import BaseModel

from ..core.utils import wrap_dict_to_toon
from .registry import register_tool, register_toolset_desc

register_toolset_desc("cursor", "Cursor CLI control toolset.")

WORKDIR = Path.home() / ".experiment_repos" / "current_repo"


class CursorResult(BaseModel):
    command: str
    returncode: int
    stdout: str
    stderr: str


def run_shell(cmd: str, cwd: Path = WORKDIR) -> CursorResult:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate()
    return CursorResult(
        command=cmd,
        returncode=proc.returncode,
        stdout=out,
        stderr=err,
    )


@register_tool(
    "cursor",
    {
        "type": "function",
        "function": {
            "name": "cursor_chat",
            "description": "Chat with Cursor CLI to get advice or refactoring suggestions (no direct code edit).",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message sent to Cursor chat.",
                    }
                },
                "required": ["message"],
            },
        },
    },
)
def cursor_chat_tool(message: str) -> str:
    try:
        cmd = f"cursor chat --message {message!r}"
        result = run_shell(cmd)
        return wrap_dict_to_toon(result.dict())
    except Exception as e:
        return wrap_dict_to_toon({"error": str(e)})


@register_tool(
    "cursor",
    {
        "type": "function",
        "function": {
            "name": "cursor_edit",
            "description": "Ask Cursor CLI to edit files based on a prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Instruction for Cursor to perform code edits.",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of files to limit editing scope.",
                    },
                },
                "required": ["message"],
            },
        },
    },
)
def cursor_edit_tool(message: str, files: list[str] | None = None) -> str:
    try:
        files_arg = " ".join(files) if files else ""
        cmd = f"cursor edit {files_arg} --message {message!r}"

        result = run_shell(cmd)
        return wrap_dict_to_toon(result.dict())
    except Exception as e:
        return wrap_dict_to_toon({"error": str(e)})


@register_tool(
    "cursor",
    {
        "type": "function",
        "function": {
            "name": "cursor_fix_tests",
            "description": "Let Cursor CLI attempt to fix failing tests.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
)
def cursor_fix_tests_tool() -> str:
    try:
        cmd = "cursor fix-tests"
        result = run_shell(cmd)
        return wrap_dict_to_toon(result.dict())
    except Exception as e:
        return wrap_dict_to_toon({"error": str(e)})
