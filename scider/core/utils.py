import json
import re
from typing import Type, TypeVar

import toon
from json_repair import repair_json
from pydantic import BaseModel

from .types import Message

T = TypeVar("T", bound=BaseModel)


def wrap_text_with_block(s: str, block_name: str) -> str:
    return f"```{block_name}\n{s}\n```"


def wrap_dict_to_toon(d: dict) -> str:
    s = toon.encode(d)
    if s == "null":
        raise ValueError("Failed to encode dict to TOON")
    return wrap_text_with_block(s, "toon")


def _normalize_toon_content(toon_content: str) -> str:
    """
    Normalize TOON content to fix common format issues.

    Fixes:
    1. Illegal key syntax like `authors[6]: value1,value2,...` -> `authors: [value1, value2, ...]`
    2. Handles comma-separated values in indexed keys
    3. Preserves proper TOON format for other lines
    """
    # Check if normalization is needed
    if not re.search(r"\w+\[\d+\]\s*:", toon_content):
        # No illegal syntax found, return as-is
        return toon_content

    lines = []

    for line in toon_content.splitlines():
        # Match illegal indexed key syntax: key[number]: value
        # Example: authors[6]: Fengyu She,Nan Wang,Hongfei Wu,...
        indexed_key_match = re.match(r"^(\w+)\[(\d+)\]\s*:\s*(.+)$", line)
        if indexed_key_match:
            key, index, value = indexed_key_match.groups()

            # Split comma-separated values
            # Handle both quoted and unquoted values
            values = []
            for v in value.split(","):
                v = v.strip()
                # Remove quotes if present
                if (v.startswith('"') and v.endswith('"')) or (
                    v.startswith("'") and v.endswith("'")
                ):
                    v = v[1:-1]
                if v:  # Only add non-empty values
                    values.append(v)

            # Convert to proper TOON list format
            if values:
                # Use YAML-style list format for better compatibility
                formatted_values = ", ".join(f'"{v}"' for v in values)
                lines.append(f"{key}: [{formatted_values}]")
            else:
                lines.append(f"{key}: []")
        else:
            # Regular line - preserve as-is
            lines.append(line)

    return "\n".join(lines)


def unwrap_dict_from_toon(toon_str: str) -> dict:
    """Parse a toon-formatted string back to a dictionary."""
    if isinstance(toon_str, dict):
        return toon_str

    if not isinstance(toon_str, str):
        raise TypeError(f"Expected str or dict, got {type(toon_str)}")
    match = re.search(
        r"(?:```\s*)?(?:toon\s*)?(.*)(?:```)?",
        toon_str,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:
        toon_content = match.group(1).strip()
    else:
        toon_content = toon_str.strip()

    if ":" not in toon_content:
        raise ValueError(
            "Invalid TOON content: no ':' found. " "Likely code block extraction failed."
        )

    # Normalize TOON content to fix common format issues
    # (e.g., illegal indexed key syntax like authors[6]: value)
    toon_content = _normalize_toon_content(toon_content)

    try:
        if hasattr(toon, "decode"):
            return toon.decode(toon_content)
        if hasattr(toon, "loads"):
            return toon.loads(toon_content)
        if hasattr(toon, "parse"):
            return toon.parse(toon_content)

        raise RuntimeError("toon library has no decode / loads / parse method")

    except Exception as e:
        logger = __import__("loguru", fromlist=["logger"]).logger
        logger.debug(f"Full TOON content: {toon_content}")
        raise ValueError(f"Failed to decode TOON: {e}") from e


def parse_json_from_llm_response(llm_response: str | Message, tgt_type: Type[T]) -> T:
    if isinstance(llm_response, Message):
        text = llm_response.content
    else:
        text = llm_response
    json_match = re.search(
        r"(?:```\s*)?(?:json\s*)?(.*)(?:```)?", text, flags=re.DOTALL | re.IGNORECASE
    )  # must find something, at least return the entire text
    if not json_match:
        raise ValueError("Failed to find JSON in LLM response")
    json_str = json_match.group(1).strip()
    json_str = repair_json(json_str)
    return tgt_type.model_validate_json(json_str)


def parse_markdown_from_llm_response(llm_response: str | Message) -> str:
    if isinstance(llm_response, Message):
        text = llm_response.content
    else:
        text = llm_response
    markdown_match = re.search(
        r"(?:```\s*)?(?:markdown\s*)?(.*)(?:```)?",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )  # must find something, at least return the entire text
    if not markdown_match:
        raise ValueError("Failed to find markdown in LLM response")
    markdown_str = markdown_match.group(1).strip()
    return markdown_str


def parse_json_from_text(text: str, tgt_type: Type[T] | None = None) -> T | object:
    json_match = re.search(
        r"(?:```\s*)?(?:json\s*)?(.*)(?:```)?", text, flags=re.DOTALL | re.IGNORECASE
    )  # must find something, at least return the entire text
    if not json_match:
        raise ValueError("Failed to find JSON in text")
    json_str = json_match.group(1).strip()
    json_str = repair_json(json_str)
    if tgt_type is not None:
        return tgt_type.model_validate_json(json_str)
    return json.loads(json_str)


def array_to_bullets(arr: list[str]) -> str:
    return "\n".join([f"- {s}" for s in arr])


_python_runtime_cache: str | None = None


def detect_python_runtime() -> str:
    """Detect the Python runtime environment (uv or plain python).

    Probes once and caches the result. Returns a short string suitable for
    injection into agent system context, e.g.:

        "python_runtime: uv (use `uv run python ...` and `uv add ...`)"
        "python_runtime: python (use `python ...` and `pip install ...`)"
    """
    global _python_runtime_cache
    if _python_runtime_cache is not None:
        return _python_runtime_cache

    import shutil
    import subprocess

    if shutil.which("uv"):
        try:
            result = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=5)
            version = result.stdout.strip() if result.returncode == 0 else "uv"
            _python_runtime_cache = (
                f"python_runtime: {version}\n"
                "  Use `uv run python script.py` to execute scripts (NOT `python script.py`).\n"
                "  Use `uv add <package>` to install packages (NOT `pip install`).\n"
                "  Use `uv run pytest` to run tests."
            )
        except Exception:
            _python_runtime_cache = (
                "python_runtime: uv (detected but version check failed)\n"
                "  Use `uv run python script.py` and `uv add <package>`."
            )
    else:
        _python_runtime_cache = (
            "python_runtime: python\n"
            "  Use `python script.py` to execute scripts.\n"
            "  Use `pip install <package>` to install packages."
        )

    return _python_runtime_cache


_gpu_runtime_cache: str | None = None


def detect_gpu_runtime() -> str:
    """Detect NVIDIA GPU availability via nvidia-smi.

    Probes once and caches the result. Returns a short string suitable for
    injection into agent system context.
    """
    global _gpu_runtime_cache
    if _gpu_runtime_cache is not None:
        return _gpu_runtime_cache

    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        _gpu_runtime_cache = "gpu: none detected (nvidia-smi not found). Use CPU-only code."
        return _gpu_runtime_cache

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            _gpu_runtime_cache = "gpu: nvidia-smi failed. Assume CPU-only."
            return _gpu_runtime_cache

        lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        if not lines:
            _gpu_runtime_cache = "gpu: none detected. Use CPU-only code."
            return _gpu_runtime_cache

        gpu_count = len(lines)
        gpu_descriptions = []
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            name = parts[0] if parts else "Unknown"
            raw_mem = parts[1] if len(parts) > 1 else ""
            if "GB10" in name:
                # GB10 uses Grace Blackwell unified memory; nvidia-smi reports [N/A].
                mem = "Hybrid 128 GiB (unified CPU+GPU memory)"
            elif raw_mem and raw_mem != "[N/A]":
                mem = f"{raw_mem} MiB"
            else:
                mem = "unknown memory"
            gpu_descriptions.append(f"  GPU {i}: {name} ({mem})")

        _gpu_runtime_cache = (
            f"gpu: {gpu_count} NVIDIA GPU(s) available\n"
            + "\n".join(gpu_descriptions)
            + "\n  PyTorch/TensorFlow can use CUDA. Prefer GPU when training models."
        )
    except Exception:
        _gpu_runtime_cache = "gpu: detection failed. Assume CPU-only."

    return _gpu_runtime_cache


def get_git_status(cwd: str | None = None, max_status_chars: int = 2000) -> str | None:
    """Get a git status snapshot for context injection.

    Returns a formatted string with branch, status, and recent commits,
    or None if the directory is not a git repo. Modeled after Claude Code's
    getGitStatus() (src/utils/git.ts).
    """
    import subprocess

    def _run(args: list[str]) -> str:
        try:
            r = subprocess.run(
                ["git", "--no-optional-locks"] + args,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=5,
            )
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    # Check if it's a git repo
    if not _run(["rev-parse", "--is-inside-work-tree"]):
        return None

    branch = _run(["branch", "--show-current"]) or _run(["rev-parse", "--short", "HEAD"])
    status = _run(["status", "--short"])
    log = _run(["log", "--oneline", "-n", "5"])
    user_name = _run(["config", "user.name"])

    if status and len(status) > max_status_chars:
        status = status[:max_status_chars] + "\n... (truncated)"

    parts = [
        "gitStatus: This is the git status at the start of the conversation. "
        "Note that this status is a snapshot in time, and will not update during the conversation.",
        f"Current branch: {branch}" if branch else None,
        f"Git user: {user_name}" if user_name else None,
        f"Status:\n{status}" if status else "Status: (clean)",
        f"Recent commits:\n{log}" if log else None,
    ]
    return "\n\n".join(p for p in parts if p)


def smart_truncate(text: str, max_length: int = 32000) -> str:
    """Truncate text if it exceeds max_length. Keep the head and tail, and adding in the middle."""
    if len(text) <= max_length:
        return text
    MIDDLE_TEXT = f"\n...[truncated about {len(text) - max_length} characters]...\n"
    half_length = (max_length - len(MIDDLE_TEXT)) // 2  # keep some buffer for the middle part
    return text[:half_length] + MIDDLE_TEXT + text[-half_length:]
