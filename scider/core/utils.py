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


# Non-greedy fenced-block regex — finds `` ```lang\n...\n``` `` correctly even
# when the LLM mixes prose with multiple code blocks. ``findall`` + last-match
# lets us honor the final block (where LLMs usually put the definitive answer).
_FENCED_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_-]*)\s*([\s\S]*?)```")


def _extract_fenced_blocks(text: str, lang: str | None = None) -> list[str]:
    """Return every fenced code block's body, optionally filtered by language.

    When ``lang`` is given we accept:
      - exact-match language tag (``json`` / ``markdown``)
      - empty tag (untagged ```...``` blocks are common from LLMs)
    """
    blocks: list[str] = []
    for tag, body in _FENCED_BLOCK_RE.findall(text):
        if lang is None or tag.lower() == lang.lower() or tag == "":
            blocks.append(body.strip())
    return blocks


def _coerce_json_str(raw: str) -> str:
    """Return a JSON string parseable by ``json.loads``, using ``repair_json``
    only if strict parsing fails."""
    try:
        json.loads(raw)
        return raw
    except Exception:
        return repair_json(raw)


def parse_json_from_llm_response(
    llm_response: str | Message, tgt_type: Type[T] | None = None
) -> T | object:
    """Parse JSON from an LLM response (str or ``Message``).

    Thin wrapper around :func:`parse_json_from_text` that accepts a ``Message``
    and extracts ``.content``. When ``tgt_type`` is ``None`` the raw dict/list
    is returned; otherwise the result is validated as that Pydantic model.
    """
    text = llm_response.content if isinstance(llm_response, Message) else llm_response
    return parse_json_from_text(text or "", tgt_type=tgt_type)


def parse_markdown_from_llm_response(llm_response: str | Message) -> str:
    """Strip a single surrounding ```markdown`` ``` (or plain ``` ``` ```) fence
    if present; otherwise return the text as-is. Intended for extracting the
    model's narrative output when it decides to wrap the whole response.
    """
    text = llm_response.content if isinstance(llm_response, Message) else llm_response
    if text is None:
        raise ValueError("Failed to find markdown in LLM response: empty text")

    blocks = _extract_fenced_blocks(text, lang="markdown")
    if blocks:
        # Use the LAST fenced block (matches "prose preamble, then final answer" shape).
        return blocks[-1]
    return text.strip()


def parse_json_from_text(text: str, tgt_type: Type[T] | None = None) -> T | object:
    if not text or not text.strip():
        raise ValueError("Failed to find JSON in text: empty")

    # Prefer the LAST fenced JSON block; fall back to the whole stripped body.
    candidates = _extract_fenced_blocks(text, lang="json")
    candidates.reverse()
    candidates.append(text.strip())

    last_err: Exception | None = None
    for snippet in candidates:
        if not snippet:
            continue
        try:
            parsed_str = _coerce_json_str(snippet)
            if tgt_type is not None:
                return tgt_type.model_validate_json(parsed_str)
            return json.loads(parsed_str)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"Failed to parse JSON from text: {last_err}")


def array_to_bullets(arr: list[str]) -> str:
    return "\n".join([f"- {s}" for s in arr])


# Cache keyed by WorkspaceInitConfig.cache_key() so different configs don't
# collide. Kept as a dict rather than lru_cache because the fn is called from
# many call sites and the key space is tiny (~3 distinct configs in practice).
_python_runtime_cache: dict[tuple, str] = {}


def detect_python_runtime(config=None) -> str:
    """Detect the Python runtime environment string for agent system context.

    The returned text depends on the caller's `WorkspaceInitConfig`:

    - `env_manager="uv"` (default): tells the agent to use `uv run python` /
      `uv add`, and probes the installed uv version.
    - `env_manager="python"` with `venv_path=None`: bare python + `pip install`.
    - `env_manager="python"` with `venv_path` set: bare python in a prepared
      venv; forbids `uv add` / `pip install` so the agent doesn't try to install
      into a pre-built environment.

    The result is cached per-config so repeated calls are free.
    """
    from scider.core.code_env import DEFAULT_WORKSPACE_INIT_CONFIG, WorkspaceInitConfig

    if config is None:
        config = DEFAULT_WORKSPACE_INIT_CONFIG
    elif not isinstance(config, WorkspaceInitConfig):
        raise TypeError(f"config must be WorkspaceInitConfig or None, got {type(config).__name__}")

    key = config.cache_key()
    cached = _python_runtime_cache.get(key)
    if cached is not None:
        return cached

    import shutil
    import subprocess

    if config.env_manager == "uv":
        if shutil.which("uv"):
            try:
                result = subprocess.run(
                    ["uv", "--version"], capture_output=True, text=True, timeout=5
                )
                version = result.stdout.strip() if result.returncode == 0 else "uv"
                text = (
                    f"python_runtime: {version}\n"
                    "  Use `uv run python script.py` to execute scripts (NOT `python script.py`).\n"
                    "  Use `uv add <package>` to install packages (NOT `pip install`).\n"
                    "  Use `uv run pytest` to run tests."
                )
            except Exception:
                text = (
                    "python_runtime: uv (detected but version check failed)\n"
                    "  Use `uv run python script.py` and `uv add <package>`."
                )
        else:
            # Config asked for uv but it's not installed; fall back to python text.
            text = (
                "python_runtime: python (uv requested but not found)\n"
                "  Use `python script.py` to execute scripts.\n"
                "  Use `pip install <package>` to install packages."
            )
    else:  # env_manager == "python"
        if config.venv_path is not None:
            text = (
                f"python_runtime: python (prebuilt venv at {config.venv_path})\n"
                "  Use `python script.py` to execute scripts — the correct "
                "interpreter is already on PATH.\n"
                "  Do NOT run `uv add` / `pip install` / `uv pip install`: all "
                "dependencies are preinstalled in the venv."
            )
        else:
            text = (
                "python_runtime: python\n"
                "  Use `python script.py` to execute scripts.\n"
                "  Use `pip install <package>` to install packages."
            )

    _python_runtime_cache[key] = text
    return text


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
