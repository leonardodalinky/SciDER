"""
Base classes for the new-style tool system.

BaseTool provides:
- Pydantic-based input schema (auto-generates JSON schema + validates input)
- Explicit ToolContext (replaces inspect.signature magic injection)
- Dynamic permission methods: is_read_only(input), check_permissions(input, context)
- Optional tool-specific prompt co-located with tool code
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..core.permissions import PermissionResult
    from ..core.types import HistoryState


@dataclass
class ToolContext:
    """Explicit context passed to every BaseTool.call() invocation.

    Replaces the old inspect.signature-based magic injection of agent_state and ctx.
    """

    agent_name: str
    agent_state: HistoryState | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolImage:
    """A single image attached to a tool result.

    ``data`` is pure base64 (no ``data:...`` URL prefix).
    """

    media_type: str  # "image/png" | "image/jpeg" | "image/webp" | "image/gif"
    data: str


@dataclass
class ToolResult:
    """Structured tool result carrying both text and optional images.

    Tools that only return text can still return a plain ``str`` from
    ``call()``. Only tools that need to attach images (e.g. Read on a PNG)
    need to return ``ToolResult``.
    """

    text: str
    images: list[ToolImage] = field(default_factory=list)


class BaseTool(ABC):
    """Base class for all new-style tools.

    Permission model (following Claude Code):
    - is_read_only(kwargs) — dynamic, per-call check. Returns True if this
      specific invocation only reads (no side effects). Used by plan mode
      to filter tools, and by the permission system.
    - check_permissions(kwargs, context) — custom permission logic per call.
      Returns PermissionResult (allow/deny). Called before execution.

    Subclasses override these methods to implement input-dependent logic.
    For example, BashTool checks the command string to decide if it's read-only.
    """

    # --- Class-level declarations (set by each subclass) ---
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]

    # Max result size in chars before Level 1 persist kicks in.
    # Set to float("inf") for tools whose output should never be persisted.
    max_result_size_chars: ClassVar[float] = 50_000

    # Optional tool-specific prompt appended to system prompt
    prompt: ClassVar[str | None] = None

    @abstractmethod
    def call(self, context: ToolContext, **kwargs) -> "str | ToolResult":
        """Execute the tool. kwargs come from validated input_schema.

        Return a plain ``str`` for text-only results, or a ``ToolResult``
        when the tool needs to attach images (e.g. reading an image file).
        """
        ...

    def is_read_only(self, **kwargs) -> bool:
        """Whether this specific call is read-only (no side effects).

        Override for input-dependent logic (e.g., BashTool checks the command).
        If not overridden, checks for a class-level `_always_read_only` attribute
        for tools that are unconditionally read-only.
        Default: False (conservative — assume writes).
        """
        return getattr(self, "_always_read_only", False)

    def check_permissions(self, kwargs: dict, context: ToolContext) -> "PermissionResult":
        """Check permissions for this specific call.

        Override for custom permission logic (e.g., path constraints,
        dangerous command detection). Default: allow all.

        Called BEFORE call() in the tool execution pipeline.
        """
        from ..core.permissions import allow

        return allow()

    def validate_input(self, args: dict) -> dict:
        """Validate and coerce input using the Pydantic model.

        Override for custom validation beyond what Pydantic provides.
        Returns the validated dict. Raises ValidationError on failure.
        """
        validated = self.input_schema.model_validate(args)
        return validated.model_dump()

    def to_json_schema(self) -> dict:
        """Generate the OpenAPI-style function schema for the LLM."""
        params_schema = self.input_schema.model_json_schema()
        params_schema.pop("title", None)
        if "required" not in params_schema:
            params_schema["required"] = []
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params_schema,
            },
        }
