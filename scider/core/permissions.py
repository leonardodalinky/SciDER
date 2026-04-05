"""Permission system for tool execution.

Modeled after Claude Code's permission architecture (§11 permission-security).
Adapted for SciDER's backend agent context — no interactive UI, but provides
rule-based allow/deny and per-call permission checking.

Each tool can implement:
- is_read_only(input) — whether this specific call only reads (no side effects)
- check_permissions(input, context) — custom permission logic per call

Permission decisions flow:
1. Check deny rules → DENY
2. Call tool.check_permissions() → allow/deny/log
3. Check allow rules → ALLOW
4. Default → ALLOW (backend agent, no user to ask)
"""

from __future__ import annotations

import enum
import json
import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Permission result types
# ---------------------------------------------------------------------------


class PermissionBehavior(str, enum.Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass
class PermissionResult:
    behavior: PermissionBehavior
    reason: str | None = None
    # The tool name and input that was checked
    tool_name: str | None = None

    @property
    def allowed(self) -> bool:
        return self.behavior == PermissionBehavior.ALLOW


def allow(reason: str | None = None) -> PermissionResult:
    return PermissionResult(behavior=PermissionBehavior.ALLOW, reason=reason)


def deny(reason: str) -> PermissionResult:
    return PermissionResult(behavior=PermissionBehavior.DENY, reason=reason)


# ---------------------------------------------------------------------------
# Permission rules (loaded from JSON config)
# ---------------------------------------------------------------------------

# Default dangerous paths that should always be protected
DANGEROUS_PATHS = {
    ".git",
    ".gitconfig",
    ".gitmodules",
    ".bashrc",
    ".bash_profile",
    ".zshrc",
    ".zprofile",
    ".profile",
    ".env",
    ".claude",
}


@dataclass
class PermissionRules:
    """Permission rules loaded from settings."""

    allow: list[str]  # Tool names or Tool(pattern) rules
    deny: list[str]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PermissionRules":
        """Load rules from JSON file, or return defaults."""
        if path is None:
            path = os.getenv("SCIDER_PERMISSIONS_FILE", ".claude/permissions.json")

        path = Path(path)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls(
                    allow=data.get("allow", []),
                    deny=data.get("deny", []),
                )
            except Exception as e:
                logger.warning("Failed to load permission rules from {}: {}", path, e)

        return cls(allow=[], deny=[])

    def check(self, tool_name: str, input_str: str | None = None) -> PermissionResult | None:
        """Check rules against tool name and optional input string.

        Returns PermissionResult if a rule matches, None if no rule applies.
        Deny rules are checked first (deny > allow).
        """
        # Check deny rules first
        for rule in self.deny:
            if _rule_matches(rule, tool_name, input_str):
                return deny(f"Denied by rule: {rule}")

        # Check allow rules
        for rule in self.allow:
            if _rule_matches(rule, tool_name, input_str):
                return allow(f"Allowed by rule: {rule}")

        return None


def _rule_matches(rule: str, tool_name: str, input_str: str | None) -> bool:
    """Match a permission rule against a tool call.

    Rule formats:
    - "ToolName" — matches all calls to that tool
    - "ToolName(exact)" — exact match on input
    - "ToolName(prefix:*)" — prefix match
    """
    if "(" not in rule:
        # Whole-tool rule
        return rule == tool_name

    # Parse Tool(pattern)
    paren_idx = rule.index("(")
    rule_tool = rule[:paren_idx]
    if rule_tool != tool_name:
        return False

    pattern = rule[paren_idx + 1 : -1]  # Strip parens

    if input_str is None:
        return False

    if pattern.endswith(":*"):
        # Prefix match
        prefix = pattern[:-2]
        return input_str.startswith(prefix)
    elif "*" in pattern:
        # Simple wildcard (only at start/end)
        if pattern.startswith("*"):
            return input_str.endswith(pattern[1:])
        elif pattern.endswith("*"):
            return input_str.startswith(pattern[:-1])
        return False
    else:
        # Exact match
        return input_str == pattern


# ---------------------------------------------------------------------------
# Dangerous path checking
# ---------------------------------------------------------------------------


def is_dangerous_path(file_path: str) -> bool:
    """Check if a file path targets a dangerous location."""
    path = Path(file_path).resolve()
    parts = path.parts

    for part in parts:
        if part.lower() in DANGEROUS_PATHS:
            return True

    # Check the filename itself
    if path.name.lower() in DANGEROUS_PATHS:
        return True

    return False


def check_path_in_workspace(file_path: str, workspace: str | None) -> PermissionResult:
    """Check if a file path is within the allowed workspace."""
    if workspace is None:
        return allow()

    try:
        resolved = Path(file_path).resolve()
        workspace_resolved = Path(workspace).resolve()

        if not str(resolved).startswith(str(workspace_resolved)):
            return deny(
                f"Path '{file_path}' is outside workspace '{workspace}'. "
                f"Only operations within the workspace are allowed."
            )
    except Exception:
        pass

    return allow()


# ---------------------------------------------------------------------------
# Global rules instance (loaded once)
# ---------------------------------------------------------------------------

_rules: PermissionRules | None = None


def get_rules() -> PermissionRules:
    global _rules
    if _rules is None:
        _rules = PermissionRules.load()
    return _rules


def check_tool_permission(
    tool_name: str,
    input_str: str | None = None,
) -> PermissionResult:
    """Check global permission rules for a tool call.

    This is called from query()'s _execute_tool_calls before each tool execution.
    """
    rules = get_rules()
    result = rules.check(tool_name, input_str)
    if result is not None:
        if not result.allowed:
            logger.warning("Permission denied for {}: {}", tool_name, result.reason)
        return result

    # No rule matched — default allow (backend agent, no user to ask)
    return allow()
