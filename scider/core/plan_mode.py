"""Plan mode state management.

Plan mode restricts the agent to read-only tools while it explores the codebase
and formulates a plan. Once the plan is submitted via ExitPlanMode, the agent
returns to normal mode and can execute the plan.

Inspired by Claude Code's plan mode (docs/10-plan-mode.md).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class QueryMode(str, enum.Enum):
    NORMAL = "normal"
    PLAN = "plan"


@dataclass
class PlanModeState:
    """Tracks plan mode state within a single query() session."""

    mode: QueryMode = QueryMode.NORMAL
    # Mode before entering plan (for restoration on exit)
    pre_plan_mode: QueryMode | None = None
    # The plan text submitted via ExitPlanMode
    plan_content: str | None = None
    # Whether the plan has been approved (auto-approve for now)
    plan_approved: bool = False

    def enter_plan(self) -> None:
        """Transition to plan mode."""
        self.pre_plan_mode = self.mode
        self.mode = QueryMode.PLAN
        self.plan_content = None
        self.plan_approved = False

    def exit_plan(self, plan: str) -> None:
        """Exit plan mode, store the plan, and restore previous mode."""
        self.plan_content = plan
        self.plan_approved = True
        self.mode = self.pre_plan_mode or QueryMode.NORMAL
        self.pre_plan_mode = None

    @property
    def is_plan_mode(self) -> bool:
        return self.mode == QueryMode.PLAN
