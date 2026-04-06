"""ExitPlanModeTool — exit plan mode and submit plan for approval."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext


class ExitPlanModeInput(BaseModel):
    plan: str = Field(
        description="The complete plan text to submit for approval.",
        min_length=10,
    )


class ExitPlanModeTool(BaseTool):
    name = "ExitPlanMode"
    description = (
        "Exit plan mode and submit your plan for approval. "
        "The plan should include: context, approach, files to modify, "
        "and verification steps. Only call this after thorough exploration."
    )
    input_schema = ExitPlanModeInput
    _always_read_only = True  # doesn't modify files, just transitions state
    prompt = (
        "# ExitPlanMode tool usage\n"
        "- Call this ONLY after thorough exploration and plan writing.\n"
        "- The plan must include: context (why), approach (step-by-step), "
        "files to modify, and verification steps.\n"
        "- Do NOT use this for research tasks — only for implementation planning.\n"
    )

    def call(self, context: ToolContext, *, plan: str) -> str:
        from scider.core.plan_mode import PlanModeState

        plan_state: PlanModeState | None = context.extra.get("plan_mode_state")
        if plan_state is None:
            return "Error: plan mode state not available in this context."

        if not plan_state.is_plan_mode:
            return (
                "Error: not in plan mode. Call enter_plan_mode first "
                "if you want to create a plan."
            )

        plan_state.exit_plan(plan)

        return "Plan approved. You can now start implementing.\n\n" f"## Approved Plan\n\n{plan}"
