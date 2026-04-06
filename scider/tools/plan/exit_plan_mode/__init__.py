"""ExitPlanModeTool — exit plan mode and submit plan for user approval."""

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
        "Exit plan mode and submit your plan for user approval. "
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
        "- The user will review and approve/reject your plan before you can proceed.\n"
        "- Do NOT use this for research tasks — only for implementation planning.\n"
    )

    def call(self, context: ToolContext, *, plan: str) -> str:
        from scider.core.approval import ApprovalResult, _get_handler
        from scider.core.plan_mode import PlanModeState

        plan_state: PlanModeState | None = context.extra.get("plan_mode_state")
        if plan_state is None:
            return "Error: plan mode state not available in this context."

        if not plan_state.is_plan_mode:
            return (
                "Error: not in plan mode. Call enter_plan_mode first "
                "if you want to create a plan."
            )

        # Present plan to user for approval
        handler = _get_handler()
        response = handler.request_approval(
            node_name="plan_review",
            summary=f"## Proposed Plan\n\n{plan}",
            title="Review the agent's plan before implementation",
        )

        if response.result == ApprovalResult.APPROVED:
            plan_state.exit_plan(plan)
            return (
                "Plan approved by user. You can now start implementing.\n\n"
                f"## Approved Plan\n\n{plan}"
            )
        elif response.result == ApprovalResult.FEEDBACK:
            user_feedback = response.feedback or ""
            # Stay in plan mode — agent should revise the plan
            return (
                f"Plan NOT approved. The user provided feedback. "
                f"Please revise your plan and call ExitPlanMode again.\n\n"
                f"**User feedback:**\n{user_feedback}"
            )
        else:
            # Rejected — exit plan mode but plan is not approved
            plan_state.exit_plan(plan)
            plan_state.plan_approved = False
            return (
                "Plan rejected by user. Exiting plan mode. "
                "Consider a different approach or ask the user for guidance."
            )
