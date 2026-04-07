"""EnterPlanModeTool — enter plan mode for structured planning."""

from __future__ import annotations

from pydantic import BaseModel

from ...base import BaseTool, ToolContext

PLAN_MODE_INSTRUCTIONS = """\
Plan mode is now active. You have read-only access to the codebase.

## Workflow

### Phase 1: Explore
Read relevant files to understand the codebase structure and the task requirements.
Use Read, Glob, Grep, WebSearch, and other read-only tools.

### Phase 2: Design
Design an implementation approach. Consider:
- What needs to change and why
- Trade-offs between different approaches
- Which existing code/patterns to reuse

### Phase 3: Clarify
If anything is unclear about the requirements, ask the user for clarification.

### Phase 4: Write Plan
Formulate your plan with these sections:
- **Context**: Why this change is needed
- **Approach**: Step-by-step implementation plan
- **Files**: Which files to create/modify (with specific changes)
- **Verification**: How to test the changes

### Phase 5: Submit
Call ExitPlanMode with your complete plan text to submit it for approval.

## Rules
- You can ONLY use read-only tools (Read, Glob, Grep, WebSearch, etc.)
- You CANNOT modify files, run commands, or make changes
- Focus on understanding before proposing solutions
- Be specific about file paths and code changes in your plan
"""


class EnterPlanModeInput(BaseModel):
    pass


class EnterPlanModeTool(BaseTool):
    name = "EnterPlanMode"
    description = (
        "Enter plan mode for complex tasks that need careful planning. "
        "In plan mode, only read-only tools are available. "
        "Use this when: implementing new features, making multi-file changes, "
        "architecture decisions, or when the approach is unclear."
    )
    input_schema = EnterPlanModeInput
    _always_read_only = True
    prompt = (
        "# EnterPlanMode tool usage\n"
        "- Use for complex tasks: new features, multi-file changes, architecture decisions.\n"
        "- Do NOT use for simple tasks: single-line fixes, obvious bugs, small tweaks.\n"
        "- In plan mode, only read-only tools are available. You explore, then write a plan.\n"
        "- Call ExitPlanMode with your plan when done to submit for approval.\n"
    )

    def call(self, context: ToolContext) -> str:
        from scider.core.plan_mode import PlanModeState

        plan_state: PlanModeState | None = context.extra.get("plan_mode_state")
        if plan_state is None:
            return "Error: plan mode state not available in this context."

        if plan_state.is_plan_mode:
            return "Already in plan mode. Continue exploring and planning."

        plan_state.enter_plan()
        return PLAN_MODE_INSTRUCTIONS
