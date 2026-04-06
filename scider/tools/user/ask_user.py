"""AskUserQuestion tool — ask the user a question with structured options.

Modeled after Claude Code's AskUserQuestionTool. The agent calls this tool
when it needs clarification or a decision from the user. The tool blocks
until the user responds, then returns their answer as a tool result.

This is different from the workflow-level approval system:
- AskUserQuestion: agent-initiated, inside query() loop, for clarification
- Approval nodes: workflow-initiated, between graph nodes, for gate-keeping

Uses the existing ApprovalHandler infrastructure for cross-environment support
(CLI, Jupyter, Streamlit).
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from ..base import BaseTool, ToolContext


class QuestionOption(BaseModel):
    """A single option for a question."""

    label: str = Field(description="Short display text for this option (1-5 words)")
    description: str = Field(
        default="",
        description="Explanation of what this option means or what happens if chosen",
    )


class Question(BaseModel):
    """A question to ask the user."""

    question: str = Field(
        description=(
            "The question to ask. Should be clear, specific, and end with a question mark."
        ),
    )
    options: list[QuestionOption] = Field(
        default_factory=list,
        description=(
            "Available choices (2-4 options). If empty, the user provides a free-text answer."
        ),
    )


class AskUserQuestionInput(BaseModel):
    questions: list[Question] = Field(
        description="Questions to ask the user (1-4 questions)",
        min_length=1,
        max_length=4,
    )


class AskUserQuestionTool(BaseTool):
    name = "AskUserQuestion"
    description = (
        "Ask the user a question when you need clarification or a decision. "
        "Each question can have 2-4 predefined options, or be open-ended (no options). "
        "Use this when you are unsure about the user's intent, need to choose between "
        "approaches, or want confirmation before a significant action. "
        "Do NOT use this for routine status updates — only for genuine questions."
    )
    input_schema = AskUserQuestionInput
    _always_read_only = True
    prompt = (
        "# AskUserQuestion tool usage\n"
        "- NEVER ask a question in plain text. If you need user input, you MUST call this tool. "
        "A plain-text question ends your turn and the user CANNOT answer.\n"
        "- Use to gather preferences, clarify ambiguous instructions, or get decisions.\n"
        "- Provide 2-4 clear, distinct options when possible. Each option needs a label and description.\n"
        "- For open-ended questions, omit options — the user provides free-text input.\n"
        "- Do NOT use for routine updates or obvious next steps. Only for genuine ambiguity.\n"
        "- You can ask 1-4 questions in a single call.\n"
    )

    def call(
        self,
        context: ToolContext,
        *,
        questions: list[dict],
    ) -> str:
        from scider.core.approval import _get_handler

        handler = _get_handler()
        answers: dict[str, str] = {}

        for q_data in questions:
            q = Question.model_validate(q_data)

            if q.options and len(q.options) >= 2:
                # Structured question with options → use selection approval
                items = [{"title": opt.label, "description": opt.description} for opt in q.options]
                response = handler.request_approval_with_selection(
                    node_name="AskUserQuestion",
                    summary=q.question,
                    items=items,
                    title="Agent Question",
                )

                if response.selected_index is not None and response.selected_index < len(q.options):
                    answers[q.question] = q.options[response.selected_index].label
                elif response.feedback:
                    answers[q.question] = response.feedback
                else:
                    answers[q.question] = "(no selection)"

            else:
                # Open-ended question → use basic approval with feedback
                response = handler.request_approval(
                    node_name="AskUserQuestion",
                    summary=q.question,
                    title="Agent Question",
                )

                if response.feedback:
                    answers[q.question] = response.feedback
                elif response.result.value == "approved":
                    answers[q.question] = "(approved without specific answer)"
                else:
                    answers[q.question] = "(rejected)"

        return json.dumps(
            {"answers": answers},
            ensure_ascii=False,
        )
