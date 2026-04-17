"""Hypothetical Data Workflow.

Generates synthetic data from user-described features, gets user approval on
the data specification, then runs the standard DataWorkflow on the generated data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from scider.core import constant
from scider.core.approval import ApprovalResult, _get_handler
from scider.core.code_env import WorkspaceInitConfig
from scider.core.hypo_data_gen import (
    DataGenSpec,
    format_spec_for_review,
    generate_csv_from_spec,
    generate_data_spec,
)
from scider.workflows.data_workflow import DataWorkflow


class HypoDataWorkflow(BaseModel):
    """Workflow that generates hypothetical data then analyzes it.

    Usage::

        w = HypoDataWorkflow(
            feature_desc="A dataset about iris flowers with sepal/petal measurements",
            workspace_path=Path("workspace"),
        )
        w.run()
        print(w.data_summary)
    """

    # ── Input ──
    feature_desc: str
    workspace_path: Path
    num_rows: int = 1000
    user_query: str = ""
    recursion_limit: int = 100
    # None → LocalEnv's default (uv-managed, auto `uv init`). Forwarded to the
    # wrapped DataWorkflow.
    workspace_init_config: WorkspaceInitConfig | None = None

    # Memory directories (optional)

    # ── Internal state ──
    current_phase: Literal[
        "init", "spec_gen", "data_gen", "data_analysis", "complete", "failed"
    ] = "init"
    data_spec: DataGenSpec | None = None
    generated_data_path: Path | None = None
    data_summary: str | None = None
    final_status: str = "init"
    error_message: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    # ── Max retry attempts for spec generation ──
    MAX_SPEC_RETRIES: int = 3

    def run(self) -> "HypoDataWorkflow":
        """Run the full workflow: generate spec → approve → generate CSV → analyze."""
        try:
            # Phase 1: Generate and approve data spec
            self.current_phase = "spec_gen"
            spec = self._generate_and_approve_spec()
            if spec is None:
                self.final_status = "failed"
                self.error_message = "Data spec generation was rejected by user."
                return self
            self.data_spec = spec

            # Phase 2: Generate CSV from spec
            self.current_phase = "data_gen"
            self.generated_data_path = generate_csv_from_spec(
                spec, self.workspace_path, self.num_rows
            )
            logger.info(f"Generated data at: {self.generated_data_path}")

            # Phase 3: Run data analysis
            self.current_phase = "data_analysis"
            data_desc = (
                f"This is hypothetical/synthetic data generated from the following description:\n"
                f"{self.feature_desc}\n\n"
                f"The data was generated with known distributions "
                f"({self.num_rows} rows, {len(spec.features)} features). "
                f"Focus on verifying the data structure and providing analysis insights."
            )

            query = self.user_query or f"Analyze this synthetic dataset: {self.feature_desc}"

            data_workflow = DataWorkflow(
                data_path=self.generated_data_path,
                workspace_path=self.workspace_path,
                recursion_limit=self.recursion_limit,
                data_desc=data_desc,
                workspace_init_config=self.workspace_init_config,
            )
            data_workflow.run()

            self.data_summary = data_workflow.data_summary
            self.final_status = data_workflow.final_status
            self.error_message = data_workflow.error_message
            self.current_phase = "complete" if self.final_status == "success" else "failed"

        except Exception as e:
            logger.exception("HypoDataWorkflow failed")
            self.final_status = "failed"
            self.error_message = str(e)
            self.current_phase = "failed"

        return self

    def _generate_and_approve_spec(self) -> DataGenSpec | None:
        """Generate data spec via LLM, then request user approval.

        Retries up to MAX_SPEC_RETRIES times if user provides feedback.
        Returns None if user ultimately rejects.
        """
        handler = _get_handler()
        feedback = None

        for attempt in range(self.MAX_SPEC_RETRIES):
            logger.info(f"Generating data spec (attempt {attempt + 1}/{self.MAX_SPEC_RETRIES})")
            try:
                spec = generate_data_spec(self.feature_desc, self.num_rows, feedback)
            except Exception as e:
                logger.error(f"Failed to generate data spec: {e}")
                if attempt < self.MAX_SPEC_RETRIES - 1:
                    feedback = f"Previous attempt failed with error: {e}. Please try again."
                    continue
                raise

            # Request approval
            if not constant.USER_APPROVAL_ENABLED:
                logger.info("Auto-approved data spec (USER_APPROVAL_ENABLED=false)")
                return spec

            summary = format_spec_for_review(spec)
            response = handler.request_approval(
                node_name="hypo_data_spec",
                summary=summary,
                title="Review the generated data specification before generating data.",
            )

            if response.result == ApprovalResult.APPROVED:
                logger.info("Data spec approved by user")
                return spec
            elif response.result == ApprovalResult.FEEDBACK:
                feedback = response.feedback
                logger.info(f"User provided feedback: {feedback}")
                continue
            else:
                logger.info("Data spec rejected by user")
                return None

        logger.warning("Max spec retries reached")
        return None


def run_hypo_data_workflow(
    feature_desc: str,
    workspace_path: str | Path,
    num_rows: int = 1000,
    user_query: str = "",
    recursion_limit: int = 100,
    user_approval_enabled: bool = True,
    workspace_init_config: WorkspaceInitConfig | None = None,
) -> HypoDataWorkflow:
    """Convenience function to run the hypothetical data workflow."""
    from scider.core.constant import override_user_approval

    with override_user_approval(user_approval_enabled):
        w = HypoDataWorkflow(
            feature_desc=feature_desc,
            workspace_path=Path(workspace_path),
            num_rows=num_rows,
            user_query=user_query,
            recursion_limit=recursion_limit,
            workspace_init_config=workspace_init_config,
        )
        w.run()
    return w
