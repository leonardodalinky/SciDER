from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from scider.core.code_env import WorkspaceInitConfig
    from scider.workflows.writing_workflow import WritingWorkflow


def run_paper_writing_phase(
    *,
    workspace_path: Path,
    idea_md: str,
    experimental_log: str,
    user_query: str,
    data_summary: str,
    experiment_summary: str,
    paper_workspace_path: Path | None,
    paper_template_dir_path: Path | None,
    paper_template_tex_path: Path | None,
    paper_conference_guidelines_path: Path | None,
    paper_agent_recursion_limit: int,
    workspace_init_config: "WorkspaceInitConfig | None" = None,
) -> "WritingWorkflow":
    """Shared paper writing phase used by both FullWorkflow and FullWorkflowWithIdeation.

    Returns the completed WritingWorkflow instance.
    Raises on failure (caller should catch and set error state).
    """
    from scider.workflows.writing_workflow import WritingWorkflow

    wf = WritingWorkflow(
        scider_workspace_path=workspace_path,
        paper_workspace_path=paper_workspace_path,
        idea_summary=idea_md,
        experimental_log=experimental_log,
        user_query=user_query,
        template_dir_path=paper_template_dir_path,
        template_tex_path=paper_template_tex_path,
        conference_guidelines_path=paper_conference_guidelines_path,
        figures_src_path=workspace_path / "imgs",
        data_summary=data_summary,
        experiment_summary=experiment_summary,
        recursion_limit=paper_agent_recursion_limit,
        workspace_init_config=workspace_init_config,
    )
    wf.run()
    return wf


def get_separator(margin: int = 4, char: str = "=") -> str:
    """
    Generate a separator that fits the terminal width.

    Args:
        margin: Number of characters to leave as margin (default: 4)
        char: Character to use for separator (default: '=')

    Returns:
        Separator string that fits terminal width
    """
    terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    # Leave margin to be safe and ensure minimum width
    separator_width = max(terminal_width - margin, 10)
    return char * separator_width
