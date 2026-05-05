"""AI-Idea-Bench internal plumbing — used only by ``generation``.

* ``run_ideation_task`` — invoke ``run_ideation_workflow`` on a single research
  seed. The workflow itself persists ``<workspace>/ideation_agent_history.json``
  (plus an ``ideation_summary.md`` if you call ``save_summary``); we just
  surface the workflow object so the caller can write the per-task
  ``output.json`` summary stub.
* ``is_workspace_complete`` / ``scan_completed_uids`` — skip-existing helpers
  mirroring ``data_generation.ds1000._common`` (look for a valid
  ``output.json`` + non-empty history).
* ``write_output_json`` — atomic write of the per-task summary stub used by
  the skip-existing scan.

If a third benchmark needs the same primitives later, lift the shared bits up
to ``data_generation/_common.py`` then — premature sharing makes both
consumers harder to evolve.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

# scider is on sys.path because the entry-point modules
# (data_generation.aiidea.generation) prepend the project root before
# importing this module.
from scider.workflows.ideation_workflow import IdeationWorkflow, run_ideation_workflow

OUTPUT_FILENAME = "output.json"
HISTORY_FILENAME = "ideation_agent_history.json"
PROMPT_FILENAME = "prompt.md"
SUMMARY_FILENAME = "ideation_summary.md"


# --------------------------------------------------------------------------- #
# Skip-existing helpers                                                       #
# --------------------------------------------------------------------------- #


def is_workspace_complete(workspace: Path) -> bool:
    """A workspace is "done" iff:
    - ``output.json`` exists, parses, and has ``ok == True``
    - ``ideation_agent_history.json`` exists and is non-empty (the
      trajectory is what we actually care about for SFT data — no point
      keeping a run that lost its history)
    """
    out = workspace / OUTPUT_FILENAME
    hist = workspace / HISTORY_FILENAME
    try:
        if not out.is_file() or out.stat().st_size == 0:
            return False
        data = json.loads(out.read_text(encoding="utf-8"))
        if not data.get("ok"):
            return False
    except (OSError, json.JSONDecodeError):
        return False
    if not hist.is_file() or hist.stat().st_size == 0:
        return False
    return True


def scan_completed_uids(output_root: Path) -> set[str]:
    """Used by ``--skip-existing``: returns uids whose workspace passes
    ``is_workspace_complete``."""
    if not output_root.is_dir():
        return set()
    completed: set[str] = set()
    for uid_dir in output_root.iterdir():
        if not uid_dir.is_dir():
            continue
        if is_workspace_complete(uid_dir):
            completed.add(uid_dir.name)
    return completed


def write_output_json(workspace: Path, payload: dict[str, Any]) -> None:
    """Write the per-task summary stub atomically (tmp + rename)."""
    out = workspace / OUTPUT_FILENAME
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)


# --------------------------------------------------------------------------- #
# Ideation-task runner                                                        #
# --------------------------------------------------------------------------- #


def run_ideation_task(
    *,
    user_query: str,
    workspace_dir: Path,
    research_domain: str | None = None,
    recursion_limit: int = 50,
    save_summary_md: bool = True,
) -> IdeationWorkflow:
    """Run one ideation task in ``workspace_dir`` and return the
    ``IdeationWorkflow`` object (already executed).

    The workflow's ``_finalize`` step persists the captured history to
    ``<workspace>/ideation_agent_history.json`` even on crash, so the
    trajectory survives partial failures the same way the ds1000 runner
    does via ``capture_messages``.
    """
    workspace_dir = Path(workspace_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running ideation workflow in {}", workspace_dir)
    workflow = run_ideation_workflow(
        user_query=user_query,
        workspace_path=workspace_dir,
        research_domain=research_domain,
        recursion_limit=recursion_limit,
    )

    if save_summary_md and workflow.ideation_summary:
        try:
            workflow.save_summary(workspace_dir / SUMMARY_FILENAME)
        except Exception as e:
            logger.warning("Failed to save ideation_summary.md: {}", e)

    return workflow
