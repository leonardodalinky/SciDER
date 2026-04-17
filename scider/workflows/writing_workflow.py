"""
Writing Workflow

Partial workflow that bootstraps a PaperOrchestra-ready paper workspace
under a given SciDER workspace and then runs the writing agent on it.

Inputs accepted as plain strings **or file paths** (if a ``Path`` or a
string pointing to an existing file is given, its contents are read
automatically):
- ``idea_summary``          → inputs/idea.md
- ``experimental_log``      → inputs/experimental_log.md
- ``data_summary``          (optional, shown to agent as context)
- ``experiment_summary``    (optional, shown to agent as context)

Optional:
- ``user_query`` — top-level task message shown to the agent; defaults to
  ``DEFAULT_USER_QUERY`` (a generic "run the pipeline" nudge). The real
  content lives in ``inputs/`` so this is just display framing.

Optional assets (fall back to the bundled ``simple`` template directory):
- ``template_dir_path`` — directory with ``template.tex``, ``guidelines.md``,
  and any auxiliary ``.sty`` / ``.bst`` / ``.cls`` / ``.bib`` support files.
  All top-level files are copied verbatim into ``inputs/``.
- ``template_tex_path`` — explicit override for just ``template.tex``
  (takes precedence over the file from ``template_dir_path``).
- ``conference_guidelines_path`` — explicit override for just the
  guidelines file (takes precedence over the file from ``template_dir_path``).
- ``figures_src_path`` (contents copied into inputs/figures/)

Useful for debugging the paper-writing phase independently. For the full
"ideation → data → experiment → writing" pipeline see ``full_workflow.py``
and ``full_workflow_with_ideation.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, PrivateAttr

from scider.agents import writing_agent
from scider.agents.writing_agent.state import WritingAgentState
from scider.core.code_env import LocalEnv, WorkspaceInitConfig
from scider.core.constant import override_user_approval
from scider.workflows.paper_bootstrap import bootstrap_paper_workspace
from scider.workflows.utils import get_separator

# Default top-level task message shown to the writing agent when the caller
# doesn't supply one. The real content — idea and experimental log — is
# already staged in ``inputs/`` before the agent starts, so the "task" here
# is just a nudge pointing at the pipeline.
DEFAULT_USER_QUERY = (
    "Run the full PaperOrchestra pipeline on the prepared paper workspace. "
    "The idea, experimental log, template, and guidelines are already in "
    "`inputs/`. Produce a compiled `final/paper.pdf` by walking through the "
    "six paper-orchestra steps."
)


class WritingWorkflow(BaseModel):
    """
    Writing Workflow — bootstrap a paper workspace and run WritingAgent.
    """

    # ==================== INPUT ====================
    scider_workspace_path: Path
    # Default resolved in ``_resolve_paths`` → scider_workspace_path / "paper"
    paper_workspace_path: Path | None = None

    # Required content (already shaped as PaperOrchestra inputs).
    # These strings become ``inputs/idea.md`` and ``inputs/experimental_log.md``
    # verbatim, so the caller is responsible for giving them the right shape
    # (see ``paper_bootstrap.build_*`` helpers for producing them from a
    # SciDER run, or write them by hand for a one-off paper job).
    idea_summary: str
    experimental_log: str
    # Top-level task message shown to the agent. Optional — if the caller
    # doesn't care, ``DEFAULT_USER_QUERY`` is a sensible nudge pointing at
    # the pipeline. The real content (idea + log) lives in ``inputs/``.
    user_query: str = DEFAULT_USER_QUERY

    # Assets (fall back to bundled "simple" template directory when all None)
    template_dir_path: Path | None = None
    template_tex_path: Path | None = None
    conference_guidelines_path: Path | None = None
    figures_src_path: Path | None = None

    # Pass-through context for the agent (shown as preview in the init message)
    data_summary: str = ""
    experiment_summary: str = ""

    recursion_limit: int = 150
    # None → LocalEnv's default (uv-managed, auto `uv init`). Override to
    # point agents at a prebuilt venv or to skip `uv init` in the workspace.
    workspace_init_config: WorkspaceInitConfig | None = None

    # ==================== INTERNAL STATE ====================
    current_phase: Literal["init", "bootstrap", "writing", "complete", "failed"] = "init"

    # ==================== OUTPUT ====================
    final_status: Literal["success", "failed"] | None = None
    final_summary: str = ""
    final_tex_path: Path | None = None
    final_pdf_path: Path | None = None
    writing_agent_history: list = []
    writing_agent_intermediate_state: list[dict] = []
    error_message: str | None = None

    # Internal: lazily compiled graph
    _writing_agent_graph: Any = PrivateAttr(default=None)

    def _ensure_graph(self) -> None:
        if self._writing_agent_graph is None:
            self._writing_agent_graph = writing_agent.build().compile()

    def _resolve_paths(self) -> None:
        """Materialize defaults for optional path params."""
        if self.paper_workspace_path is None:
            self.paper_workspace_path = self.scider_workspace_path / "paper"
        self.paper_workspace_path = Path(self.paper_workspace_path).resolve()
        self.paper_workspace_path.mkdir(parents=True, exist_ok=True)

        # Default figures source: <scider_workspace>/imgs if present
        if self.figures_src_path is None:
            default_imgs = self.scider_workspace_path / "imgs"
            if default_imgs.is_dir():
                self.figures_src_path = default_imgs

    def run(self) -> "WritingWorkflow":
        """Bootstrap the paper workspace and run the writing agent."""
        logger.info(get_separator())
        logger.info("Starting Writing Workflow")
        logger.info(get_separator())

        try:
            self._resolve_paths()
            self._ensure_graph()
        except Exception as e:
            logger.exception("WritingWorkflow setup failed")
            self.error_message = f"Setup failed: {e}"
            self.current_phase = "failed"
            self._finalize(success=False)
            return self

        if not self._bootstrap_phase():
            self._finalize(success=False)
            return self

        success = self._run_writing_phase()
        self._finalize(success=success)
        return self

    # ------------------------------------------------------------------ #
    # Phases
    # ------------------------------------------------------------------ #

    def _bootstrap_phase(self) -> bool:
        """Write PaperOrchestra inputs/ layout into paper_workspace_path."""
        logger.info("Phase 1: Bootstrapping paper workspace at {}", self.paper_workspace_path)
        self.current_phase = "bootstrap"

        try:
            template_tex = None
            if self.template_tex_path is not None:
                template_tex = Path(self.template_tex_path).read_text(encoding="utf-8")

            conference_guidelines = None
            if self.conference_guidelines_path is not None:
                conference_guidelines = Path(self.conference_guidelines_path).read_text(
                    encoding="utf-8"
                )

            bootstrap_paper_workspace(
                self.paper_workspace_path,
                idea_summary=self.idea_summary,
                experimental_log=self.experimental_log,
                template_dir=self.template_dir_path,
                template_tex=template_tex,
                conference_guidelines=conference_guidelines,
                figures_src=self.figures_src_path,
            )
            return True
        except Exception as e:
            logger.exception("Paper workspace bootstrap failed")
            self.error_message = f"Bootstrap failed: {e}"
            self.current_phase = "failed"
            return False

    def _run_writing_phase(self) -> bool:
        """Invoke the writing agent graph on the bootstrapped workspace."""
        logger.info("Phase 2: Running WritingAgent on {}", self.paper_workspace_path)
        self.current_phase = "writing"

        writing_state = WritingAgentState(
            workspace=LocalEnv(self.paper_workspace_path, init_config=self.workspace_init_config),
            scider_workspace=self.scider_workspace_path,
            user_query=self.user_query,
            idea_summary=self.idea_summary,
            experiment_summary=self.experiment_summary,
            data_summary=self.data_summary,
        )

        from scider.workflows.history_export import capture_messages

        with capture_messages() as captured:
            try:
                result = self._writing_agent_graph.invoke(
                    writing_state,
                    {"recursion_limit": self.recursion_limit},
                )
                result_state = WritingAgentState(**result)

                self.writing_agent_history = list(captured)
                self.writing_agent_intermediate_state = result_state.intermediate_state
                self.final_summary = result_state.output_summary or ""

                if result_state.final_tex_path:
                    self.final_tex_path = Path(result_state.final_tex_path)
                if result_state.final_pdf_path:
                    self.final_pdf_path = Path(result_state.final_pdf_path)

                # Fallback probing in case the agent failed to set them
                self._probe_artifacts_on_disk()

                # If agent didn't produce final/paper.pdf, try to salvage
                # from drafts/ or refinement/ as a last resort.
                if self.final_pdf_path is None or not self.final_pdf_path.exists():
                    logger.warning(
                        "WritingAgent did not produce final/paper.pdf — "
                        "attempting automatic salvage from partial artifacts"
                    )
                    if self._salvage_final_pdf():
                        logger.info("Salvage succeeded — final/paper.pdf produced")
                    else:
                        self.error_message = (
                            "WritingAgent finished but no final/paper.pdf was produced. "
                            "Check the paper_workspace for partial artifacts."
                        )
                        self.current_phase = "failed"
                        return False

                logger.info("WritingAgent completed successfully")
                return True
            except Exception as e:
                self.writing_agent_history = list(captured)
                logger.exception("WritingAgent failed")
                self.error_message = f"WritingAgent failed: {e}"
                self.current_phase = "failed"
                return False

    def _probe_artifacts_on_disk(self) -> None:
        """Fallback — scan paper_workspace/final/ in case the agent didn't set
        final_tex_path / final_pdf_path explicitly."""
        if self.paper_workspace_path is None:
            return
        final_dir = self.paper_workspace_path / "final"
        if self.final_tex_path is None:
            tex = final_dir / "paper.tex"
            if tex.is_file():
                self.final_tex_path = tex.resolve()
        if self.final_pdf_path is None:
            pdf = final_dir / "paper.pdf"
            if pdf.is_file():
                self.final_pdf_path = pdf.resolve()

    def _salvage_final_pdf(self) -> bool:
        """Last-resort: promote the best available .tex to final/ and compile.

        Searches refinement/iter*/paper.tex (latest first), then drafts/paper.tex.
        Returns True if final/paper.pdf is successfully produced.
        """
        import shutil
        import subprocess

        if self.paper_workspace_path is None:
            return False

        ws = self.paper_workspace_path
        final_dir = ws / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Find best .tex source: latest refinement iteration, else drafts
        source_tex: Path | None = None
        refinement_dir = ws / "refinement"
        if refinement_dir.is_dir():
            iters = sorted(refinement_dir.glob("iter*/paper.tex"), reverse=True)
            if iters:
                source_tex = iters[0]

        if source_tex is None:
            drafts_tex = ws / "drafts" / "paper.tex"
            if drafts_tex.is_file():
                source_tex = drafts_tex

        if source_tex is None:
            logger.warning("Salvage: no paper.tex found in refinement/ or drafts/")
            return False

        logger.info("Salvage: promoting {} to final/paper.tex", source_tex)
        shutil.copy2(source_tex, final_dir / "paper.tex")

        # Copy supporting files into final/
        for src_name, dst_name in [
            (ws / "drafts" / "intro_relwork.tex", "intro_relwork.tex"),
            (ws / "refs.bib", "refs.bib"),
        ]:
            if src_name.is_file() and not (final_dir / dst_name).exists():
                shutil.copy2(src_name, final_dir / dst_name)

        # Copy figures/
        figures_src = ws / "figures"
        figures_dst = final_dir / "figures"
        if figures_src.is_dir() and not figures_dst.exists():
            shutil.copytree(figures_src, figures_dst)

        # Copy aux LaTeX files from inputs/
        inputs_dir = ws / "inputs"
        if inputs_dir.is_dir():
            for ext in ("*.sty", "*.bst", "*.cls", "*.cfg", "*.bib"):
                for f in inputs_dir.glob(ext):
                    if not (final_dir / f.name).exists():
                        shutil.copy2(f, final_dir / f.name)

        # Compile
        try:
            result = subprocess.run(
                ["latexmk", "-pdf", "-interaction=nonstopmode", "paper.tex"],
                cwd=str(final_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
            pdf = final_dir / "paper.pdf"
            if pdf.is_file():
                self.final_tex_path = (final_dir / "paper.tex").resolve()
                self.final_pdf_path = pdf.resolve()
                return True
            else:
                logger.warning("Salvage: latexmk finished but paper.pdf not found")
                logger.debug("Salvage latexmk stdout: {}", result.stdout[-500:])
                return False
        except Exception as e:
            logger.warning("Salvage: latexmk failed: {}", e)
            return False

    def _finalize(self, success: bool) -> None:
        """Finalize status and persist history."""
        logger.info("Finalizing writing workflow")

        if success:
            self.final_status = "success"
            self.current_phase = "complete"
        else:
            self.final_status = "failed"

        if self.writing_agent_history:
            from scider.workflows.history_export import save_conversation_history

            try:
                history_path = (
                    Path(self.paper_workspace_path) / "writing_agent_history.json"
                    if self.paper_workspace_path is not None
                    else Path(self.scider_workspace_path) / "writing_agent_history.json"
                )
                save_conversation_history(
                    self.writing_agent_history,
                    history_path,
                    agent_name="writing",
                )
            except Exception as e:
                logger.warning("Failed to save writing agent history: {}", e)

        logger.info(get_separator())
        logger.info(f"Writing Workflow completed: {self.final_status}")
        logger.info(get_separator())

    def save_summary(self, path: str | Path | None = None) -> Path:
        """Save the writing report to a file."""
        if path is None:
            path = (self.paper_workspace_path or self.scider_workspace_path) / "writing_report.md"
        path = Path(path)
        path.write_text(self.final_summary, encoding="utf-8")
        logger.info(f"Writing report saved to {path}")
        return path


def _resolve_text(value: str | Path) -> str:
    """If *value* points to an existing file, read and return its contents;
    otherwise return *value* as a plain string."""
    if isinstance(value, Path):
        if value.is_file():
            return value.read_text(encoding="utf-8")
        return str(value)
    # For str: only try as a path if it's short enough to be a plausible
    # file path (OS limit is typically 4096) and contains no newlines.
    if len(value) <= 4096 and "\n" not in value:
        try:
            p = Path(value)
            if p.is_file():
                return p.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            logger.debug("_resolve_text: path resolution failed for '{}': {}", value, exc)
    return value


def run_writing_workflow(
    scider_workspace_path: str | Path,
    idea_summary: str | Path,
    experimental_log: str | Path,
    *,
    user_query: str | None = None,
    paper_workspace_path: str | Path | None = None,
    template_dir_path: str | Path | None = None,
    template_tex_path: str | Path | None = None,
    conference_guidelines_path: str | Path | None = None,
    figures_src_path: str | Path | None = None,
    data_summary: str | Path = "",
    experiment_summary: str | Path = "",
    recursion_limit: int = 150,
    user_approval_enabled: bool = False,
    workspace_init_config: WorkspaceInitConfig | None = None,
) -> WritingWorkflow:
    """
    Convenience function to run the paper writing workflow.

    Args:
        scider_workspace_path: Path to the SciDER workspace holding original
            data/code/experiment artifacts (used as read-only source).
        idea_summary: Idea markdown (or path to a ``.md`` file) to write into
            ``inputs/idea.md``. If a ``Path`` or a string pointing to an
            existing file is given, its contents are read automatically.
        experimental_log: Experimental log markdown (or path to a ``.md``
            file) to write into ``inputs/experimental_log.md``. Same
            path-auto-read behaviour as ``idea_summary``.
        user_query: Optional top-level task message shown to the writing
            agent as the "# Task" line in its init message. The idea and
            experimental log are already staged under ``inputs/`` by the
            time the agent starts, so this is just a nudge. Defaults to
            ``DEFAULT_USER_QUERY`` which points at the pipeline.
        paper_workspace_path: Where to bootstrap the PaperOrchestra layout.
            Defaults to ``<scider_workspace_path>/paper``.
        template_dir_path: Optional directory containing ``template.tex``,
            ``guidelines.md``, and any auxiliary LaTeX support files
            (``.sty`` / ``.bst`` / ``.cls`` / ``.bib``). All top-level files
            are copied verbatim into ``inputs/``. Defaults to the bundled
            ``simple`` template directory.
        template_tex_path: Optional explicit override for just
            ``template.tex``. Takes precedence over whatever
            ``template_dir_path`` provided.
        conference_guidelines_path: Optional explicit override for just the
            guidelines markdown file. Takes precedence over
            ``template_dir_path``.
        figures_src_path: Optional directory whose image files are copied
            into ``inputs/figures/``. Defaults to ``<scider_ws>/imgs`` if it
            exists.
        data_summary: Optional SciDER data-analysis summary (or file path).
            Shown to the agent in its init message as reference context.
        experiment_summary: Optional SciDER experiment summary (or file path).
        recursion_limit: Recursion limit for WritingAgent (default=150).
        user_approval_enabled: Forwarded to ``override_user_approval``.
        workspace_init_config: Override LocalEnv init behaviour (uv init, PATH
            injection, env manager). Leave ``None`` for historical defaults.

    Returns:
        WritingWorkflow: Completed workflow with results.
    """
    # Resolve str|Path inputs: if an existing file path is given, read it.
    idea_summary_text = _resolve_text(idea_summary)
    experimental_log_text = _resolve_text(experimental_log)
    data_summary_text = _resolve_text(data_summary) if data_summary else ""
    experiment_summary_text = _resolve_text(experiment_summary) if experiment_summary else ""

    workflow = WritingWorkflow(
        scider_workspace_path=Path(scider_workspace_path),
        paper_workspace_path=Path(paper_workspace_path) if paper_workspace_path else None,
        idea_summary=idea_summary_text,
        experimental_log=experimental_log_text,
        # None → Pydantic uses the field default (``DEFAULT_USER_QUERY``).
        **({"user_query": user_query} if user_query is not None else {}),
        template_dir_path=Path(template_dir_path) if template_dir_path else None,
        template_tex_path=Path(template_tex_path) if template_tex_path else None,
        conference_guidelines_path=(
            Path(conference_guidelines_path) if conference_guidelines_path else None
        ),
        figures_src_path=Path(figures_src_path) if figures_src_path else None,
        data_summary=data_summary_text,
        experiment_summary=experiment_summary_text,
        recursion_limit=recursion_limit,
        workspace_init_config=workspace_init_config,
    )
    with override_user_approval(user_approval_enabled):
        return workflow.run()
