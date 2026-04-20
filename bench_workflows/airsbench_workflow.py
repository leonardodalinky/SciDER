"""
AIRS-Bench Workflow — run SciDER's FullWorkflow (data + experiment, no ideation)
on each AIRS-Bench task and score the resulting ``submission.csv``.

Strategy
--------
For each task under ``airsbench/tasks/rad/<TaskName>/``:

1. Create a per-task workspace at ``<output_root>/<TaskName>/`` and ``uv init``
   it so the sub-env is isolated from everything else.
2. Copy the **agent-visible** files into the workspace:
   ``project_description.md``, ``metadata.yaml``, ``prepare.py`` (plus
   ``utils.py`` if present). Evaluator files (``evaluate.py`` /
   ``evaluate_prepare.py``) are withheld — the agent must not see them.
3. Run ``prepare.py`` via ``uv run --with <container_python_requirements>``
   (read from ``metadata.yaml``) so ``<ws>/data/{train,test,validation}/`` is
   populated before the agent starts.
4. Use ``project_description.md`` as the user query and hand the task to
   ``run_full_workflow`` (data + experiment, no ideation, no paper writing).
   ``WorkspaceInitConfig`` uses the uv-default (``init_uv=True``); since the
   workspace already has ``pyproject.toml`` from step 1, ``LocalEnv`` does
   not re-init. The agent's deliverable is ``<ws>/submission.csv``.
5. After the agent finishes, copy ``evaluate.py`` / ``evaluate_prepare.py``
   into ``<ws>/_eval/``, run ``evaluate_prepare.py`` (via ``uv run --with ...``
   using ``evaluate_container_python_requirements`` ∪ container reqs) to
   materialize the labelled test set and copy over the submission, then run
   ``evaluate.py`` the same way. Parse the JSON block from evaluate's stdout
   and persist ``<ws>/final_score.json``.

Per-task records accumulate in ``<output_root>/results.json``.

Usage
-----
    python -m bench_workflows.airsbench_workflow \\
        --tasks-root   benchmarks/airsbench/airs-bench/airsbench/tasks/rad \\
        --data-root    /home/klin/data/SciDER/airs-bench/datasets_download_location \\
        --output-root  benchmarks/airsbench/workspace \\
        --skip-existing
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from scider.core.skills import SkillRegistry
from scider.default.models import register_defaults_from_yaml
from scider.workflows.full_workflow import run_full_workflow

# ---- Constants ----
SUBMISSION_FILENAME = "submission.csv"
FINAL_SCORE_FILENAME = "final_score.json"
DATA_SUBDIR = "data"
EVAL_SUBDIR = "_eval"

# Files copied INTO the agent's workspace (agent may read/edit these).
AGENT_VISIBLE_FILES = ("project_description.md", "metadata.yaml", "prepare.py", "utils.py")
# Files copied INTO the post-agent eval sandbox (agent never sees them).
EVALUATOR_FILES = ("evaluate.py", "evaluate_prepare.py", "utils.py")

ROLES_YAML_PATH = Path(__file__).parent / "model_configs" / "airsbench_roles.yaml"

# Skills we force-preload for this benchmark specifically. The skill's own
# frontmatter leaves preload_for=[] (so it's on-demand elsewhere); we override
# here so every experiment/coding agent run starts with the HF-model recipe
# already in the system prompt — AIRS-Bench tasks all depend on picking the
# right backbone, and not having to re-discover the skill saves a round-trip.
_PROJECT_ROOT = Path(__file__).parent.parent
PRELOAD_SKILLS: tuple[tuple[Path, tuple[str, ...]], ...] = (
    (
        _PROJECT_ROOT / ".scider" / "skills" / "huggingface-model-search",
        ("experiment", "native_coding"),
    ),
)


def _preload_airsbench_skills() -> None:
    """Force-register the HF model-search skill as preloaded for this workflow."""
    registry = SkillRegistry.instance()
    for skill_dir, preload_for in PRELOAD_SKILLS:
        if not (skill_dir / "SKILL.md").is_file():
            logger.warning("Skill dir missing SKILL.md: {}", skill_dir)
            continue
        registry.register_skill_dirs(skill_dir, preload_for=preload_for)
        logger.info("Preloaded skill from {} for agents {}", skill_dir, preload_for)


def _fmt_sota_list(sota_entries: list[dict]) -> str:
    """Render the metadata's SOTA list as a compact bullet section."""
    lines: list[str] = []
    for e in sota_entries or []:
        title = (e.get("sota_paper_title") or "").strip()
        url = (e.get("sota_paper_url") or "").strip()
        score = e.get("sota_score")
        year = e.get("sota_year")
        venue = (e.get("sota_venue") or "").strip()
        bits = [f"**{title}**" if title else "**(untitled)**"]
        if score is not None:
            bits.append(f"score={score}")
        if year:
            bits.append(f"year={year}")
        if venue:
            bits.append(venue)
        head = " — ".join(bits)
        lines.append(f"  - {head}" + (f"  ({url})" if url else ""))
    return "\n".join(lines) if lines else "  - (no SOTA listed in metadata)"


def _build_preamble(metadata: dict, evaluate_source: str | None = None) -> str:
    """Render a task-specific strategy preamble from ``metadata.yaml``.

    Inlining the key fields (task name, dataset, metric, output type, shape,
    optimal/worst score, prior SOTA) saves the agent a `Read metadata.yaml`
    round-trip AND surfaces the SOTA paper URL + year so its paper search can
    anchor on something concrete instead of guessing from the dataset name.
    """
    info = metadata.get("logging_info") or {}
    name = info.get("name") or "<unknown>"
    dataset = info.get("dataset") or "<unknown>"
    metric = info.get("metric") or "<unknown>"
    add_metrics = info.get("additional_metrics") or None
    research_problem = info.get("research_problem") or "<unknown>"
    category = info.get("category") or "<unknown>"
    output_type = info.get("output_type") or "<unknown>"
    shape = info.get("shape")
    optimal = info.get("optimal_score")
    worst = info.get("estimated_worst_score")
    metric_lower = metadata.get("metric_lower_is_better")
    sota_block = _fmt_sota_list(info.get("sota") or [])
    dataset_paper = (info.get("dataset_paper_url") or "").strip()

    grader_block: list[str] = []
    if evaluate_source:
        grader_block = [
            "",
            "## Grader source — this is EXACTLY how your submission is scored",
            "",
            "The following is the content of `evaluate.py` that will run on your",
            "final `submission.csv`. Your submission MUST satisfy every check in",
            "it — most critically the row-count assertion. Use this same scoring",
            "function to self-score on the `validation` split (which HAS labels)",
            "before writing your final submission.",
            "",
            "```python",
            evaluate_source.strip(),
            "```",
        ]

    header = [
        "# Solution Strategy (read before the task spec below)",
        "",
        "## Task at a glance (extracted from metadata.yaml)",
        "",
        f"- **Task**: `{name}`",
        f"- **Category / problem**: {category} — {research_problem}",
        f"- **Dataset (HF id)**: `{dataset}`" + (f"  ({dataset_paper})" if dataset_paper else ""),
        f"- **Output type**: {output_type}",
        f"- **Target metric**: {metric}"
        + (f"  (additional: {add_metrics})" if add_metrics else "")
        + (
            f"  — lower is better"
            if metric_lower
            else f"  — higher is better" if metric_lower is False else ""
        ),
        f"- **Submission shape**: {shape}",
        f"- **Optimal / worst reference score**: optimal={optimal}, " f"estimated_worst={worst}",
        "",
        "### Prior SOTA (from the metadata — USE THIS AS A STARTING POINT, NOT A CEILING)",
        "",
        sota_block,
        "",
        "The SOTA above may be years old. **Your job is to beat it** — look for",
        "newer work (papers, HF models, leaderboards) published AFTER the year",
        "shown. Anchor your literature search on the exact dataset name above.",
    ]

    strategy = [
        "",
        "## How to solve this (approach)",
        "",
        "You are solving a single task on a single machine. Training a model",
        "from scratch is almost never the right call — prefer the following:",
        "",
        "1. **Hunt for newer SOTA first — this is step one, not optional.**",
        "   Before writing any training code, use the `WebSearch` tool (and",
        "   `paper_search` if available) to look for **2025 / 2026** work that",
        "   beats the prior SOTA listed above. The SOTA in metadata is often",
        "   2023-or-earlier; entire model families (Qwen3/3.5, Llama 3.1/4,",
        "   Gemma 3, ModernBERT, Moirai 2.0, Qwen3-Embedding, ...) landed",
        "   AFTER that and routinely beat older numbers.",
        "   Concrete queries to run (substitute the real dataset name):",
        '   - `WebSearch(query="<dataset> state of the art 2025 2026")`',
        '   - `WebSearch(query="<dataset> <metric> benchmark leaderboard 2025")`',
        '   - `WebSearch(query="<dataset> huggingface model fine-tune 2025")`',
        "   - If a SOTA paper is listed, also search its title + 'follow-up'",
        "     or 'improved' to find successor papers that cite it.",
        "   Always include a year (2025/2026) filter in the query — without",
        "   it, search engines surface old blog posts first. Check Papers With",
        "   Code, HuggingFace leaderboards, and ACL/NeurIPS/ICLR/CVPR recent",
        "   tracks. Record the best reported number and the backbone it used.",
        "2. **Find a HuggingFace-hosted checkpoint that matches.** Load the",
        "   `huggingface-model-search` skill for `HfApi.list_models` filters,",
        "   verification snippets, and LoRA/QLoRA loading code. Prefer a model",
        "   whose paper / release is MORE RECENT than the prior SOTA. The skill",
        "   includes a shortlist of strong open ~7B backbones as a safety net —",
        "   it's a fallback, not the target. Don't restrict yourself to it.",
        "3. **Model size target: ~7B parameters, at most 16B.** Larger won't",
        "   fit a single-GPU LoRA run; much smaller than 1B is usually too weak",
        "   unless the task is well-matched to the architecture.",
        "4. **Fine-tune with PEFT / LoRA, not full training.** Use `peft` (LoRA,",
        "   QLoRA, adapter variants as appropriate) on top of the frozen base",
        "   for any 1B+ checkpoint. Few epochs, modest LR.",
        "   **Exception — small models get all-parameter fine-tuning (AFT).**",
        "   If the base is <1B (BERT-base, RoBERTa-large, T5-base/large, small",
        "   GNN), full fine-tuning is cheap and typically beats LoRA.",
        "5. **No-training baselines are fair game.** For some tasks a strong",
        "   zero-shot or few-shot prompt against an instruction-tuned LLM beats",
        "   a quick fine-tune.",
        "6. **Budget your compute.** Pick batch size / seq len / LoRA rank that",
        "   fit VRAM without OOM. Don't sweep hyperparameters exhaustively —",
        "   one sane config is enough.",
        "7. **Verify before you submit — this is a HARD gate, not optional.**",
        "   Before declaring done you MUST do all of the following:",
        "   a) **Row-count sanity check**:",
        "      ```bash",
        f"      wc -l submission.csv   # must equal {shape[0] if isinstance(shape, (list, tuple)) and shape else '<expected rows>'} + 1 (header)",
        "      head -3 submission.csv",
        "      ```",
        f"      The evaluator will reject the submission if the row count is",
        f"      not exactly `{shape}`.",
        "   b) **Self-score on the `validation` split using the grader's own",
        "      `evaluate()` function** (see the `Grader source` code block",
        "      above). Copy that function verbatim, run your trained model on",
        "      `./data/validation` (which has labels), call `evaluate(preds,",
        "      labels)`, and print the metric. This is your proxy for the real",
        "      score — if it's near `estimated_worst_score` your model hasn't",
        "      learned; don't submit until it clears a reasonable bar above",
        "      worst (aim for halfway between worst and the prior SOTA at a",
        "      minimum).",
        "   c) **Distribution check**: print `Counter(preds)` — an all-one-class",
        "      output (e.g. all zeros) means the model collapsed; iterate",
        "      before submitting.",
        "   Common ways this goes wrong: leaving `test_mode=True` in your",
        "   training script (trains on a tiny slice, then predicts only that",
        "   slice), predicting on `validation` instead of `test`, or writing",
        "   predictions only for a sampled subset. Always predict on the FULL",
        "   test set and double-check the row count.",
        "",
        "---",
        "",
    ]
    return "\n".join(header + grader_block + strategy)


# --------------------------------------------------------------------------- #
# Task discovery                                                              #
# --------------------------------------------------------------------------- #


def discover_tasks(tasks_root: Path) -> list[str]:
    """Return the sorted list of task names under ``tasks_root``.

    A task is any direct subdir containing a ``metadata.yaml`` file. Stray
    files (e.g. ``test_task_folder.py``) are skipped.
    """
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"tasks_root is not a directory: {tasks_root}")
    names: list[str] = []
    for child in sorted(tasks_root.iterdir()):
        if child.is_dir() and (child / "metadata.yaml").is_file():
            names.append(child.name)
    if not names:
        raise FileNotFoundError(f"No tasks (dirs with metadata.yaml) found under {tasks_root}")
    return names


def scan_completed_tasks(output_root: Path) -> set[str]:
    """Return task names that already have a valid ``final_score.json``."""
    if not output_root.is_dir():
        return set()
    done: set[str] = set()
    for d in output_root.iterdir():
        if not d.is_dir():
            continue
        score = d / FINAL_SCORE_FILENAME
        if score.is_file() and score.stat().st_size > 0:
            try:
                json.loads(score.read_text(encoding="utf-8"))
                done.add(d.name)
            except json.JSONDecodeError:
                continue
    return done


# --------------------------------------------------------------------------- #
# Metadata + uv helpers                                                       #
# --------------------------------------------------------------------------- #


def _read_metadata(task_dir: Path) -> dict:
    with (task_dir / "metadata.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _uv_init_workspace(workspace: Path) -> None:
    """Run ``uv init`` on an empty workspace so ``uv run`` works inside it.

    Idempotent: if ``pyproject.toml`` already exists, this is a no-op. The
    flags mirror what ``LocalEnv`` uses, so when the agent starts and LocalEnv
    checks for pyproject.toml, it sees ours and skips its own init.
    """
    if (workspace / "pyproject.toml").is_file():
        return
    r = subprocess.run(
        ["uv", "init", "--no-readme", "--no-pin-python", "--vcs", "none"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"uv init failed in {workspace} (rc={r.returncode})\n"
            f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        )
    # `uv init` creates a stub `main.py` — delete it so the agent isn't
    # confused by a 'hello world' sitting in the workspace root.
    stub = workspace / "main.py"
    if stub.is_file():
        stub.unlink()


def _uv_run_with(
    workspace: Path,
    reqs: list[str],
    script: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 1800,
) -> subprocess.CompletedProcess:
    """Run ``uv run --with <reqs...> <script...>`` inside ``workspace``'s project.

    ``--with`` layers each dep on top of the workspace's base env for just this
    invocation, leaving ``pyproject.toml`` untouched — so the agent's uv env
    isn't polluted with eval-only deps like torchmetrics.
    """
    with_args: list[str] = []
    for r in reqs:
        with_args += ["--with", r]
    cmd = ["uv", "run", "--project", str(workspace), *with_args, *script]
    logger.info("Running: {}", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=cwd or workspace,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# --------------------------------------------------------------------------- #
# Workspace setup                                                             #
# --------------------------------------------------------------------------- #


def _copy_agent_files(task_dir: Path, workspace: Path) -> None:
    """Copy only the files the agent is allowed to see."""
    for fname in AGENT_VISIBLE_FILES:
        src = task_dir / fname
        if src.is_file():
            shutil.copy2(src, workspace / fname)


def _run_prepare(
    workspace: Path,
    data_root: Path,
    container_reqs: list[str],
) -> None:
    """Run ``prepare.py`` to materialize ``data/{train,test,validation}``."""
    data_dir = workspace / DATA_SUBDIR
    data_dir.mkdir(parents=True, exist_ok=True)
    prepare_py = workspace / "prepare.py"
    if not prepare_py.is_file():
        raise FileNotFoundError(f"prepare.py missing in {workspace}")
    r = _uv_run_with(
        workspace,
        container_reqs,
        [
            "python",
            str(prepare_py),
            "--global-shared-data-dir",
            str(data_root),
            "--agent-data-mount-dir",
            str(data_dir),
        ],
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"prepare.py failed (rc={r.returncode})\n"
            f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-2000:]}"
        )


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #


_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_eval_result(stdout: str) -> dict:
    """Extract the JSON metric dict from ``evaluate.py``'s stdout.

    ``evaluate.py`` prints ``--- EVALUATION RESULT ---`` then a
    ``json.dumps(result, indent=2)`` block. We find the LAST JSON-looking
    brace block. Uses DOTALL since ``indent=2`` injects newlines.
    """
    candidates = list(_JSON_BLOCK_RE.finditer(stdout))
    for m in reversed(candidates):
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed:
            return parsed
    raise ValueError(f"Could not find JSON result in evaluate.py stdout:\n{stdout[-2000:]}")


def _run_evaluation(
    task_dir: Path,
    workspace: Path,
    data_root: Path,
    eval_reqs: list[str],
) -> dict:
    """Copy evaluator files into ``<ws>/_eval/``, run the eval pipeline, return the score.

    Layout after this function:
      ``<ws>/_eval/evaluate.py``, ``<ws>/_eval/evaluate_prepare.py``
      ``<ws>/_eval/data/test_with_labels/``   (created by evaluate_prepare)
      ``<ws>/_eval/data/submission.csv``      (copied by evaluate_prepare)
    ``evaluate.py`` references ``./data/test_with_labels`` so we run it with
    ``cwd=<ws>/_eval``. The uv project is the workspace itself (``--project
    <ws>`` inside ``_uv_run_with``), so eval can reuse the agent's env.
    """
    eval_dir = workspace / EVAL_SUBDIR
    eval_dir.mkdir(parents=True, exist_ok=True)

    for fname in EVALUATOR_FILES:
        src = task_dir / fname
        if src.is_file():
            shutil.copy2(src, eval_dir / fname)
    if not (eval_dir / "evaluate.py").is_file():
        raise FileNotFoundError(f"evaluate.py missing in source task dir {task_dir}")
    if not (eval_dir / "evaluate_prepare.py").is_file():
        raise FileNotFoundError(f"evaluate_prepare.py missing in source task dir {task_dir}")

    submission = workspace / SUBMISSION_FILENAME
    if not submission.is_file():
        raise FileNotFoundError(f"Agent did not produce {submission}")

    eval_data_dir = eval_dir / DATA_SUBDIR
    eval_data_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: evaluate_prepare.py — reads submission.csv from --agent-log-dir.
    r = _uv_run_with(
        workspace,
        eval_reqs,
        [
            "python",
            str(eval_dir / "evaluate_prepare.py"),
            "--global-shared-data-dir",
            str(data_root),
            "--agent-data-mount-dir",
            str(eval_data_dir),
            "--agent-log-dir",
            str(workspace),
        ],
        cwd=eval_dir,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"evaluate_prepare.py failed (rc={r.returncode})\n"
            f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-2000:]}"
        )

    # Step 2: evaluate.py — loads ./data/test_with_labels + --submission-file.
    r = _uv_run_with(
        workspace,
        eval_reqs,
        [
            "python",
            str(eval_dir / "evaluate.py"),
            "--submission-file",
            str(eval_data_dir / SUBMISSION_FILENAME),
        ],
        cwd=eval_dir,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"evaluate.py failed (rc={r.returncode})\n"
            f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-2000:]}"
        )

    return _parse_eval_result(r.stdout)


# --------------------------------------------------------------------------- #
# Per-task driver                                                             #
# --------------------------------------------------------------------------- #


def run_one_task(
    *,
    task_name: str,
    task_dir: Path,
    data_root: Path,
    output_root: Path,
    max_revisions: int,
    data_recursion_limit: int,
    experiment_recursion_limit: int,
) -> dict:
    """Run the full pipeline (setup → prepare → agent → evaluate) for one task."""
    workspace = (output_root / task_name).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "task": task_name,
        "workspace": str(workspace),
    }

    try:
        metadata = _read_metadata(task_dir)
        container_reqs = list(metadata.get("container_python_requirements") or [])
        evaluate_reqs = list(metadata.get("evaluate_container_python_requirements") or [])
        # Eval scripts may need BOTH the container deps (e.g. `datasets`) and
        # the eval-only ones (e.g. `torchmetrics`). Preserve order, dedup.
        eval_union = list({r: None for r in (*evaluate_reqs, *container_reqs)}.keys())

        # 1. Pre-init the workspace so `uv run` works AND so LocalEnv won't
        #    re-init when the agent starts.
        _uv_init_workspace(workspace)

        # 2. Stage agent-visible files (NOT evaluate*.py).
        _copy_agent_files(task_dir, workspace)
        description_path = workspace / "project_description.md"
        if not description_path.is_file():
            raise FileNotFoundError(f"project_description.md missing for task {task_name}")
        # Read evaluate.py source (but do NOT copy it into the workspace —
        # the agent sees the scoring logic inline in the prompt, not as an
        # editable file). evaluate_prepare.py is still withheld entirely to
        # avoid leaking the test-label materialization path.
        evaluate_source: str | None = None
        eval_src = task_dir / "evaluate.py"
        if eval_src.is_file():
            evaluate_source = eval_src.read_text(encoding="utf-8")
        user_query = _build_preamble(metadata, evaluate_source) + description_path.read_text(
            encoding="utf-8"
        )

        # 3. Materialize the dataset.
        _run_prepare(workspace, data_root, container_reqs)

        # 4. Hand to FullWorkflow (data + experiment). Agent writes
        #    submission.csv to the workspace root. We don't pass a custom
        #    WorkspaceInitConfig — the default (env_manager='uv',
        #    init_uv=True) is correct, and LocalEnv will skip re-init because
        #    we already created pyproject.toml in step 1.
        logger.info("Running FullWorkflow for task={} in {}", task_name, workspace)
        wf = run_full_workflow(
            data_path=workspace / DATA_SUBDIR,
            workspace_path=workspace,
            user_query=user_query,
            data_desc=None,
            max_revisions=max_revisions,
            data_agent_recursion_limit=data_recursion_limit,
            experiment_agent_recursion_limit=experiment_recursion_limit,
            user_approval_enabled=False,
        )
        record["workflow_status"] = wf.final_status

        submission = workspace / SUBMISSION_FILENAME
        if not submission.is_file():
            raise RuntimeError(
                f"Agent finished ({wf.final_status}) but {submission} was not created."
            )
        if submission.stat().st_size == 0:
            raise RuntimeError(f"{submission} is empty (0 bytes).")

        # 5. Score.
        score = _run_evaluation(task_dir, workspace, data_root, eval_union)
        (workspace / FINAL_SCORE_FILENAME).write_text(json.dumps(score, indent=2), encoding="utf-8")
        record.update(ok=True, score=score)
        logger.info("task={} scored {}", task_name, score)
    except Exception as e:
        logger.exception("task={} failed: {}", task_name, e)
        record.update(ok=False, error=str(e))
    return record


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _register_models_from_yaml() -> None:
    if not ROLES_YAML_PATH.exists():
        raise FileNotFoundError(f"Role yaml missing at {ROLES_YAML_PATH}")
    logger.info("Registering roles from {}", ROLES_YAML_PATH)
    registered = register_defaults_from_yaml(ROLES_YAML_PATH)
    logger.info("Registered {} roles: {}", len(registered), sorted(registered))


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AIRS-Bench workflow — per-task FullWorkflow run that produces "
            "submission.csv and scores it against the held-out test labels."
        ),
        prog="python -m bench_workflows.airsbench_workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks-root",
        required=True,
        help="Path to airs-bench/airsbench/tasks/rad/ (contains one subdir per task).",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to the AIRS-Bench raw-data download location "
        "(passed to prepare.py as --global-shared-data-dir).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory under which per-task workspaces are created "
        "(<output_root>/<TaskName>/). Must be an area that doesn't conflict "
        "with other uv projects — each task gets its own uv env inside.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated list of task names to run (overrides auto-discover). "
        "Useful for dev/debug: --tasks TextualClassificationSickAccuracy",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip any task whose <output_root>/<task>/final_score.json already "
        "exists and parses as JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Dev knob: process at most the first N tasks (after --skip-existing "
        "and --tasks filtering).",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=3,
        help="Upper bound on critic/approval retries (default 3).",
    )
    parser.add_argument(
        "--data-recursion-limit",
        type=int,
        default=512,
        help="Recursion limit for DataAgent (default 512).",
    )
    parser.add_argument(
        "--experiment-recursion-limit",
        type=int,
        default=512,
        help="Recursion limit for ExperimentAgent (default 512).",
    )

    args = parser.parse_args()

    _register_models_from_yaml()
    _preload_airsbench_skills()

    tasks_root = Path(args.tasks_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_tasks = discover_tasks(tasks_root)
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        missing = wanted - set(all_tasks)
        if missing:
            logger.warning("Requested tasks not found under tasks_root: {}", sorted(missing))
        tasks = [t for t in all_tasks if t in wanted]
    else:
        tasks = list(all_tasks)

    if args.skip_existing:
        done = scan_completed_tasks(output_root)
        if done:
            logger.info("--skip-existing: skipping {} already-scored tasks", len(done))
            tasks = [t for t in tasks if t not in done]

    if args.limit is not None:
        tasks = tasks[: args.limit]

    logger.info("Processing {} tasks: {}", len(tasks), tasks)

    results_path = output_root / "results.json"
    results: list[dict] = []
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        logger.info("Resuming from {} ({} prior records)", results_path, len(results))

    for name in tasks:
        record = run_one_task(
            task_name=name,
            task_dir=tasks_root / name,
            data_root=data_root,
            output_root=output_root,
            max_revisions=args.max_revisions,
            data_recursion_limit=args.data_recursion_limit,
            experiment_recursion_limit=args.experiment_recursion_limit,
        )
        results = [r for r in results if r.get("task") != record["task"]]
        results.append(record)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.get("ok"))
    logger.info("Done. {}/{} tasks scored. Results at {}", n_ok, len(results), results_path)


if __name__ == "__main__":
    _main()
