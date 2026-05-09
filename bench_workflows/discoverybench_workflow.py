"""
Workflow for DiscoveryBench — hand each (dataset, metadata, query) triple to
SciDER's full workflow (data agent + experiment agent) and have it produce a
natural-language hypothesis directly.

Strategy (vs. upstream)
-----------------------
Upstream's ``discovery_agent.py`` runs a single Coder/ReAct LangChain agent
over the CSV + NL question and emits a hypothesis (+ optional workflow). We
swap that for SciDER's ``run_full_workflow`` (DataWorkflow → ExperimentWorkflow,
no ideation, no paper writing). The deliverable is the same shape the upstream
scorer wants:

    ./result.json  →  {"hypothesis": "...", "workflow": "..."}

Per-uid workspaces look like:

    <output_root>/<uid>/
    ├── inputs/                       # symlinks to upstream CSV(s)
    ├── metadata.json                 # copy of metadata_N.json (agent-readable)
    ├── task.md                       # persisted user_query
    ├── result.json                   # the deliverable
    └── (data_agent_history.json, experiment_agent_history.json, ...)

The unit of work is one ``(task_dir, metadata_N, qid)`` triple — each
metadata file can carry multiple queries, and each (task_dir, metadataid, qid)
row in the upstream answer key is scored independently. uid format:

    "{dataset}__m{metadataid}__q{qid}"     e.g. "archaeology__m0__q42"

Eval is intentionally separate — see
``benchmarks/discoverybench/discoverybench/eval/discovery_eval_scider.py``.

Usage
-----
    python -m bench_workflows.discoverybench_workflow \\
        --bench-root  benchmarks/discoverybench/discoverybench/discoverybench \\
        --split-type  real \\
        --split       test \\
        --output-root benchmarks/discoverybench/workspace \\
        --skip-existing
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from loguru import logger

# Make scider + bench modules importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scider.default.models import register_defaults_from_yaml
from scider.workflows.full_workflow import run_full_workflow

# ---- Constants ----
RESULT_FILENAME = "result.json"
METADATA_FILENAME = "metadata.json"
INPUTS_SUBDIR = "inputs"
TASK_PROMPT_FILENAME = "task.md"
ROLES_YAML_PATH = Path(__file__).parent / "model_configs" / "discoverybench_roles.yaml"

# metadata_N.json — N is the metadataid we surface in the uid.
_METADATA_RE = re.compile(r"^metadata_(\d+)\.json$")


# --------------------------------------------------------------------------- #
# Task model + discovery                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class Task:
    """One unit of work: a single (dataset, metadata file, query) triple."""

    uid: str
    split_type: Literal["synth", "real"]
    split: str
    dataset: str
    metadataid: int
    qid: int
    metadata_path: Path
    metadata: dict
    query_text: str
    csv_paths: list[Path]


def _build_uid(dataset: str, metadataid: int, qid: int) -> str:
    return f"{dataset}__m{metadataid}__q{qid}"


def _resolve_csv_paths(metadata: dict, metadata_dir: Path) -> list[Path]:
    """Resolve every ``datasets[].name`` against the metadata's directory.

    Missing files raise — these benchmarks ship the CSVs alongside the
    metadata, so a missing path indicates a bad clone, not a usable task.
    """
    paths: list[Path] = []
    for ds in metadata.get("datasets") or []:
        name = ds.get("name")
        if not name:
            continue
        p = (metadata_dir / name).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"CSV referenced by metadata not found: {p}")
        paths.append(p)
    if not paths:
        raise ValueError(f"No datasets[] entries in metadata at {metadata_dir}")
    return paths


def _iter_queries(metadata: dict) -> Iterator[dict]:
    """Yield every query dict from ``metadata.queries``.

    Real-test metadata stores queries as a list of lists (one inner list per
    "question group"); synth stores a flat list. Flatten both shapes here so
    discover_tasks doesn't need to know.
    """
    for entry in metadata.get("queries") or []:
        if isinstance(entry, list):
            for q in entry:
                if isinstance(q, dict):
                    yield q
        elif isinstance(entry, dict):
            yield entry


def discover_tasks(
    bench_root: Path,
    split_type: Literal["synth", "real"],
    split: str,
) -> Iterator[Task]:
    """Walk ``<bench_root>/<split_type>/<split>/<task>/metadata_N.json`` and
    yield one ``Task`` per (task, metadataid, qid).
    """
    split_root = bench_root / split_type / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"split dir not found: {split_root}")

    for task_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        dataset = task_dir.name
        for meta_path in sorted(task_dir.iterdir()):
            m = _METADATA_RE.match(meta_path.name)
            if not m:
                continue
            metadataid = int(m.group(1))
            with meta_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            csv_paths = _resolve_csv_paths(metadata, meta_path.parent)
            for q in _iter_queries(metadata):
                qid = int(q["qid"])
                yield Task(
                    uid=_build_uid(dataset, metadataid, qid),
                    split_type=split_type,
                    split=split,
                    dataset=dataset,
                    metadataid=metadataid,
                    qid=qid,
                    metadata_path=meta_path,
                    metadata=metadata,
                    query_text=q["question"],
                    csv_paths=csv_paths,
                )


# --------------------------------------------------------------------------- #
# Result.json validation + scan-completed                                     #
# --------------------------------------------------------------------------- #


def is_valid_result_json(path: Path) -> bool:
    """A result.json counts as valid only if it parses, has a ``hypothesis``
    key, and that key holds a non-empty string. ``workflow`` is optional —
    upstream's scorer accepts an empty workflow.
    """
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    hypo = data.get("hypothesis")
    return isinstance(hypo, str) and bool(hypo.strip())


def scan_completed_uids(output_root: Path) -> set[str]:
    """Used by ``--skip-existing``: returns uids whose result.json passes
    ``is_valid_result_json``."""
    if not output_root.is_dir():
        return set()
    completed: set[str] = set()
    for uid_dir in output_root.iterdir():
        if not uid_dir.is_dir():
            continue
        if is_valid_result_json(uid_dir / RESULT_FILENAME):
            completed.add(uid_dir.name)
    return completed


def read_result_json(path: Path) -> dict | None:
    """Parse the deliverable. Returns ``None`` if unreadable / invalid; the
    eval companion uses this to label a record as ``Miss``."""
    if not is_valid_result_json(path):
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Workspace staging                                                           #
# --------------------------------------------------------------------------- #


def _stage_inputs(workspace: Path, csv_paths: list[Path]) -> list[str]:
    """Symlink each CSV into ``<workspace>/inputs/``. Returns the ordered list
    of bare CSV filenames so the prompt builder can reference them by name."""
    inputs_dir = workspace / INPUTS_SUBDIR
    inputs_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for src in csv_paths:
        dst = inputs_dir / src.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        names.append(src.name)
    return names


def _copy_metadata(
    workspace: Path,
    metadata: dict,
    *,
    use_domain_knowledge: bool = False,
    use_workflow_tags: bool = False,
) -> Path:
    """Persist a copy of metadata_N.json at ``<workspace>/metadata.json`` so
    the agent has a stable filename to read regardless of which metadata index
    was selected upstream.

    The on-disk copy is sanitized so it can't leak information the prompt
    deliberately withholds:

    * ``hypotheses`` and ``intermediate`` are stripped — these fields are
      empty across test split today, but stripping unconditionally guards
      against future upstream changes that might populate them.
    * ``domain_knowledge`` / ``workflow_tags`` are stripped unless the
      caller opts in via the matching CLI flags.
    """
    sanitized = dict(metadata)
    sanitized.pop("hypotheses", None)
    sanitized.pop("intermediate", None)
    if not use_domain_knowledge:
        sanitized.pop("domain_knowledge", None)
    if not use_workflow_tags:
        sanitized.pop("workflow_tags", None)
    out = workspace / METADATA_FILENAME
    out.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Prompt building                                                             #
# --------------------------------------------------------------------------- #


def _format_columns_table(metadata: dict) -> str:
    """Render every dataset's columns as ``- name — description`` bullets,
    grouped per-CSV. Real metadata's ``columns`` may be nested under ``raw``;
    handle both shapes.
    """
    blocks: list[str] = []
    for ds in metadata.get("datasets") or []:
        name = ds.get("name", "<unknown>")
        desc = (ds.get("description") or "").strip()
        cols = ds.get("columns")
        if isinstance(cols, dict) and "raw" in cols:
            cols = cols["raw"]
        cols = cols or []
        bullet_lines = [
            f"  - `{c.get('name')}` — {(c.get('description') or '').strip() or '(no description)'}"
            for c in cols
        ]
        header = f"### `./{INPUTS_SUBDIR}/{name}`"
        if desc:
            header += f" — {desc}"
        blocks.append(header + "\n" + "\n".join(bullet_lines))
    return "\n\n".join(blocks)


def _build_user_query(
    task: Task,
    csv_names: list[str],
    use_domain_knowledge: bool = False,
    use_workflow_tags: bool = False,
) -> str:
    """Build the experiment-agent prompt for one DiscoveryBench task.

    The prompt is structured so the deliverable contract (``./result.json``)
    appears at the top — we want the agent to anchor on what it must produce
    before it disappears into the analysis.
    """
    csv_bullets = "\n".join(f"- `./{INPUTS_SUBDIR}/{n}`" for n in csv_names)

    parts: list[str] = [
        "# DiscoveryBench: Hypothesis Discovery Task",
        "",
        "Your deliverable is a single JSON file at "
        f"`./{RESULT_FILENAME}` in this workspace, with EXACTLY this schema:",
        "",
        "```json",
        '{"hypothesis": "<one or two sentences stating the discovered relationship>",',
        ' "workflow": "<evidence-backed analytical steps that justify the hypothesis>"}',
        "```",
        "",
        "Both fields must be non-empty strings. The hypothesis will be "
        "scored by an LLM-as-judge against a held-out gold hypothesis "
        "along three axes (context, variables, relations) — be specific "
        "about which columns are involved and what form of relationship "
        "you find.",
        "",
        "### Why this matters (empirical priors)",
        "",
        "On a 239-task pilot run, **84% of zero scores were caused by "
        "framing / context-mismatch**, not by wrong numbers. The judge "
        "first decomposes both hypotheses into sub-hypotheses, matches "
        "them by **context phrase**, and scores ONLY matched pairs. If "
        "your context phrase doesn't match the gold's, you score 0 even "
        "when your underlying numbers are exactly right. The five rules "
        "below address every common framing-mismatch pattern observed.",
        "",
        "### RULE 1 — Lift scope words verbatim from the question",
        "",
        "If the question says 'in Psychology', your hypothesis says 'in "
        "Psychology' — NOT 'in Psychology, specifically Social and "
        "Cognitive disciplines'. Do NOT add sub-categories, project "
        "codes, time windows, or cohort identifiers that the question "
        "did NOT ask about. Extra specificity is not rewarded — it "
        "punishes you by breaking context match.",
        "",
        "### RULE 2 — Mirror the question's answer shape",
        "",
        "- **Comparative** ('Is A higher than B?', 'Which domain has "
        "  more Y?'): hypothesis MUST name BOTH sides AND the direction. "
        "  Write 'Psychology has 84 citations vs Experimental Economics' "
        "  80, so Psychology > Experimental Economics' — NOT 'Experimental "
        "  Economics has 80 citations'. Single-side stats break the "
        "  context match even when the underlying number is right.",
        "- **Relational** ('What is the relationship between X and Y?'): "
        "  hypothesis MUST state the type (linear / monotonic / inverse / "
        "  threshold / none) AND the strength (correlation r, slope, "
        "  effect size).",
        "- **Locative** ('In which century did X peak?'): hypothesis "
        "  MUST give the located value AND a supporting magnitude (e.g. "
        "  'peak at 32nd century BCE with z=2.6 vs next-highest 1.3').",
        "- **Conditional / restricted** ('among women aged 25-50', 'in "
        "  studies after 2010'): repeat the restriction VERBATIM in the "
        "  hypothesis. Do not paraphrase or generalise the cohort.",
        "- **Direct-answer** ('Which variable has property X?', 'What "
        "  type of compensation is used?'): hypothesis MUST start with "
        "  the literal answer — 'The variable XYZ ...', 'The compensation "
        "  type is ABC ...'. Do NOT write a process narrative.",
        "",
        "### RULE 3 — Use the column name the question uses, not your " "derived column",
        "",
        "If you analyzed `Depot_inter` (interpolated) but the question "
        "says 'Depots', the column name in your hypothesis and `variables` "
        "list must be `Depot` (or `Depot (analyzed via the interpolated "
        "Depot_inter)`). The judge's variable-set F1 is computed against "
        "the gold's column names, which use the conceptual / primary "
        "names, not derived / `_inter` / `_smoothed` ones.",
        "",
        "### RULE 4 — Variables = subjects of the claim, NOT every column " "you touched",
        "",
        "If your hypothesis is about Psychology vs Experimental Economics "
        "paper length, the variables are {discipline, length} — NOT "
        "{discipline, length, citations, n_authors, year, project_id}. "
        "Listing every column you happened to inspect during EDA dilutes "
        "the variable F1.",
        "",
        "### RULE 5 — Direction-sign self-check",
        "",
        "For every directional / comparative word in the hypothesis "
        "('increased / decreased', 'converged / diverged', 'positive / "
        "negative correlation', 'higher / lower', 'more / fewer'), "
        "re-read the underlying statistic value and confirm the sign. "
        "A reversed direction returns 0 even when context and variables "
        "match perfectly. Pay special attention to: convergence vs "
        "divergence, before/after comparisons, BCE/CE sign conventions.",
        "",
        "### Evidence-based requirement",
        "",
        "Every numeric claim in `hypothesis` must be supported by a "
        "concrete computed value recorded in `workflow`. The `workflow` "
        "field documents, for each step: which CSV(s) + column(s) the "
        "step touched (using the names per RULE 3), what was computed "
        "with its actual value (e.g. `Pearson r = -0.495 (n=120, "
        "p<0.001)`, `mean=3.4 ± 0.7`, `peak at year=3200 BCE`), and how "
        "that supports the hypothesis. Vague phrasings like 'a strong "
        "correlation' without the number are NOT acceptable. State and "
        "justify any analytical assumptions (ordinal treatment, outlier "
        "drops, interpolation) explicitly.",
        "",
        "### Mandatory pre-submission self-review",
        "",
        "Before writing `./result.json`, run through this checklist "
        "explicitly in your reasoning (don't skip — this is the single "
        "highest-leverage step you can take):",
        "",
        "1. **Question echo**: Quote the original question on one line; "
        "   write your hypothesis directly below. Do they share the same "
        "   scope words (per RULE 1)?",
        "2. **Answer-shape match**: Does your hypothesis sentence have "
        "   the same shape the question implies (per RULE 2)?",
        "3. **Direction-sign check**: For each directional word in the "
        "   hypothesis, name the supporting statistic value and confirm "
        "   the sign matches (per RULE 5).",
        "4. **Variables**: Are these the columns the QUESTION names "
        "   (per RULE 3), and only the subjects of the claim (per RULE 4)?",
        "5. **Numeric backing**: Every numeric claim in hypothesis has a "
        "   matching workflow step with the actual computed value.",
        "",
        "If any check fails, REWRITE the hypothesis before finalizing.",
        "",
        "### Critic / approval review checklist",
        "",
        "The critic / approval reviewer is instructed to FORCE A REVISION "
        "if it finds any of these defects (these mirror the rules above; "
        "if you self-reviewed correctly, the critic should pass on the "
        "first try):",
        "",
        "(a) **Direction-sign error**: hypothesis says 'increased / "
        "    converged / positive' but the supporting numbers show the "
        "    opposite. Highest-frequency failure mode — verify carefully.",
        "(b) **Context over-specification**: hypothesis frames the result "
        "    in sub-categories the question did NOT mention (sub-"
        "    disciplines, project codes, narrow cohorts).",
        "(c) **Wrong column reference**: hypothesis cites `_inter` / "
        "    smoothed / derived columns instead of the question's "
        "    conceptual name; or cites a column not in the relevant CSV.",
        "(d) **Variable list bloat**: `variables` list includes columns "
        "    that aren't subjects of the hypothesis claim.",
        "(e) **Numeric claim without supporting computation** in workflow.",
        "(f) **Process narrative instead of an answer**: 'I ran a "
        "    regression that showed...' instead of 'X is higher than Y'.",
        "(g) **Off-by-one in time / unit conversions** (BCE/CE sign, "
        "    calBP arithmetic, decade vs. century).",
        "(h) **Statistical confusion**: correlation read as causation, "
        "    Z-scores confused with raw values, log-scaled values misread.",
        "",
        "Any of these is grounds for revision. Fix yourself before " "declaring done.",
        "",
        "## Available data",
        "",
        "The following CSV files are available under `./{INPUTS_SUBDIR}/` "
        "(symlinks to read-only upstream files — do NOT edit them):".format(
            INPUTS_SUBDIR=INPUTS_SUBDIR
        ),
        "",
        csv_bullets,
        "",
        f"A copy of the upstream metadata is at `./{METADATA_FILENAME}` for "
        "convenient programmatic access.",
        "",
        "## Column metadata",
        "",
        _format_columns_table(task.metadata),
    ]

    if task.split_type == "real":
        if use_domain_knowledge:
            dk = (task.metadata.get("domain_knowledge") or "").strip()
            if dk:
                parts += [
                    "",
                    "## Domain knowledge (provided by the benchmark)",
                    "",
                    dk,
                ]
        if use_workflow_tags:
            wt = task.metadata.get("workflow_tags")
            if wt:
                if isinstance(wt, list):
                    wt_str = ", ".join(str(x) for x in wt)
                else:
                    wt_str = str(wt).strip()
                if wt_str:
                    parts += [
                        "",
                        "## Suggested analytical techniques",
                        "",
                        wt_str,
                    ]

    parts += [
        "",
        "## Question",
        "",
        task.query_text.strip(),
        "",
        "## Output protocol",
        "",
        f"- Save the final answer to `./{RESULT_FILENAME}` (exact name, "
        "workspace root) using `json.dump` with the schema above.",
        "- Do NOT install packages with `pip install`. The default uv "
        "environment already has pandas / numpy / scikit-learn / "
        "statsmodels available.",
        "- Do NOT download anything from the internet — every CSV you need "
        f"is staged under `./{INPUTS_SUBDIR}/`.",
        f"- Do NOT generate plots, figures, PDFs, or other extra deliverables "
        f"— the ONLY scored output is `./{RESULT_FILENAME}`. Use printed "
        "summary statistics (e.g. `df.describe()`, `corr()`, regression "
        "coefficients) instead of saving images.",
        f"- Before finishing, verify `./{RESULT_FILENAME}` parses as JSON, "
        "contains both keys, and that `hypothesis` is non-empty.",
    ]
    return "\n".join(parts)


def _build_data_summary(task: Task, csv_names: list[str]) -> str:
    """Short context the DataAgent sees under 'Data summary'. The DataAgent's
    job here is intentionally narrow — confirm schema + sanity-check the
    files. The actual analytical work (correlations, hypothesis testing,
    forming a hypothesis) belongs to the ExperimentAgent that runs next.
    """
    bullets = "\n".join(
        f"- `./{INPUTS_SUBDIR}/{n}` — see metadata.datasets[] for full column descriptions"
        for n in csv_names
    )
    return (
        f"DiscoveryBench task `{task.uid}` (domain: "
        f"{task.metadata.get('domain', 'unknown')}). Inputs:\n\n"
        f"{bullets}\n\n"
        f"A copy of the upstream metadata JSON is at `./{METADATA_FILENAME}`. "
        "The canonical column schema lives in metadata.datasets[].columns.\n\n"
        "## Scope of THIS phase (light EDA only)\n\n"
        "Do JUST enough to make the experiment phase productive — nothing "
        "more:\n"
        "1. Confirm each CSV loads (right delimiter, encoding, header row).\n"
        "2. Record shape (rows × columns) and column dtypes.\n"
        "3. Spot-check missing values / obvious outliers / unit oddities "
        "(e.g. BCE vs CE sign convention, negative counts).\n"
        "4. Note column-name aliases when one concept appears under "
        "different names across CSVs (will save the experiment agent time).\n\n"
        "Do NOT in this phase:\n"
        "- Compute correlations, regressions, group statistics, or any "
        "  inferential test.\n"
        "- Form, refine, or rank hypotheses.\n"
        "- Draw analytical conclusions about the question — that question "
        "  is the experiment agent's job, not yours.\n"
        "- Write `result.json` — the deliverable file is produced ONLY by "
        "  the experiment phase. Do not touch it from this phase.\n"
        "- Generate plots / figures / PDFs.\n\n"
        "Standard tabular tools (pandas / numpy / scikit-learn / statsmodels) "
        "are preinstalled; do not install anything new."
    )


# --------------------------------------------------------------------------- #
# Per-task driver                                                             #
# --------------------------------------------------------------------------- #


def run_one_task(
    *,
    task: Task,
    output_root: Path,
    max_revisions: int,
    data_recursion_limit: int,
    exp_recursion_limit: int,
    use_domain_knowledge: bool,
    use_workflow_tags: bool,
) -> dict:
    """Run the full SciDER workflow on one DiscoveryBench task. Returns a
    record dict (always) — exceptions are caught so the loop can continue."""
    workspace = (output_root / task.uid).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "uid": task.uid,
        "dataset": task.dataset,
        "metadataid": task.metadataid,
        "qid": task.qid,
        "split_type": task.split_type,
        "split": task.split,
        "workspace": str(workspace),
        "query": task.query_text,
    }

    try:
        csv_names = _stage_inputs(workspace, task.csv_paths)
        _copy_metadata(
            workspace,
            task.metadata,
            use_domain_knowledge=use_domain_knowledge,
            use_workflow_tags=use_workflow_tags,
        )

        user_query = _build_user_query(
            task,
            csv_names,
            use_domain_knowledge=use_domain_knowledge,
            use_workflow_tags=use_workflow_tags,
        )
        data_summary = _build_data_summary(task, csv_names)

        # Persist the prompt — handy for debugging and re-running by hand.
        (workspace / TASK_PROMPT_FILENAME).write_text(user_query, encoding="utf-8")

        logger.info("Running FullWorkflow for uid={} in {}", task.uid, workspace)
        # data_path points at the inputs/ directory so DataAgent sees ALL
        # staged CSVs (matters for the multi-file real tasks). Default
        # WorkspaceInitConfig is fine — uv init writes pyproject.toml so the
        # agent can `uv add` if it ever genuinely needs an extra package.
        result = run_full_workflow(
            data_path=workspace / INPUTS_SUBDIR,
            workspace_path=workspace,
            user_query=user_query,
            data_desc=data_summary,
            max_revisions=max_revisions,
            data_agent_recursion_limit=data_recursion_limit,
            experiment_agent_recursion_limit=exp_recursion_limit,
            user_approval_enabled=False,
        )
        record["workflow_status"] = result.final_status

        result_path = workspace / RESULT_FILENAME
        if not result_path.is_file():
            raise RuntimeError(
                f"Agent finished ({result.final_status}) but {result_path} " f"was not created."
            )
        if not is_valid_result_json(result_path):
            raise RuntimeError(f"{result_path} is malformed or has empty hypothesis.")
        with result_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        record.update(
            ok=True,
            hypothesis=payload["hypothesis"],
            workflow=payload.get("workflow", ""),
            result_path=str(result_path),
        )
        logger.info("uid={} produced result.json", task.uid)
    except Exception as e:
        logger.exception("uid={} failed: {}", task.uid, e)
        record.update(ok=False, error=str(e))
    return record


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _register_models_from_yaml() -> None:
    """Load role assignments from ``model_configs/discoverybench_roles.yaml``."""
    if not ROLES_YAML_PATH.exists():
        raise FileNotFoundError(
            f"Role yaml missing at {ROLES_YAML_PATH}. "
            "This file pins the data/experiment/critic/approval/... roles "
            "for DiscoveryBench; don't rename it."
        )
    logger.info("Registering roles from {}", ROLES_YAML_PATH)
    registered = register_defaults_from_yaml(ROLES_YAML_PATH)
    logger.info("Registered {} roles: {}", len(registered), sorted(registered))


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DiscoveryBench workflow — per-(dataset, metadata, qid) "
            "FullWorkflow run that produces ./result.json with a "
            "natural-language hypothesis + workflow."
        ),
        prog="python -m bench_workflows.discoverybench_workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bench-root",
        required=True,
        help="Path to the DiscoveryBench data dir (the inner one with "
        "real/ and synth/ subdirs), e.g. "
        "benchmarks/discoverybench/discoverybench/discoverybench.",
    )
    parser.add_argument(
        "--split-type",
        choices=["synth", "real"],
        default="real",
        help="Which task family to run (default: real).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which split under split-type (default: test).",
    )
    parser.add_argument(
        "--output-root",
        "-o",
        required=True,
        help="Directory under which per-uid workspaces are created "
        "(<output_root>/<uid>/result.json).",
    )
    parser.add_argument(
        "--use-domain-knowledge",
        action="store_true",
        help="(real-only) Inject metadata.domain_knowledge into the prompt. "
        "Off by default to match the canonical 'no-hint' upstream setting.",
    )
    parser.add_argument(
        "--use-workflow-tags",
        action="store_true",
        help="(real-only) Inject metadata.workflow_tags as a hint about "
        "which analytical techniques to consider. Off by default.",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=1,
        help="Critic/approval retry budget (subagent gate may stop earlier). " "Default 1.",
    )
    parser.add_argument(
        "--data-recursion-limit",
        type=int,
        default=80,
        help="Recursion limit for the data agent (default 80).",
    )
    parser.add_argument(
        "--exp-recursion-limit",
        type=int,
        default=100,
        help="Recursion limit for the experiment agent (default 100).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip uids whose <output_root>/<uid>/result.json is already a "
        "valid deliverable. Lets you resume long runs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most the first N tasks (after --skip-existing). " "Omit to run all.",
    )
    parser.add_argument(
        "--uids",
        default=None,
        help="Comma-separated list of uids to run; overrides --limit. "
        "Useful for targeted debug runs.",
    )

    args = parser.parse_args()

    # 1. Register models first so config errors surface before any file I/O.
    _register_models_from_yaml()

    bench_root = Path(args.bench_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 2. Discover tasks.
    tasks = list(discover_tasks(bench_root, args.split_type, args.split))
    logger.info(
        "Discovered {} tasks under {}/{}/{}",
        len(tasks),
        bench_root,
        args.split_type,
        args.split,
    )

    # 3. --skip-existing filtering.
    if args.skip_existing:
        completed = scan_completed_uids(output_root)
        if completed:
            logger.info(
                "--skip-existing: {} uids already have a valid result.json",
                len(completed),
            )
            tasks = [t for t in tasks if t.uid not in completed]

    # 4. --uids takes precedence over --limit.
    if args.uids:
        wanted = {u.strip() for u in args.uids.split(",") if u.strip()}
        selected = [t for t in tasks if t.uid in wanted]
        missing = wanted - {t.uid for t in selected}
        if missing:
            logger.warning("Requested uids not in candidate set: {}", sorted(missing))
    elif args.limit is not None:
        selected = tasks[: args.limit]
    else:
        selected = tasks
    logger.info("Processing {} tasks", len(selected))

    # 5. Resume results.json across runs.
    results_path = output_root / "results.json"
    results: list[dict] = []
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        logger.info("Resuming from {} ({} prior records)", results_path, len(results))

    for t in selected:
        record = run_one_task(
            task=t,
            output_root=output_root,
            max_revisions=args.max_revisions,
            data_recursion_limit=args.data_recursion_limit,
            exp_recursion_limit=args.exp_recursion_limit,
            use_domain_knowledge=args.use_domain_knowledge,
            use_workflow_tags=args.use_workflow_tags,
        )
        # Dedup by uid on successive runs.
        results = [r for r in results if r.get("uid") != record["uid"]]
        results.append(record)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.get("ok"))
    logger.info(
        "Done. {}/{} produced a valid result.json. Results at {}", n_ok, len(results), results_path
    )


if __name__ == "__main__":
    _main()
