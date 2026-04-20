"""
Workflow for AstroVisBench — hand each query to SciDER's experiment agent and
have it produce the final visualization PNG directly.

Strategy (vs. upstream)
-----------------------
Upstream AstroVisBench wants you to (a) generate `processing_gen_code`,
(b) generate `visualization_gen_code`, (c) run both through a notebook
executor + variable-inspection eval, (d) LLM-as-judge the visualization.
Stages (c) and (d) are eval — we skip them entirely. We also skip (a)
because we only want the *final image*.

What SciDER does instead:
- Queries are pulled straight from the HuggingFace dataset
  ``sebajoe/AstroVisBench`` (the upstream-provided ``astrovisbench_queries.json``).
- Ground-truth processing is already executed upstream; the bridge variables
  (vars assigned in processing_gt_code and used in visualization_gt_code) are
  pickled under ``<cache_root>/<uid>/<varname>.pkl``. We expose those pickles
  to the agent at ``<workspace>/<uid>/inputs/<varname>.pkl`` (symlinked).
- Each query gets its OWN per-uid workspace at ``<output_root>/<uid>/``.
- The experiment agent runs inside the astrovisbench Python 3.10 venv
  (prepended to PATH via ``WorkspaceInitConfig``) and has ``astropy``,
  ``matplotlib``, ``numpy`` etc. already installed — the agent must NOT
  ``uv add`` / ``pip install``.
- The agent's deliverable is a single file at
  ``<workspace>/<uid>/generated_image.png``.

Model roles are loaded from ``bench_workflows/model_configs/astrovisbench_roles.yaml``
(no ``--models`` knob — edit the yaml to tune).

Early stopping: ``max_revisions=1``. The approval subagent gate
(``SubagentApprovalHandler``) decides after the first pass whether the image
is good enough — if yes, we skip the revision round.

Usage
-----
    python -m bench_workflows.astrovisbench_workflow \\
        --cache-root     /home/klin/data/SciDER/AstroVisBench/gt_processing_cache \\
        --output-root    benchmarks/astrovisbench/workspace \\
        --astrovis-venv  benchmarks/astrovisbench/.venv \\
        --bench-env-root /home/klin/data/SciDER/AstroVisBench/bench_environment \\
        --skip-existing
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

# Make scider + bench modules importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scider.core.code_env import WorkspaceInitConfig
from scider.default.models import register_defaults_from_yaml
from scider.workflows.experiment_workflow import run_experiment_workflow

# ---- Constants ----
OUTPUT_IMAGE_FILENAME = "generated_image.png"
INPUTS_SUBDIR = "inputs"
# The per-project model-role assignment for this benchmark. Edit the yaml to
# swap models; do NOT add a --models CLI knob.
ROLES_YAML_PATH = Path(__file__).parent / "model_configs" / "astrovisbench_roles.yaml"
# HuggingFace dataset & file name — upstream publishes queries as one JSON.
HF_REPO_ID = "sebajoe/AstroVisBench"
HF_QUERIES_FILENAME = "astrovisbench_queries.json"
# PNG magic bytes — used by `is_valid_png` for --skip-existing.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# Queries (HuggingFace) + existing-result scan                                #
# --------------------------------------------------------------------------- #


def load_queries_from_hf() -> list[dict]:
    """Download ``astrovisbench_queries.json`` from HuggingFace and return it.

    Uses the default HF cache (``HF_HOME`` / ``~/.cache/huggingface``) so
    subsequent runs are offline-fast.
    """
    from huggingface_hub import hf_hub_download

    logger.info("Downloading queries from HF: {}/{}", HF_REPO_ID, HF_QUERIES_FILENAME)
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_QUERIES_FILENAME,
        repo_type="dataset",
    )
    queries = json.loads(Path(local_path).read_text(encoding="utf-8"))
    logger.info("Loaded {} queries (cached at {})", len(queries), local_path)
    return queries


def is_valid_png(path: Path) -> bool:
    """A file counts as a valid PNG output if it exists, is non-empty, and
    starts with the PNG magic bytes. We don't decode the image — that would
    require Pillow and the astrovis venv; this cheap check is enough to
    distinguish a real output from a stub / truncated file."""
    try:
        if not path.is_file() or path.stat().st_size < len(_PNG_MAGIC):
            return False
        with path.open("rb") as f:
            return f.read(len(_PNG_MAGIC)) == _PNG_MAGIC
    except OSError:
        return False


def scan_completed_uids(output_root: Path) -> set[str]:
    """Return uids whose ``<output_root>/<uid>/generated_image.png`` is a
    valid PNG. Used by ``--skip-existing``."""
    if not output_root.is_dir():
        return set()
    completed: set[str] = set()
    for uid_dir in output_root.iterdir():
        if not uid_dir.is_dir():
            continue
        if is_valid_png(uid_dir / OUTPUT_IMAGE_FILENAME):
            completed.add(uid_dir.name)
    return completed


# --------------------------------------------------------------------------- #
# Pickle cache resolution                                                     #
# --------------------------------------------------------------------------- #


def _resolve_cache_pickles(cache_root: Path, uid: str) -> list[Path]:
    """Return the list of pickle files for this uid, excluding the executed
    notebook and macOS ``._*`` metadata files."""
    uid_dir = cache_root / uid
    if not uid_dir.is_dir():
        raise FileNotFoundError(f"No cache dir for uid {uid} at {uid_dir}")
    pkls = sorted(
        p
        for p in uid_dir.iterdir()
        if p.is_file() and p.suffix == ".pkl" and not p.name.startswith("._")
    )
    if not pkls:
        raise FileNotFoundError(f"No .pkl bridge variables under {uid_dir}")
    return pkls


def _stage_inputs(workspace: Path, pickles: list[Path]) -> list[str]:
    """Symlink each pickle into ``<workspace>/inputs/``. Returns the list of
    variable names (file stems) for use in the agent prompt."""
    inputs_dir = workspace / INPUTS_SUBDIR
    inputs_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for src in pickles:
        dst = inputs_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        names.append(src.stem)
    return names


# Subprocess probe — tries `pickle.load` on each argv path, prints
# {filename: "ErrType: msg"} for failures (empty dict if all OK). Runs in the
# astrovis venv so the right `lightkurve` / `astropy` / etc. classes are
# importable; SciDER's own venv doesn't have those.
#
# Pointer-valued pickles (filename strings, PosixPaths) are NOT rejected
# here — they're legitimately used by upstream queries, and the prompt now
# points the agent at ``bench_env_root`` so it can resolve the referenced
# files itself.
_VALIDATE_PICKLE_PROBE = (
    "import pickle, sys, json\n"
    "broken = {}\n"
    "for p in sys.argv[1:]:\n"
    "    try:\n"
    "        with open(p, 'rb') as f:\n"
    "            pickle.load(f)\n"
    "    except Exception as e:\n"
    "        msg = f'{type(e).__name__}: {e}'\n"
    "        broken[p] = msg[:200]\n"
    "sys.stdout.write(json.dumps(broken))\n"
)


def validate_pickles(pickles: list[Path], python_exec: Path) -> dict[str, str]:
    """Return a mapping of ``filename → "ErrType: msg"`` for pickles that fail
    to load. Empty dict → all OK.

    Runs a short subprocess probe with ``python_exec`` so upstream-specific
    classes (``lightkurve.LightCurveCollection`` etc.) resolve properly.
    SciDER's main venv doesn't have those, so we must use the astrovis venv.
    """
    try:
        r = subprocess.run(
            [str(python_exec), "-c", _VALIDATE_PICKLE_PROBE, *[str(p) for p in pickles]],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {p.name: "timeout (>120s) during pickle.load" for p in pickles}
    except (FileNotFoundError, OSError) as e:
        # python_exec missing / not executable — fail conservatively.
        return {p.name: f"probe interpreter unavailable: {e}" for p in pickles}

    if r.returncode != 0:
        err_tail = (r.stderr or r.stdout or "")[-300:]
        return {p.name: f"probe crashed (rc={r.returncode}): {err_tail}" for p in pickles}

    try:
        raw = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return {p.name: f"probe returned non-json: {r.stdout[:200]}" for p in pickles}

    # Rewrite keys from full paths (what the probe saw) to bare filenames
    # (what's useful in a log/error record).
    return {Path(k).name: v for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# Prompt building                                                             #
# --------------------------------------------------------------------------- #


def _build_user_query(
    query: dict,
    var_names: list[str],
    bench_env_root: Path | None = None,
) -> str:
    """Assemble the experiment-agent task prompt.

    When ``bench_env_root`` is provided, an extra section points the agent
    at ``<bench_env_root>/<nb_stem>/`` where the notebook's working-directory
    state lives (FITS files etc. referenced by pickle filenames). Without
    that hint, the agent would wander the filesystem hunting for those
    files — see the astrovis log for the real case that motivated this.
    """
    pkl_list = "\n".join(f"  - `./{INPUTS_SUBDIR}/{name}.pkl` → `{name}`" for name in var_names)
    proc_underspec = (query.get("processing_underspecifications") or "").strip()
    vis_underspec = (query.get("visualization_underspecifications") or "").strip()

    parts: list[str] = [
        "# AstroVisBench Visualization Task",
        "",
        f"Your deliverable is a single PNG file at `./{OUTPUT_IMAGE_FILENAME}` "
        "in this workspace.",
        "",
        "## Available inputs (already computed, pickled)",
        "",
        "The following variables — produced by the processing stage — are "
        f"available as pickle files under `./{INPUTS_SUBDIR}/`. Load them with "
        "``pickle.load(open(path, 'rb'))``:",
        "",
        pkl_list,
        "",
        "Do NOT re-run the processing code and do NOT install packages. All "
        "necessary Python deps (astropy, matplotlib, numpy, ...) are already "
        "available on PATH.",
    ]

    if bench_env_root is not None:
        nb_stem = Path(query.get("nb_path") or "").stem or "<unknown>"
        parts += [
            "",
            "## Additional resource — notebook bench environment",
            "",
            "Some pickles are filename strings or `Path` objects pointing at "
            "files that were produced at notebook-execution time (FITS files, "
            "downloaded data, intermediate outputs) and not themselves "
            "picklable. The notebook's working-directory state for THIS query "
            "is available locally at:",
            "",
            f"    `{bench_env_root}/{nb_stem}/`",
            "",
            "Inside that directory you will typically find:",
            "",
            f"  - `{nb_stem}_comped.tar.gz` — compressed snapshot of the "
            "working dir used when processing was run. If a pickle holds a "
            "relative filename/path, the referenced file is almost certainly "
            "inside this tarball. Extract it to a temp dir (e.g. "
            "`tarfile.open(...).extractall(tmp_dir)`) and read what you need.",
            f"  - `{nb_stem}.ipynb` — the original notebook (reference only; " "don't rerun it).",
            "",
            "Use this path ONLY when the pickles alone aren't enough. For "
            "queries whose pickles already contain the plotting data (arrays, "
            "tables, dataframes), there's no need to touch bench_env.",
            "",
            "Do NOT download anything from the internet — everything you need "
            "is either in `./inputs/` or under that bench_env path.",
        ]

    parts += [
        "",
        "## Notebook context",
        "",
        "### Setup description",
        query["setup_query"].strip(),
        "",
        "### Setup code (imports + matplotlib config — re-apply in your script)",
        "```python",
        query["setup_gt_code"].strip(),
        "```",
        "",
        "### Processing description (context only — do NOT re-run)",
        query["processing_query"].strip(),
    ]
    if proc_underspec:
        parts += ["", "#### Processing underspecifications (clarifications)", proc_underspec]
    parts += [
        "",
        "### Processing code (shows how the pickled vars were produced — "
        "context only, do NOT re-run)",
        "```python",
        query["processing_gt_code"].strip(),
        "```",
        "",
        "## Your task — produce the visualization",
        "",
        query["visualization_query"].strip(),
    ]
    if vis_underspec:
        parts += ["", "### Visualization underspecifications (clarifications)", vis_underspec]
    parts += [
        "",
        "## Output protocol",
        "",
        f"- Save the figure as `./{OUTPUT_IMAGE_FILENAME}` (exact name, workspace root).",
        "- Use the matplotlib Agg backend: `matplotlib.use('Agg')`.",
        f"- Call `plt.savefig('{OUTPUT_IMAGE_FILENAME}', dpi=150, bbox_inches='tight')`.",
        "- Do NOT call `plt.show()`.",
        f"- Verify the produced file is a valid non-empty PNG (e.g. `Read {OUTPUT_IMAGE_FILENAME}` "
        "returns an image) before finishing.",
    ]
    return "\n".join(parts)


def _build_data_summary(var_names: list[str]) -> str:
    """Short context string shown by the experiment agent under 'Data summary'."""
    bullets = "\n".join(f"- `{n}` — loaded from `./{INPUTS_SUBDIR}/{n}.pkl`" for n in var_names)
    return (
        "Pre-computed processing outputs (astropy / numpy objects) are "
        f"available as pickle files under `./{INPUTS_SUBDIR}/`:\n\n"
        f"{bullets}\n\n"
        "Python deps are preinstalled in the venv on PATH; do not install anything."
    )


# --------------------------------------------------------------------------- #
# Per-query driver                                                            #
# --------------------------------------------------------------------------- #


def run_one_query(
    *,
    query: dict,
    cache_root: Path,
    output_root: Path,
    astrovis_venv: Path | None,
    max_revisions: int,
    recursion_limit: int,
    bench_env_root: Path | None = None,
) -> dict:
    """Run experiment_agent on one AstroVisBench query. Returns a record dict.

    The record has ``ok``, ``uid``, ``workspace``, plus ``image_path`` on
    success or ``error`` on failure. Exceptions are caught so the caller
    can continue through the rest of the queries.
    """
    uid = query["uid"]
    workspace = (output_root / uid).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    image_path = workspace / OUTPUT_IMAGE_FILENAME

    record: dict = {
        "uid": uid,
        "nb_path": query.get("nb_path"),
        "workspace": str(workspace),
    }

    try:
        pickles = _resolve_cache_pickles(cache_root, uid)

        # Pre-flight: the upstream cache occasionally contains truncated /
        # unloadable pickles (e.g. lightkurve.LightCurveCollection objects
        # whose pickling silently failed). Feeding these to the agent wastes
        # 3+ minutes of LLM calls per broken uid as it spirals into
        # byte-level pickle disassembly — short-circuit here instead.
        if astrovis_venv is not None:
            venv_python = astrovis_venv / "bin" / "python"
            if venv_python.exists():
                broken = validate_pickles(pickles, venv_python)
                if broken:
                    raise RuntimeError(f"Broken pickles in cache for uid={uid}: {broken}")

        var_names = _stage_inputs(workspace, pickles)
        user_query = _build_user_query(query, var_names, bench_env_root=bench_env_root)
        data_summary = _build_data_summary(var_names)

        # Persist the prompt so the run is reproducible / debuggable.
        (workspace / "task.md").write_text(user_query, encoding="utf-8")

        init_config = WorkspaceInitConfig(
            env_manager="python",
            init_uv=False,
            venv_path=astrovis_venv.resolve() if astrovis_venv else None,
        )

        logger.info("Running experiment agent for uid={} in {}", uid, workspace)
        run_experiment_workflow(
            workspace_path=workspace,
            user_query=user_query,
            data_summary=data_summary,
            max_revisions=max_revisions,
            recursion_limit=recursion_limit,
            user_approval_enabled=False,  # force the subagent gate
            workspace_init_config=init_config,
        )

        if not image_path.exists():
            raise RuntimeError(
                f"Agent finished but {image_path} was not created. "
                f"Check {workspace / 'experiment_agent_history.json'} for context."
            )
        if image_path.stat().st_size == 0:
            raise RuntimeError(f"{image_path} is empty (0 bytes).")

        record.update(ok=True, image_path=str(image_path))
        logger.info("uid={} produced {}", uid, image_path)
    except Exception as e:
        logger.exception("uid={} failed: {}", uid, e)
        record.update(ok=False, error=str(e))
    return record


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _register_models_from_yaml() -> None:
    """Load role assignments from ``model_configs/astrovisbench_roles.yaml``."""
    if not ROLES_YAML_PATH.exists():
        raise FileNotFoundError(
            f"Role yaml missing at {ROLES_YAML_PATH}. "
            "This file pins the experiment/critic/approval/... roles for "
            "AstroVisBench; don't rename it."
        )
    logger.info("Registering roles from {}", ROLES_YAML_PATH)
    registered = register_defaults_from_yaml(ROLES_YAML_PATH)
    logger.info("Registered {} roles: {}", len(registered), sorted(registered))


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AstroVisBench workflow — per-query experiment-agent run that "
            "produces `generated_image.png` using pre-pickled processing outputs."
        ),
        prog="python -m bench_workflows.astrovisbench_workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-root",
        "-c",
        required=True,
        help="Path to gt_processing_cache/ (with <uid>/*.pkl subdirs)",
    )
    parser.add_argument(
        "--output-root",
        "-o",
        required=True,
        help="Directory under which per-uid workspaces are created "
        "(<output_root>/<uid>/generated_image.png).",
    )
    parser.add_argument(
        "--bench-env-root",
        default=None,
        help="Path to the extracted upstream bench_environment/ directory "
        "(i.e. the contents of bench_env.tar.gz). When set, the agent is "
        "told it can find per-notebook working-dir state under "
        "<bench_env_root>/<nb_stem>/ — used for queries whose pickles hold "
        "filenames/paths instead of actual data (FITS files etc.). Leave "
        "unset if you only have gt_processing_cache/.",
    )
    parser.add_argument(
        "--astrovis-venv",
        default=None,
        help="Path to the astrovisbench Python 3.10 venv (e.g. "
        "benchmarks/astrovisbench/.venv). If set, its bin/ is prepended to "
        "PATH for every agent subprocess; if omitted, the agent uses whatever "
        "python it finds on PATH.",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=1,
        help="Upper bound on critic/approval retries (the subagent gate may "
        "early-stop before this). Default 1 — usually one revision is plenty.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        help="Recursion limit for the experiment-agent graph (default 100).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip any uid whose <output_root>/<uid>/generated_image.png is "
        "already a valid PNG (magic-byte check). Lets you resume long runs "
        "without re-doing completed queries.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Dev/smoke-test knob: process at most the first N queries from "
        "the HF dataset (after --skip-existing filtering). Omit to run them all.",
    )
    parser.add_argument(
        "--uids",
        default=None,
        help="Comma-separated list of uids to run (after --skip-existing "
        "filtering). Overrides --limit.",
    )

    args = parser.parse_args()

    # 1. Register models first so any failure surfaces before we start work.
    _register_models_from_yaml()

    # 2. Fetch queries from HF.
    queries = load_queries_from_hf()

    cache_root = Path(args.cache_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    astrovis_venv = Path(args.astrovis_venv).resolve() if args.astrovis_venv else None
    if astrovis_venv is not None and not (astrovis_venv / "bin" / "python").exists():
        logger.warning(
            "astrovis_venv={} has no bin/python — PATH injection will still "
            "happen but the agent may not find the expected interpreter.",
            astrovis_venv,
        )
    bench_env_root = Path(args.bench_env_root).resolve() if args.bench_env_root else None
    if bench_env_root is not None and not bench_env_root.is_dir():
        logger.warning(
            "bench_env_root={} is not a directory — agent will be told about a "
            "path that doesn't exist. Fix the flag or leave it unset.",
            bench_env_root,
        )

    # 3. --skip-existing: filter out uids that already have a valid PNG on disk.
    if args.skip_existing:
        completed = scan_completed_uids(output_root)
        if completed:
            logger.info(
                "--skip-existing: found {} uids with a valid generated_image.png; skipping them",
                len(completed),
            )
            queries = [q for q in queries if q["uid"] not in completed]

    # 4. --uids takes precedence over --limit (targeted debug runs).
    if args.uids:
        wanted = {u.strip() for u in args.uids.split(",") if u.strip()}
        selected = [q for q in queries if q["uid"] in wanted]
        missing = wanted - {q["uid"] for q in selected}
        if missing:
            logger.warning("Requested uids not in candidate set: {}", sorted(missing))
    elif args.limit is not None:
        selected = queries[: args.limit]
    else:
        selected = queries
    logger.info("Processing {} queries", len(selected))

    # 5. Load prior results.json (if any) so we accumulate records across runs.
    results_path = output_root / "results.json"
    results: list[dict] = []
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        logger.info("Resuming from {} ({} prior records)", results_path, len(results))

    for q in selected:
        record = run_one_query(
            query=q,
            cache_root=cache_root,
            output_root=output_root,
            astrovis_venv=astrovis_venv,
            max_revisions=args.max_revisions,
            recursion_limit=args.recursion_limit,
            bench_env_root=bench_env_root,
        )
        # Dedup by uid on successive runs.
        results = [r for r in results if r.get("uid") != record["uid"]]
        results.append(record)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.get("ok"))
    logger.info("Done. {}/{} produced an image. Results at {}", n_ok, len(results), results_path)


if __name__ == "__main__":
    _main()
