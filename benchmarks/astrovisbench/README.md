# Benchmark for AstroVisBench

[AstroVisBench](https://github.com/UT-NLP/AstroVisBench) is a code benchmark for
scientific computing and visualization in astronomy (NeurIPS 2025 Datasets & Benchmarks).
See the upstream README under `AstroVisBench/` for the full schema and design.

We assume all commands are run from the same dir as this README file.

> **Heads up**: Upstream pins very old scientific stack (`numpy==1.24.3`,
> `tensorflow==2.13.0`, `astropy==6.1.7`, ...) that only work under **Python 3.10**.
> That is why this benchmark's uv env is different from `mlebench`/`scicodebench`.
> You also need ~100 GB of free disk space for the bench environment.

## 1. Environment setup

First, create a new uv env:

```bash
uv init --python 3.10
uv venv --seed
source .venv/bin/activate

pip install -r ../../requirements.txt
```

Then, install the upstream dependencies by running the upstream script
(it pins incompatible metadata so every line goes in with `--no-deps`):

```bash
pushd AstroVisBench/reqs_resolve; bash install_packages.sh; popd
```

Notes:
- `requirements.txt` already has the macOS-only `pyobjc-*` lines commented
  out, so Linux installs cleanly.
- If you are on macOS, use `requirements_macos.txt` instead.
- `mpi4py` is **optional** — only needed for `exec_bench.py --MPI`
  (multi-process run). Install it separately if you want that path:
  ```bash
  # Debian/Ubuntu: `apt install libopenmpi-dev openmpi-bin` first.
  # On the Spark server MPI is already available.
  pip install mpi4py
  ```
  Without it, `exec_bench.py` still works in single-process mode.

## 2. Ground-truth cache (only thing SciDER needs)

SciDER's path only needs the **ground-truth processing cache** — we skip
re-running processing code and load the pickled bridge variables directly.
That means you do **not** need upstream's heavy `bench_env.tar.gz`.

Queries are pulled from the HuggingFace dataset
[`sebajoe/AstroVisBench`](https://huggingface.co/datasets/sebajoe/AstroVisBench)
at runtime (cached under `$HF_HOME`); there is **no `--queries` flag**.

Download `gt_processing_cache.tar.gz` from the upstream
[Box folder](https://utexas.box.com/s/2evj5cs3u2gqndvgc9sd66cmlggl9fg1)
and extract it:

```bash
mkdir -p data
# put gt_processing_cache.tar.gz in data/, then extract
tar xzf data/gt_processing_cache.tar.gz -C data/
```

After extraction you should see one `<uid>/` subdir per query, each holding
pickled bridge variables (`*.pkl`).

> On the Spark server the shared extract lives at
> `/home/klin/data/SciDER/AstroVisBench/gt_processing_cache` — use that
> instead of re-downloading.

> If you also want the upstream full-eval path (variable-inspection +
> LLM-as-judge), jump to §4 — that one does need `bench_env.tar.gz`.

## 3. Run SciDER on the benchmark

Models are pinned in
[`bench_workflows/model_configs/astrovisbench_roles.yaml`](../../bench_workflows/model_configs/astrovisbench_roles.yaml).
Edit the YAML to swap models — there is **no `--models` CLI flag**.

All commands below run from **the repo root**, using SciDER's own venv
(not the Python-3.10 benchmark venv — that one is only invoked by the
experiment agent via `--astrovis-venv` to execute visualization code):

```bash
# from SciEvo/ repo root
uv run python -m bench_workflows.astrovisbench_workflow \
  --cache-root    benchmarks/astrovisbench/data/gt_processing_cache \
  --output-root   benchmarks/astrovisbench/workspace \
  --astrovis-venv benchmarks/astrovisbench/.venv \
  --skip-existing
```

Behaviour:
- Per-uid workspace at `<output-root>/<uid>/`, containing
  `inputs/*.pkl` (symlinks to cache), `task.md` (the full prompt),
  `generated_image.png` (deliverable), `experiment_agent_history.json` +
  `subagents/` (full conversation trace for debugging).
- A top-level `<output-root>/results.json` records `{uid, ok, image_path}`
  per query. Re-running with `--skip-existing` picks up where you left off.

Useful flags:

| flag | purpose |
|---|---|
| `--skip-existing` | skip any uid whose `generated_image.png` is already a valid PNG (magic-byte check) — safe for resume after interruption |
| `--limit N` | dev/smoke-test: only process the first N queries (after `--skip-existing` filtering) |
| `--uids a,b,c` | run only these uids — overrides `--limit` |
| `--max-revisions N` | upper bound on critic/approval-subagent retries. Default `1`: the approval subagent early-stops after the first pass when the image looks good enough; `1` is the hard cap otherwise |
| `--recursion-limit N` | LangGraph recursion limit for the experiment agent (default 100) |

Typical workflows:

```bash
# Smoke test on one query:
uv run python -m bench_workflows.astrovisbench_workflow \
  --cache-root benchmarks/astrovisbench/data/gt_processing_cache \
  --output-root benchmarks/astrovisbench/workspace \
  --astrovis-venv benchmarks/astrovisbench/.venv \
  --limit 1

# Full 432-query run with resume:
uv run python -m bench_workflows.astrovisbench_workflow \
  --cache-root benchmarks/astrovisbench/data/gt_processing_cache \
  --output-root benchmarks/astrovisbench/workspace \
  --astrovis-venv benchmarks/astrovisbench/.venv \
  --skip-existing

# Debug a specific query:
uv run python -m bench_workflows.astrovisbench_workflow \
  --cache-root benchmarks/astrovisbench/data/gt_processing_cache \
  --output-root benchmarks/astrovisbench/workspace \
  --astrovis-venv benchmarks/astrovisbench/.venv \
  --uids c1b95253-3b6a-4a04-b6ec-a5ab4dc1c4bd
```

### Spark-server one-liner

On `ai4scientist-spark` the cache lives outside the repo. Typical invocation
after `rsync`ing the code:

```bash
ssh klin@ai4scientist-spark
cd ~/rsync/SciEvo
set -a && source .env && set +a
.venv/bin/python -m bench_workflows.astrovisbench_workflow \
  --cache-root    /home/klin/data/SciDER/AstroVisBench/gt_processing_cache \
  --output-root   ~/rsync/SciEvo/benchmarks/astrovisbench/workspace \
  --astrovis-venv ~/rsync/SciEvo/benchmarks/astrovisbench/.venv \
  --skip-existing
```

## 4. Optional: upstream full-eval path

This is the full upstream pipeline (generate processing_gen_code +
visualization_gen_code → `exec_bench.py` variable-inspection → LLM-as-judge
→ `aggregate_results.py`). **Not needed if you only want the images** — the
SciDER workflow in §3 produces the PNG directly. Pulled in here mostly for
reproducing upstream numbers; it also requires `bench_env.tar.gz` on disk.

Fill `processing_gen_code` + `visualization_gen_code` using upstream's
single-shot generator:

```bash
pushd AstroVisBench/generate_code
python generate_code.py \
  ../../data/astrovisbench_queries.json \
  ../../data/astrovisbench_filled.json \
  gemini-2.5-pro GEM
popd
```

Run upstream's variable-inspection eval (hours):

```bash
pushd AstroVisBench
python exec_bench.py \
  ../data/astrovisbench_filled.json \
  ../data/bench_env \
  --true-cache ../data/gt_processing_cache \
  --gen-cache ../data/gen_cache \
  --vis-cache ../data/vis_cache \
  --outfile ../data/astrovisbench_executed.json \
  --temp-caching \
  --min-diff-only
popd
```

Add `--MPI` + `split_queries.py` to parallelise. See upstream README for
other flags (`--skip-test`, `--run-all`, ...).

LLM-as-judge + aggregation:

```bash
pushd AstroVisBench/vis_evaluation
python vis_evaluation.py \
  ../../data/astrovisbench_executed.json \
  ../../data/astrovisbench_viseval.json
popd

pushd AstroVisBench
python aggregate_results.py ../data/astrovisbench_viseval.json
popd
```

## Layout

```
benchmarks/astrovisbench/
├── AstroVisBench/     # upstream submodule (CC BY-SA 4.0)
├── data/              # gt_processing_cache, optional bench_env (gitignored)
├── workspace/         # per-uid runs: <uid>/generated_image.png + results.json (gitignored)
├── main.py            # uv-init stub (gitignored)
├── pyproject.toml     # Python 3.10 project (gitignored)
├── .python-version    # 3.10 (gitignored)
├── .venv/             # Python 3.10 venv with astropy/matplotlib/... (gitignored)
└── README.md          # this file
```
