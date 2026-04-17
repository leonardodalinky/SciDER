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

## 2. Dataset & bench environment

The benchmark ships as a HuggingFace dataset
([sebajoe/AstroVisBench](https://huggingface.co/datasets/sebajoe/AstroVisBench)).
Upstream recommends converting it into the single JSON file
[`astrovisbench_queries.json`](https://utexas.box.com/s/2evj5cs3u2gqndvgc9sd66cmlggl9fg1)
from their Box folder. Download it plus the bench environment archive and
(optionally) the ground-truth cache:

```bash
# files live under https://utexas.box.com/s/2evj5cs3u2gqndvgc9sd66cmlggl9fg1
#   astrovisbench_queries.json   — benchmark queries
#   bench_env.tar.gz             — required file states (~tens of GB)
#   gt_processing_cache.tar.gz   — optional, speeds up first run

mkdir -p data
# put downloads in data/, then extract
tar xzf data/bench_env.tar.gz -C data/
tar xzf data/gt_processing_cache.tar.gz -C data/   # optional
```

## 3. Generate LLM code with SciDER

AstroVisBench evaluates `processing_gen_code` and `visualization_gen_code`
inside each query. SciDER wires the generation via
`bench_workflows/astrovisbench_workflow.py` (to be added alongside the
existing `mlebench_workflow.py` / `scicodebench_workflow.py`). From the repo
root:

```bash
export SCIDER_DIR=$(git rev-parse --show-toplevel)

# Fill processing_gen_code + visualization_gen_code for every query.
python -m bench_workflows.astrovisbench_workflow \
  --queries benchmarks/astrovisbench/data/astrovisbench_queries.json \
  --outfile benchmarks/astrovisbench/data/astrovisbench_filled.json \
  --models gemini-medium-high
```

If you only want to benchmark the upstream single-shot prompt (no SciDER
agents), the original generator under `AstroVisBench/generate_code/` still
works:

```bash
pushd AstroVisBench/generate_code
python generate_code.py \
  ../../data/astrovisbench_queries.json \
  ../../data/astrovisbench_filled.json \
  gemini-2.5-pro GEM
popd
```

## 4. Execute & evaluate

Processing + caches (heavy, can take hours):

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

Add `--MPI` and split the input JSON via `split_queries.py` to parallelise.
See upstream README for the full flag list (`--skip-test`, `--run-all`, ...).

Visualization evaluation (LLM-as-judge, 3 trials per query):

```bash
pushd AstroVisBench/vis_evaluation
python vis_evaluation.py \
  ../../data/astrovisbench_executed.json \
  ../../data/astrovisbench_viseval.json
popd
```

Aggregate the final scores:

```bash
pushd AstroVisBench
python aggregate_results.py ../data/astrovisbench_viseval.json
popd
```

## Layout

```
benchmarks/astrovisbench/
├── AstroVisBench/     # upstream submodule (CC BY-SA 4.0)
├── data/              # queries JSON, bench env, caches (gitignored)
├── main.py            # uv-init stub (gitignored)
├── pyproject.toml     # Python 3.10 project (gitignored)
├── .python-version    # 3.10 (gitignored)
└── README.md          # this file
```
