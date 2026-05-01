# Benchmark for AIRS-Bench

[AIRS-Bench](https://github.com/facebookresearch/airs-bench) is a 20-task ML
research suite covering NLP, code, math, biochemical modelling, time-series
forecasting, and graph regression. Each task is a `<problem, dataset, metric>`
triplet with a known SOTA target.

We assume all commands are run from the **repo root** (`SciEvo/`) unless noted.

## 1. Download the raw datasets (one-time)

```bash
pushd benchmarks/airsbench/airs-bench
bash datasets/download_hf_datasets.sh /path/to/datasets_download_location
popd
```

That dir becomes `--data-root` below — point it at any path you have write
access to (on the Spark server we use
`/home/klin/data/SciDER/airs-bench/datasets_download_location`).

The script reads `airs-bench/datasets/hf_datasets.csv` and pulls each task's
HF dataset to disk via `datasets.load_dataset(...).save_to_disk(...)`. Each
task's `prepare.py` later reads these from the same path.

## 2. Run the workflow

The SciDER workflow (`bench_workflows/airsbench_workflow.py`) does:

1. Create a per-task workspace at `<output_root>/<TaskName>/` and `uv init` it.
2. Copy `project_description.md`, `metadata.yaml`, `prepare.py` into the
   workspace (NOT `evaluate.py` / `evaluate_prepare.py` — those are withheld
   from the agent).
3. Run `prepare.py` via `uv run --with <container_python_requirements>`.
4. Hand the task to `run_experiment_workflow` (data + experiment, no
   ideation, no paper writing). The agent's deliverable is
   `<workspace>/submission.csv`.
5. Stage `evaluate.py` / `evaluate_prepare.py` into `<workspace>/_eval/`,
   run them via `uv run --with <evaluate_container_python_requirements>`,
   parse the JSON metric and persist `<workspace>/final_score.json`.
6. Append a record (with `our_score`, `sota_score`, `beat_sota`,
   `lower_is_better`) to `<output_root>/results.json`.

Model assignments for this benchmark live in
`bench_workflows/model_configs/airsbench_roles.yaml`. Edit that yaml to swap
models — no `--models` CLI flag.

The HuggingFace model-search skill at
`.scider/skills/huggingface-model-search/` is force-preloaded for the
experiment / coding agents on every airsbench run (see
`PRELOAD_SKILLS` in `airsbench_workflow.py`), so the agent always starts with
the recipe + reference table for picking a backbone.

### Smoke test (one task)

```bash
export PATH=$HOME/.local/bin:$PATH   # if uv isn't on PATH
set -a; source .env; set +a

uv run python -m bench_workflows.airsbench_workflow \
  --tasks-root   benchmarks/airsbench/airs-bench/airsbench/tasks/rad \
  --data-root    /path/to/datasets_download_location \
  --output-root  /path/to/airs-bench/workspace \
  --tasks        TextualClassificationSickAccuracy
```

### Full 20-task run (resumable)

```bash
uv run python -m bench_workflows.airsbench_workflow \
  --tasks-root   benchmarks/airsbench/airs-bench/airsbench/tasks/rad \
  --data-root    /path/to/datasets_download_location \
  --output-root  /path/to/airs-bench/workspace \
  --skip-existing
```

`--skip-existing` skips any task whose `<output_root>/<task>/final_score.json`
already exists and parses as JSON, so you can rsync the repo, kill, resume,
etc. without redoing finished tasks.

Useful flags:

| flag | purpose |
|---|---|
| `--skip-existing` | resume after interruption (gates on a parseable `final_score.json`) |
| `--limit N` | dev/smoke: process at most the first N tasks |
| `--tasks A,B,C` | run only these task names — overrides `--limit` |
| `--max-revisions N` | upper bound on critic/approval retries (default 3) |
| `--experiment-recursion-limit N` | LangGraph recursion limit for the experiment agent (default 512) |

### Web search backend

The experiment agent's `WebSearch` tool can run on either DuckDuckGo or
Tavily, selected by `WEB_SEARCH_VERSION` in `.env` (default `duckduckgo`).
For airsbench we strongly recommend **Tavily** — its LLM-optimized snippets
+ score-ranked results give the agent better paper / leaderboard hits when
hunting for newer SOTA. Set:

```bash
WEB_SEARCH_VERSION=tavily
TAVILY_API_KEY=...   # https://app.tavily.com/
```

## 3. Compute normalized score

After a run finishes (or a partial run), evaluate the AIRS-Bench paper's
*normalized score* on the tasks present in `results.json`:

```bash
uv run python benchmarks/airsbench/airsbench_eval.py \
  --results /path/to/airs-bench/workspace/results.json
```

Per the paper:

```
phi(s) = -log10(|s - s_opt|)
NS     = (phi(s) - phi(s_min)) / (phi(s_sota) - phi(s_min))
```

where `s_opt`, `s_min`, `s_sota` come from each task's `metadata.yaml`
(`optimal_score`, `estimated_worst_score`, best `sota[*].sota_score`). NS = 1
matches prior SOTA, NS > 1 beats it, NS = 0 matches the worst observed run.

The script prints a per-task table plus the mean NS across scored tasks, and
optionally writes the full report to JSON via `--out report.json`.

## Layout

```
benchmarks/airsbench/
├── airs-bench/             # upstream submodule (CC BY-NC 4.0)
│   ├── airsbench/tasks/rad/<TaskName>/
│   │   ├── metadata.yaml          # SOTA, metric, shape, deps, etc.
│   │   ├── project_description.md # task prompt seen by the agent
│   │   ├── prepare.py             # train/test/val materializer
│   │   ├── evaluate.py            # graders (withheld from agent)
│   │   └── evaluate_prepare.py
│   └── datasets/                  # download_hf_datasets.sh + csv
├── airsbench_eval.py       # normalized-score evaluator (this dir)
└── README.md               # this file
```

Per-task workspace (under `--output-root`) after a run:

```
<TaskName>/
├── pyproject.toml          # uv-init'd
├── data/{train,test,validation}/   # materialized by prepare.py
├── prepare.py              # copied from upstream task dir
├── metadata.yaml           # copied
├── project_description.md  # copied
├── train.py / experiment.py        # written by the agent
├── submission.csv          # the deliverable
├── final_score.json        # parsed metric from evaluate.py
├── _eval/                  # evaluator sandbox (post-agent only)
├── data_analysis.md        # if data agent ran (currently skipped)
├── experiment_summary.md   # agent's run summary
├── experiment_agent_history.json
└── subagents/              # full conversation traces
```
