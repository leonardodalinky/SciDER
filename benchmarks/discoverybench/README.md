# Benchmark for DiscoveryBench

[DiscoveryBench](https://github.com/allenai/discoverybench) (arXiv:2407.01725)
is a data-driven scientific discovery benchmark: each task is a natural
language question + tabular CSV(s) + column metadata, and the agent must
return a natural-language **hypothesis** describing the discovered
relationship. Upstream evaluates hypotheses with an LLM-as-judge that scores
along three axes (context / variables / relations) and emits a `final_score`
in `[0, 1]`.

We assume all commands are run from the same dir as this README file unless
otherwise noted.

> **Heads up**: the benchmark is small and self-contained — every task
> directory ships its own `metadata_*.json` + CSV(s). No download needed.

## 1. Environment setup

The bench workflow itself runs in **SciDER's default uv env** (already
provisioned at the repo root). pandas / numpy / scikit-learn / statsmodels
are all available there.

The **eval companion** ([`discovery_eval_scider.py`](discoverybench/eval/discovery_eval_scider.py))
calls upstream's scorer (`new_eval.run_eval_gold_vs_gen_NL_hypo_workflow`),
which only needs `openai`, `tenacity`, `sympy`, and `IPython` (no langchain
— that's only used by upstream's *agent* side, which we replace). These are
already installed into SciDER's main venv (`uv add sympy ipython tenacity`
was run during integration).

You also need a working `OPENAI_API_KEY` exported in the shell — the upstream
judge calls the OpenAI API directly (default model: `gpt-4o`, which replaces
the retired `gpt-4-1106-preview` upstream pinned).

## 2. Data layout

Tasks live under `discoverybench/discoverybench/{real,synth}/{train,dev,test}/`.
Each task dir holds one or more `metadata_N.json` files plus the CSV(s) they
reference via `datasets[].name`.

- **synth** — 76 test tasks; each is a single `data.csv` with ~50 columns.
- **real** — 12 test tasks; multi-file CSVs per task; metadata also carries
  optional `domain_knowledge` and `workflow_tags` hints.

Gold answers live at `discoverybench/eval/answer_key_{synth,real}.csv`,
keyed by `(dataset, metadataid, query_id)`.

## 3. Run SciDER on the benchmark

Models are pinned in
[`bench_workflows/model_configs/discoverybench_roles.yaml`](../../bench_workflows/model_configs/discoverybench_roles.yaml).
Edit the YAML to swap models — there is **no `--models` CLI flag**.

The workflow is the **full** SciDER pipeline (data agent + experiment agent,
no ideation, no paper writing). The deliverable is `./result.json` per
task workspace, with this exact schema:

```json
{"hypothesis": "<one or two sentences>", "workflow": "<numbered analytical steps>"}
```

```bash
# from SciEvo/ repo root — real/test (default).
uv run python -m bench_workflows.discoverybench_workflow \
  --bench-root  benchmarks/discoverybench/discoverybench/discoverybench \
  --split-type  real \
  --split       test \
  --output-root benchmarks/discoverybench/workspace \
  --skip-existing
```

To run synth instead:

```bash
uv run python -m bench_workflows.discoverybench_workflow \
  --bench-root  benchmarks/discoverybench/discoverybench/discoverybench \
  --split-type  synth \
  --split       test \
  --output-root benchmarks/discoverybench/workspace_synth \
  --skip-existing
```

Behaviour:
- Per-uid workspace at `<output-root>/<uid>/` (uid format
  `{dataset}__m{metadataid}__q{qid}`, e.g. `archaeology__m0__q42`),
  containing `inputs/` (symlinks to upstream CSVs), `metadata.json` (copy),
  `task.md` (full prompt), `result.json` (deliverable),
  `data_agent_history.json` + `experiment_agent_history.json` + `subagents/`
  (debug trace).
- Top-level `<output-root>/results.json` accumulates per-task records.
  `--skip-existing` resumes from there.

Useful flags:

| flag | purpose |
|---|---|
| `--split-type {real,synth}` | which task family. Default `real`. |
| `--split test` | which split under `--split-type`. Default `test`. |
| `--use-domain-knowledge` | (real-only) inject `metadata.domain_knowledge` hint into the prompt. Off by default to match the canonical no-hint setting. |
| `--use-workflow-tags` | (real-only) inject `metadata.workflow_tags` as a hint about which analytical techniques to consider. |
| `--skip-existing` | skip uids whose `result.json` already parses with a non-empty `hypothesis`. |
| `--limit N` | dev/smoke-test: only process the first N tasks. |
| `--uids u1,u2,...` | run only these uids — overrides `--limit`. |
| `--max-revisions N` | critic/approval retry budget (subagent gate may stop earlier). Default 1. |
| `--data-recursion-limit N` | recursion limit for DataAgent. Default 80. |
| `--exp-recursion-limit N` | recursion limit for ExperimentAgent. Default 100. |

Smoke test on 2 tasks:

```bash
uv run python -m bench_workflows.discoverybench_workflow \
  --bench-root  benchmarks/discoverybench/discoverybench/discoverybench \
  --split-type  real --split test \
  --output-root /tmp/disc_smoke \
  --limit 2
```

## 4. Run the eval companion

The eval companion sits next to upstream's `new_eval.py`. It reads
`<workspace-root>/results.json`, joins each row with the answer key, calls
`run_eval_gold_vs_gen_NL_hypo_workflow` per task, and prints / dumps an
aggregate.

**Important**: do NOT `cd` into `discoverybench/eval/` before running. That
directory contains an `eval.py` file which Python's auto-prepended sys.path[0]
shadows the `eval/` namespace package with — break the import chain.
Run from the repo root using an absolute script path:

```bash
# from SciEvo/ repo root
set -a && source .env && set +a   # export OPENAI_API_KEY
uv run python benchmarks/discoverybench/discoverybench/eval/discovery_eval_scider.py \
  --workspace-root /home/link/github/SciEvo/benchmarks/discoverybench/workspace \
  --bench-root     /home/link/github/SciEvo/benchmarks/discoverybench/discoverybench/discoverybench \
  --split-type     real \
  --split          test \
  --outfile        /tmp/disc_eval.json
```

Sample output:

```
Loaded 12 gold answers from .../answer_key_real.csv

Wrote 12 records to /tmp/disc_eval.json
Miss (excluded from scoring): 1/12
Scored: 11 — mean final_score: 0.4321
Per-domain means:
  archaeology: 0.5500
  nls_incarceration: 0.3000
  ...
```

Records labelled **Miss** (no `result.json`, agent failed, gold answer not
in answer key) are written to the JSON for debugging but are excluded from
the scored denominator — same convention used by AstroVisBench.

Optional flags:

| flag | purpose |
|---|---|
| `--llm gpt-4-1106-preview` | judge model id. Default mirrors upstream. |
| `--answer-key path` | override auto-resolved `answer_key_{synth,real}.csv`. |
| `--uids ...` / `--limit N` | filter records (debug). |

## Layout

```
benchmarks/discoverybench/
├── discoverybench/                       # upstream submodule (Apache 2.0)
│   ├── discoverybench/{real,synth}/...   # task data
│   └── eval/
│       ├── new_eval.py                   # upstream scorer
│       ├── answer_key_{synth,real}.csv   # gold hypotheses
│       └── discovery_eval_scider.py      # OUR eval companion
├── workspace/                            # per-uid SciDER runs (gitignored)
└── README.md                             # this file
```
