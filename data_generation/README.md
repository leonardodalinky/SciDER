# Data Generation Workflows

These workflows mirror `bench_workflows/` in CLI shape (`--output-root`,
`--skip-existing`, per-uid workspaces, append-only `results.json`) but the
goal is to **produce SFT trajectories** from public benchmark datasets, not
to evaluate.

Each per-task workspace contains:

```
<output_root>/<uid>/
├── prompt.md                   # the wrapped task prompt (debug aid)
├── code.py                     # agent's solution (deliverable)
├── coding_agent_history.json   # ← the trajectory; this is the training data
└── output.json                 # skip-existing marker {ok, passed, ...}
```

The trajectory file is named so that `train/prepare_data.py` picks it up
automatically (it scans `*_agent_history.json` and `subagents/*.json`).

## Layout

Each benchmark lives in its own subpackage so the CLI module path,
roles.yaml, and pipeline `generate.sh` stay together:

```
data_generation/
├── ds1000/                    # coding trajectories
│   ├── _common.py             # run_coding_task, scan_completed_uids, ...
│   ├── generation.py          # python -m data_generation.ds1000.generation
│   ├── eval.py                # python -m data_generation.ds1000.eval
│   ├── roles.yaml             # role assignments for this bench
│   └── generate.sh            # one-shot pipeline (gen → score → filter)
├── aiidea/                    # ideation trajectories (no eval)
│   ├── _common.py             # run_ideation_task, scan_completed_uids, ...
│   ├── generation.py          # python -m data_generation.aiidea.generation
│   ├── roles.yaml
│   └── generate.sh            # one-shot pipeline (gen → emit traj list)
├── datascibench/              # data + experiment trajectories (no eval)
│   ├── _common.py             # discover_tasks, stage_inputs, run_full_workflow_task
│   ├── generation.py          # python -m data_generation.datascibench.generation
│   ├── roles.yaml
│   └── generate.sh            # one-shot pipeline (gen → emit workspace list)
├── sciagentbench/             # data + experiment trajectories (no eval)
│   ├── _common.py             # load_tasks, stage_inputs (symlink), run_full_workflow_task
│   ├── generation.py          # python -m data_generation.sciagentbench.generation
│   ├── roles.yaml
│   └── generate.sh
└── dsbench/                   # data + experiment trajectories (analysis + modeling families)
    ├── _common.py             # discover_{analysis,modeling}_tasks, stage_*_inputs (symlink), run_full_workflow_task
    ├── generation.py          # python -m data_generation.dsbench.generation --family {analysis|modeling|both}
    ├── roles.yaml
    └── generate.sh
```

## Available benchmarks

| Source | Subpackage | HF dataset | Trajectory type |
|---|---|---|---|
| DS-1000 | [`data_generation.ds1000`](ds1000/) | `xlangai/DS-1000` | coding (with eval) |
| AI-Idea-Bench | [`data_generation.aiidea`](aiidea/) | `yanshengqiu/AI_Idea_Bench_2025` | ideation (no eval) |
| DataSciBench | [`data_generation.datascibench`](datascibench/) | `zd21/DataSciBench` (uses local-fs copy) | data+experiment (no eval) |
| ScienceAgentBench | [`data_generation.sciagentbench`](sciagentbench/) | `osunlp/ScienceAgentBench` (verified split) + local `benchmark/datasets/` | data+experiment (no eval) |
| DSBench | [`data_generation.dsbench`](dsbench/) | `liqiang888/DSBench` (analysis + modeling, local-fs unzip) | data+experiment (no eval) |

## Running

```bash
# DS-1000: generate, score, write a passed-trajectory list
bash data_generation/ds1000/generate.sh
LIBRARY=Pandas LIMIT=20 bash data_generation/ds1000/generate.sh
OUTPUT_ROOT=/scratch/me/ds1000 bash data_generation/ds1000/generate.sh

# AI-Idea-Bench: generate ideation trajectories (no eval)
bash data_generation/aiidea/generate.sh
LIMIT=20 bash data_generation/aiidea/generate.sh
OUTPUT_ROOT=/scratch/me/aiidea bash data_generation/aiidea/generate.sh
```

Each script defaults to `data_generation/<bench>/dataset/` for output and
writes a `workspace.list` of absolute workspace dirs (passed-only for
ds1000; ok=true for aiidea / datascibench) that you feed to
`train/prepare_data.py`:

```bash
python train/prepare_data.py \
    --workspace-list data_generation/ds1000/dataset/workspace.list \
    --out raw_datafiles/ds1000.jsonl \
    --id-level 1
```

## Adding a new benchmark

1. `mkdir data_generation/<bench>/`
2. Drop in:
   - `__init__.py`
   - `generation.py` (modelled on [`ds1000/generation.py`](ds1000/generation.py)):
     - `_load_<bench>(...)` returning row dicts
     - `_build_user_query(row, code_filename)` (use `output_protocol_block`
       from `_common` if standalone-script semantics fit, otherwise hand-roll
       like ds1000 does for snippet-fill semantics)
     - `_build_uid(...)` for deterministic workspace names
   - `eval.py` (optional — only if the source has a programmatic test harness)
   - `roles.yaml` pointing at the model that drives the coding subagent
   - `generate.sh` mirroring [`ds1000/generate.sh`](ds1000/generate.sh) — just
     change the module path
3. Skip-existing comes for free — `_common.is_workspace_complete` works on
   any workspace with a valid `output.json`.

## Coding-agent backend

Controlled by `CODING_AGENT_VERSION`:
- `native` (default for openscider) — `coding_subagent_native`, drives the
  model registered for `experiment_coding` (the role yamls point this at
  `openscider-qwen3.6-27b`).
- `claude_sdk` — Claude Agent SDK (requires `ANTHROPIC_API_KEY`).

## What we don't do here

- We do **not** filter trajectories. `eval.py` writes `passed: bool` per
  workspace; `generate.sh` builds a passed-only path list as a hint, but
  preserving every trajectory (failed included) is intentional in case you
  want to learn from negative examples too.
- We do **not** run the upstream evaluator scripts; we re-implement the
  minimum needed (substitute solution into `code_context`, run subprocess).
  No DS-1000 install required.
