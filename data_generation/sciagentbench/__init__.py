"""ScienceAgentBench SFT data generation (data + experiment trajectories).

Source: https://arxiv.org/abs/2410.05080
HF dataset: https://huggingface.co/datasets/osunlp/ScienceAgentBench (split=verified)
Local benchmark dir on wm:
    /sciclone/proj-ds/ai4scientist/kelin/SciDER/sciagentbench/benchmark
    ├── datasets/<folder>/         # per-task data files
    ├── eval_programs/             # upstream evaluator scripts (we don't run)
    ├── gold_programs/             # reference solutions (we don't show agent)
    └── scoring_rubrics/           # per-task rubrics

Pipeline:
    python -m data_generation.sciagentbench.generation \\
        --bench-root <benchmark dir> \\
        --output-root <out>

For each verified-split row we hand the upstream task instruction (+ optional
domain knowledge + dataset preview) to SciDER's FullWorkflow (data +
experiment, no ideation, no paper writing). The two agent histories become
SFT trajectories.

We don't run the upstream evaluator here — the goal is data, not metric.
``output.json.ok`` only requires the workflow to have produced both histories.
``--use-knowledge`` matches upstream's flag for revealing ``domain_knowledge``
in the prompt.

uid format: ``sciagentbench_<instance_id>``  e.g. ``sciagentbench_1``.
"""
