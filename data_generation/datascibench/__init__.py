"""DataSciBench SFT data generation (data + experiment trajectories).

Source: https://arxiv.org/abs/2502.13897
HF dataset: https://huggingface.co/datasets/zd21/DataSciBench

Pipeline:
    python -m data_generation.datascibench.generation \\
        --bench-root <DataSciBench-data dir> \\
        --output-root <out>

or run the bundled ``generate.sh``.

Each upstream subfolder under ``DataSciBench-data/<family>_<id>/`` becomes one
SciDER FullWorkflow run (data + experiment, no ideation, no paper writing).
The contents of the subfolder (minus its ``prompt.json``) are staged into
``<workspace>/inputs/`` and the prompt text is wrapped to point at the new
input path.

Unlike ds1000 there is no eval phase here — DataSciBench's upstream evaluator
is an LLM-as-judge that scores against task-specific output files; we don't
reproduce it. Every trajectory is kept; ``output.json.ok`` only requires the
two agent histories to exist and be non-empty.
"""
