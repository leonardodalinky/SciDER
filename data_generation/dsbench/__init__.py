"""DSBench SFT data generation (data + experiment trajectories).

Source: https://github.com/LiqiangJing/DSBench (NeurIPS '24)
HF dataset: https://huggingface.co/datasets/liqiang888/DSBench
Local benchmark dir on wm:
    /sciclone/proj-ds/ai4scientist/kelin/SciDER/dsbench/dsbench-data
    ├── data_analysis/
    │   ├── data.json          # 40 jsonl rows (id, name, questions, answers, ...)
    │   └── data/<id>/         # one task per id (e.g. 00000001)
    │       ├── *.xlsx         # the spreadsheet to analyse
    │       ├── introduction.txt
    │       └── questionN.txt  # one file per Q
    └── data_modeling/
        └── data/
            ├── task/<comp>.txt          # Kaggle competition description
            ├── data_resplit/<comp>/     # train.csv / test.csv / sampleSubmission.csv
            │                            # (test labels withheld; resplit by upstream)
            └── answers/<comp>/test_answer.csv   # held-out gold (we DO NOT expose)

Pipeline:
    python -m data_generation.dsbench.generation \\
        --bench-root <dsbench-data dir> \\
        --output-root <out> \\
        --family {analysis|modeling}

For each task we hand the upstream prompt to SciDER's FullWorkflow
(data + experiment, no ideation, no paper writing). The two agent
histories become SFT trajectories.

We don't run the upstream evaluator here — the goal is data, not metric.
``output.json.ok`` only requires both agent histories to exist and be
non-empty. The upstream gold answers stay on disk but are NEVER copied
into the workspace or shown to the agent.

uid format:
    analysis  → ``dsbench_analysis_<id>``    e.g. ``dsbench_analysis_00000001``
    modeling  → ``dsbench_modeling_<comp>``  e.g. ``dsbench_modeling_titanic``
"""
