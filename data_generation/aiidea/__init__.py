"""AI Idea Bench SFT data generation (ideation trajectories).

Source: https://huggingface.co/datasets/yanshengqiu/AI_Idea_Bench_2025

Pipeline:
    python -m data_generation.aiidea.generation -o <out>   # produce trajectories
or run the bundled ``generate.sh`` for the same with sensible defaults.

Unlike DS-1000 there is no eval phase — this benchmark feeds research-seed
prompts (topic + motivation from a published paper) into the IdeationAgent
and persists every resulting ``ideation_agent_history.json`` as SFT data.
We do not score the ideas against the paper's ground-truth method; the goal
here is generation, not benchmark evaluation.
"""
