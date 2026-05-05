"""DS-1000 SFT data generation + scoring.

Source: https://huggingface.co/datasets/xlangai/DS-1000

Pipeline:
    python -m data_generation.ds1000.generation -o <out>   # produce trajectories
    python -m data_generation.ds1000.eval        -o <out>  # mark passed=true|false
or run the bundled `generate.sh` to do all three steps with sensible defaults.
"""
