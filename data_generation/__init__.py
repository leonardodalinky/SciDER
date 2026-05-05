"""Data-generation workflows.

These mirror ``bench_workflows/`` in CLI shape (``--output-root``,
``--skip-existing``, per-uid workspaces, append-only ``results.json``)
but their goal is to PRODUCE training trajectories from public benchmark
datasets, not to evaluate. Each workspace stores the agent's full message
history at ``<workspace>/<agent>_agent_history.json`` so that
``train/prepare_data.py`` can sweep them later for SFT.

To add a new source, see ``data_generation/README.md``.
"""
