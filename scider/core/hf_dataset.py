"""HuggingFace dataset download utility."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from . import constant


def resolve_data_path(data_path: str | Path) -> Path:
    """Resolve a data path to a local directory.

    If the path exists locally, return it as-is.
    If HF dataset download is enabled and the path doesn't exist,
    treat it as a HuggingFace repo ID and download the dataset.

    Args:
        data_path: Local path or HuggingFace dataset repo ID (e.g. ``google/fleurs``).

    Returns:
        Resolved local Path to the dataset directory.

    Raises:
        FileNotFoundError: If path doesn't exist and HF download is disabled.
        RuntimeError: If HF download fails.
    """
    path = Path(data_path)
    if path.exists():
        return path.resolve()

    if not constant.HF_DATASET_DOWNLOAD_ENABLED:
        raise FileNotFoundError(
            f"Data path does not exist: {data_path}. "
            "Set HF_DATASET_DOWNLOAD_ENABLED=true to enable HuggingFace dataset download."
        )

    repo_id = str(data_path).strip()
    local_dir = Path(constant.HF_DATASET_CACHE_DIR) / repo_id.replace("/", "_")

    # Cache hit: skip download if directory already has files
    if local_dir.exists() and any(local_dir.iterdir()):
        logger.info(f"Using cached HF dataset: {local_dir}")
        return local_dir.resolve()

    logger.info(f"Downloading HF dataset '{repo_id}' to {local_dir}")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            ignore_patterns=["*.gitattributes", ".gitignore"],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download HuggingFace dataset '{repo_id}': {e}") from e

    logger.info(f"HF dataset '{repo_id}' downloaded to {local_dir}")
    return local_dir.resolve()
