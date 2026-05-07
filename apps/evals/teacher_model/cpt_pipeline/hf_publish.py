"""Stage 5: publish train + validation manifests as a private HF Hub dataset."""
from __future__ import annotations

import os
from pathlib import Path


def run_publish(
    train_manifest: Path,
    val_manifest: Path,
    repo_id: str,
    private: bool = True,
) -> str:
    """Build a HF DatasetDict from the manifests and push to Hub.

    Returns the published Hub URL.
    Raises RuntimeError if HF_TOKEN env var is missing.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Set it to a token with 'write' scope before running stage 5."
        )
    raise NotImplementedError("subsequent tasks add the actual push")
