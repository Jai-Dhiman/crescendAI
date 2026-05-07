"""Stage 5: publish train + validation manifests as a private HF Hub dataset."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi


_FEATURES = Features({
    "text": Value("string"),
    "source": Value("string"),
    "doc_id": Value("string"),
})


def _read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({"text": r["text"], "source": r["source"], "doc_id": r["doc_id"]})
    return rows


def _rows_to_dataset(rows: list[dict]) -> Dataset:
    if rows:
        return Dataset.from_list(rows, features=_FEATURES)
    table = pa.table(
        {"text": [], "source": [], "doc_id": []},
        schema=_FEATURES.arrow_schema,
    )
    return Dataset(table)


def run_publish(
    train_manifest: Path,
    val_manifest: Path,
    repo_id: str,
    private: bool = True,
) -> str:
    """Build a HF DatasetDict from the manifests and push to Hub."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Set it to a token with 'write' scope before running stage 5."
        )

    train_rows = _read_manifest(train_manifest)
    val_rows = _read_manifest(val_manifest)
    train_ds = _rows_to_dataset(train_rows)
    val_ds = _rows_to_dataset(val_rows)
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})

    api = HfApi()
    api.create_repo(
        repo_id=repo_id, private=private, repo_type="dataset", exist_ok=True, token=token,
    )
    dataset.push_to_hub(repo_id, private=private, token=token)

    return f"https://huggingface.co/datasets/{repo_id}"
