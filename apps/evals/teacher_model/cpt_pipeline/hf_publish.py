"""Stage 5: publish train + validation manifests as a private HF Hub dataset."""
from __future__ import annotations

import json
import os
from collections import Counter
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


def _word_count(rows: list[dict]) -> int:
    return sum(len(r["text"].split()) for r in rows)


def _source_breakdown(rows: list[dict]) -> Counter:
    return Counter(r["source"] for r in rows)


def _build_card(repo_id: str, train_rows: list[dict], val_rows: list[dict]) -> str:
    train_words = _word_count(train_rows)
    val_words = _word_count(val_rows)
    src_breakdown = _source_breakdown(train_rows) + _source_breakdown(val_rows)
    src_lines = "\n".join(f"- `{src}`: {count}" for src, count in sorted(src_breakdown.items()))
    return (
        f"# {repo_id}\n\n"
        f"Private intermediate corpus for piano-pedagogy CPT / SFT data synthesis.\n\n"
        f"## Splits\n\n"
        f"- `train`: {len(train_rows)} docs, {train_words} words\n"
        f"- `validation`: {len(val_rows)} docs, {val_words} words\n\n"
        f"## Per-source breakdown (train + validation)\n\n"
        f"{src_lines}\n\n"
        f"## Provenance\n\n"
        f"Mixed sources (YouTube transcripts, OpenAlex / Semantic Scholar / Internet Archive PDFs, "
        f"scraped pedagogy web pages). See per-source `provenance_*.jsonl` in the harvest pipeline "
        f"for license claims. Private dataset, internal use only; do not redistribute.\n"
    )


def run_publish(
    train_manifest: Path,
    val_manifest: Path,
    repo_id: str,
    private: bool = True,
    card_out_dir: Path | None = None,
) -> str:
    """Build a HF DatasetDict from the manifests, write a card, and push to Hub."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Set it to a token with 'write' scope before running stage 5."
        )

    train_rows = _read_manifest(train_manifest)
    val_rows = _read_manifest(val_manifest)

    if card_out_dir is not None:
        Path(card_out_dir).mkdir(parents=True, exist_ok=True)
        (Path(card_out_dir) / "README.md").write_text(
            _build_card(repo_id, train_rows, val_rows), encoding="utf-8",
        )

    train_ds = _rows_to_dataset(train_rows)
    val_ds = _rows_to_dataset(val_rows)
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})

    api = HfApi()
    api.create_repo(
        repo_id=repo_id, private=private, repo_type="dataset", exist_ok=True, token=token,
    )
    dataset.push_to_hub(repo_id, private=private, token=token)

    return f"https://huggingface.co/datasets/{repo_id}"
