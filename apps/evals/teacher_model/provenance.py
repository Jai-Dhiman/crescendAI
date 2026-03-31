"""
Append-only JSONL provenance manifest for the CPT training corpus.

Every document ingested into the corpus is recorded here for legal and
compliance audits.  The manifest is written to by the transcription pipeline
(Task 6) and text extraction pipeline (Task 8).

Source tier values:
  "tier1_youtube"      - YouTube lecture / lesson recordings
  "tier2_literature"   - Published pedagogical literature
  "tier3_musicology"   - Musicology / theory texts
  "tier4_own"          - First-party CrescendAI content
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ProvenanceRecord:
    url: str
    title: str
    channel_or_publisher: str
    download_timestamp: str          # ISO-8601, e.g. "2026-03-30T12:00:00Z"
    license_claimed: str             # e.g. "CC BY 4.0", "fair use", "unknown"
    word_count: int
    inclusion_threshold_score: Optional[float]
    source_tier: str                 # one of the four tier values above

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "ProvenanceRecord":
        data = json.loads(s)
        return cls(**data)


class ProvenanceManifest:
    """Append-only JSONL provenance manifest."""

    DEFAULT_PATH = Path(__file__).parent / "data" / "provenance.jsonl"

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else self.DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: ProvenanceRecord) -> None:
        """Append a single record to the manifest."""
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")

    def _records(self) -> list[ProvenanceRecord]:
        if not self.path.exists():
            return []
        records: list[ProvenanceRecord] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(ProvenanceRecord.from_json(line))
        return records

    def count(self) -> int:
        """Total number of records in the manifest."""
        return len(self._records())

    def total_words(self) -> int:
        """Sum of word_count across all records."""
        return sum(r.word_count for r in self._records())

    def by_tier(self) -> dict[str, int]:
        """Count of records per source_tier."""
        return dict(Counter(r.source_tier for r in self._records()))

    def summary(self) -> str:
        records = self._records()
        total = len(records)
        words = sum(r.word_count for r in records)
        tier_counts = Counter(r.source_tier for r in records)
        lines = [
            f"Provenance manifest: {self.path}",
            f"  total records : {total}",
            f"  total words   : {words:,}",
            "  by tier:",
        ]
        for tier in sorted(tier_counts):
            lines.append(f"    {tier}: {tier_counts[tier]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = ProvenanceManifest(path=Path(tmpdir) / "test_provenance.jsonl")

        record = ProvenanceRecord(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="Piano Technique Masterclass",
            channel_or_publisher="Test Channel",
            download_timestamp="2026-03-30T12:00:00Z",
            license_claimed="CC BY 4.0",
            word_count=1234,
            inclusion_threshold_score=0.87,
            source_tier="tier1_youtube",
        )
        manifest.add(record)

        assert manifest.count() == 1, f"expected 1 record, got {manifest.count()}"
        assert manifest.total_words() == 1234, f"expected 1234 words, got {manifest.total_words()}"
        assert manifest.by_tier() == {"tier1_youtube": 1}

        # round-trip serialization
        reconstructed = ProvenanceRecord.from_json(record.to_json())
        assert reconstructed == record, "round-trip serialization failed"

        print(manifest.summary())
        print("smoke test passed")


if __name__ == "__main__":
    _smoke_test()
