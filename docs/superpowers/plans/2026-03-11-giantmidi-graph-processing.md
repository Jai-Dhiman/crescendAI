# GIANTMIDI Graph Processing Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert 10,854 GIANTMIDI-Piano MIDI files into PyG homogeneous graphs for S2 GNN pretraining expansion, with memory-safe processing on M4 (32 GB RAM).

**Architecture:** Refactor `graph.py` to accept pre-parsed PrettyMIDI objects (eliminating double-parse), then build a new processing script that converts MIDI files into graph shards (50 per shard) with RSS monitoring, writing to the existing `pretrain_cache/graphs/shards/` directory starting at shard 0075.

**Tech Stack:** PyTorch, PyTorch Geometric, pretty_midi, psutil

**Spec:** `docs/superpowers/specs/2026-03-11-giantmidi-graph-processing-design.md`

---

## Chunk 1: Refactor graph.py

### Task 1: Add `count_midi_notes` and `parsed_midi_to_graph` to graph.py

**Files:**
- Modify: `model/src/model_improvement/graph.py:164-274`
- Test: `model/tests/model_improvement/test_graph.py`

- [ ] **Step 1: Write tests for new functions**

Add to `model/tests/model_improvement/test_graph.py`. Import the new functions at the top alongside existing imports:

```python
from model_improvement.graph import (
    EDGE_TYPE_DURING,
    EDGE_TYPE_FOLLOW,
    EDGE_TYPE_ONSET,
    EDGE_TYPE_SILENCE,
    assign_voices,
    count_midi_notes,
    midi_to_graph,
    midi_to_hetero_graph,
    parsed_midi_to_graph,
    sample_negative_edges,
)
```

Then add these test classes after the existing `TestMidiToGraph` class:

```python
class TestCountMidiNotes:
    def test_counts_notes_across_instruments(self):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        piano = pretty_midi.Instrument(program=0)
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.0, end=1.0))
        midi.instruments.append(piano)
        assert count_midi_notes(midi) == 2

    def test_excludes_drum_instruments(self):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        piano = pretty_midi.Instrument(program=0)
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        drums.notes.append(pretty_midi.Note(velocity=80, pitch=36, start=0.0, end=1.0))
        midi.instruments.append(piano)
        midi.instruments.append(drums)
        assert count_midi_notes(midi) == 1

    def test_empty_midi_returns_zero(self):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        midi.instruments.append(pretty_midi.Instrument(program=0))
        assert count_midi_notes(midi) == 0


class TestParsedMidiToGraph:
    def test_matches_midi_to_graph_output(self):
        """parsed_midi_to_graph should produce identical output to midi_to_graph."""
        path = _make_midi([
            (60, 80, 0.0, 1.0),
            (64, 80, 0.0, 1.0),
            (67, 80, 1.0, 2.0),
            (72, 80, 2.0, 3.0),
        ])
        graph_from_path = midi_to_graph(path)
        midi_obj = pretty_midi.PrettyMIDI(str(path))
        graph_from_parsed = parsed_midi_to_graph(midi_obj)

        assert torch.allclose(graph_from_path.x, graph_from_parsed.x)
        assert torch.equal(graph_from_path.edge_index, graph_from_parsed.edge_index)
        assert torch.equal(graph_from_path.edge_type, graph_from_parsed.edge_type)

    def test_empty_midi_raises(self):
        midi = pretty_midi.PrettyMIDI()
        midi.instruments.append(pretty_midi.Instrument(program=0))
        with pytest.raises(ValueError, match="No notes found"):
            parsed_midi_to_graph(midi)

    def test_pedal_feature_preserved(self):
        """Pedal CC64 should be captured from parsed MIDI object."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        piano = pretty_midi.Instrument(program=0)
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=1.0, end=2.0))
        piano.control_changes.append(
            pretty_midi.ControlChange(number=64, value=127, time=0.5)
        )
        midi.instruments.append(piano)

        data = parsed_midi_to_graph(midi)
        assert data.x[1, 4].item() == 1.0  # second note has pedal on
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd model && uv run pytest tests/model_improvement/test_graph.py::TestCountMidiNotes -v && uv run pytest tests/model_improvement/test_graph.py::TestParsedMidiToGraph -v`
Expected: FAIL with `ImportError` (functions don't exist yet)

- [ ] **Step 3: Implement `count_midi_notes` and `parsed_midi_to_graph` in graph.py**

Add `count_midi_notes` right before the existing `midi_to_graph` function (before line 164):

```python
def count_midi_notes(midi: pretty_midi.PrettyMIDI) -> int:
    """Count non-drum notes in a parsed PrettyMIDI object.

    Args:
        midi: Already-parsed PrettyMIDI object.

    Returns:
        Total number of non-drum notes across all instruments.
    """
    return sum(
        len(inst.notes) for inst in midi.instruments if not inst.is_drum
    )
```

Replace the body of `midi_to_graph` (lines 188-274) to delegate to `parsed_midi_to_graph`:

```python
def midi_to_graph(
    midi_path: str | Path,
    max_voices: int = 8,
    follow_tolerance: float = 0.05,
) -> Data:
    """Convert a MIDI file to a PyTorch Geometric Data object.

    Node features (6-dim): [pitch, velocity, onset, duration, pedal, voice]
    All features normalized to [0, 1].

    Edge types: onset (0), during (1), follow (2), silence (3).
    Edges are bidirectional.

    Args:
        midi_path: Path to MIDI file.
        max_voices: Maximum voices for voice assignment.
        follow_tolerance: Time tolerance (seconds) for follow edges.

    Returns:
        PyG Data object with x, edge_index, edge_type attributes.

    Raises:
        ValueError: If the MIDI file contains no notes.
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    try:
        return parsed_midi_to_graph(midi, max_voices, follow_tolerance)
    except ValueError:
        raise ValueError(f"No notes found in {midi_path}")
```

Add `parsed_midi_to_graph` right after `midi_to_graph`:

```python
def parsed_midi_to_graph(
    midi: pretty_midi.PrettyMIDI,
    max_voices: int = 8,
    follow_tolerance: float = 0.05,
) -> Data:
    """Build a homogeneous graph from an already-parsed PrettyMIDI object.

    Same output contract as midi_to_graph(). Use this when you already
    have a parsed PrettyMIDI object (e.g., after checking note count)
    to avoid double-parsing.

    Args:
        midi: Already-parsed PrettyMIDI object.
        max_voices: Maximum voices for voice assignment.
        follow_tolerance: Time tolerance (seconds) for follow edges.

    Returns:
        PyG Data object with x, edge_index, edge_type attributes.

    Raises:
        ValueError: If the MIDI contains no non-drum notes.
    """
    # Collect all notes across instruments
    all_notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            all_notes.extend(instrument.notes)

    if not all_notes:
        raise ValueError("No notes found in MIDI")

    # Sort by onset then pitch
    all_notes.sort(key=lambda n: (n.start, n.pitch))

    total_duration = max(n.end for n in all_notes)
    if total_duration == 0:
        total_duration = 1.0  # avoid division by zero

    # Build CC64 (sustain pedal) lookup from all instruments
    pedal_events: list[tuple[float, bool]] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for cc in instrument.control_changes:
            if cc.number == 64:
                pedal_events.append((cc.time, cc.value >= 64))
    pedal_events.sort(key=lambda x: x[0])

    def pedal_at(time: float) -> float:
        """Return 1.0 if sustain pedal is on at given time, else 0.0."""
        state = False
        for t, on in pedal_events:
            if t > time:
                break
            state = on
        return 1.0 if state else 0.0

    # Voice assignment
    voices = assign_voices(all_notes, max_voices=max_voices)
    max_voice_id = max(voices) if voices else 0
    voice_norm = max_voice_id if max_voice_id > 0 else 1

    # Node features: [pitch, velocity, onset, duration, pedal, voice]
    features = []
    onsets = np.zeros(len(all_notes))
    offsets = np.zeros(len(all_notes))

    for i, note in enumerate(all_notes):
        pitch = (note.pitch - 21) / 87.0  # Piano range A0(21) to C8(108)
        velocity = note.velocity / 127.0
        onset = note.start / total_duration
        duration = (note.end - note.start) / total_duration
        pedal = pedal_at(note.start)
        voice = voices[i] / voice_norm

        # Clamp to [0, 1]
        pitch = max(0.0, min(1.0, pitch))
        velocity = max(0.0, min(1.0, velocity))
        onset = max(0.0, min(1.0, onset))
        duration = max(0.0, min(1.0, duration))

        features.append([pitch, velocity, onset, duration, pedal, voice])
        onsets[i] = note.start
        offsets[i] = note.end

    x = torch.tensor(features, dtype=torch.float32)

    # Build edges
    edges, edge_types = _build_edges(onsets, offsets, follow_tolerance)

    # Make bidirectional
    bi_edges = []
    bi_types = []
    for (src, dst), etype in zip(edges, edge_types):
        bi_edges.append((src, dst))
        bi_edges.append((dst, src))
        bi_types.append(etype)
        bi_types.append(etype)

    if bi_edges:
        edge_index = torch.tensor(bi_edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(bi_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_type=edge_type)
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `cd model && uv run pytest tests/model_improvement/test_graph.py::TestCountMidiNotes tests/model_improvement/test_graph.py::TestParsedMidiToGraph -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run ALL existing graph tests to verify no regressions**

Run: `cd model && uv run pytest tests/model_improvement/test_graph.py -v`
Expected: All tests PASS (existing tests use `midi_to_graph` which now delegates to `parsed_midi_to_graph`)

- [ ] **Step 6: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai
git add model/src/model_improvement/graph.py model/tests/model_improvement/test_graph.py
git commit -m "refactor: extract parsed_midi_to_graph and count_midi_notes from graph.py

Eliminates double-parsing when processing MIDI files: parse once,
check note count, then build graph from the parsed object. Existing
midi_to_graph() API unchanged -- it delegates to parsed_midi_to_graph()."
```

---

## Chunk 2: Build the processing script

### Task 2: Create `process_giantmidi_graphs.py`

**Files:**
- Create: `model/scripts/process_giantmidi_graphs.py`

**Dependencies:** Task 1 must be complete (needs `count_midi_notes` and `parsed_midi_to_graph`)

- [ ] **Step 1: Write the processing script**

Create `model/scripts/process_giantmidi_graphs.py`:

```python
"""Convert GIANTMIDI-Piano MIDI files into PyG graph shards for S2 pretraining.

Memory-safe processing for M4 (32 GB RAM):
- Single MIDI parse per file (no double-parse)
- Homogeneous graphs only (no hetero -- S2H lost)
- 50-graph shards with RSS monitoring
- Resumable via manifest file

Usage:
    cd model
    uv run python scripts/process_giantmidi_graphs.py
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import psutil
import pretty_midi
import torch

from model_improvement.graph import count_midi_notes, parsed_midi_to_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
MIDI_DIR = Path(__file__).parent.parent / "data/giantmidi_raw/GiantMIDI-PIano/midis"
SHARD_DIR = Path(__file__).parent.parent / "data/pretrain_cache/graphs/shards"
MANIFEST_PATH = Path(__file__).parent.parent / "data/giantmidi_raw/giantmidi_manifest.json"

MAX_NOTES = 5_000
SHARD_SIZE = 50
RSS_FLUSH_THRESHOLD_GB = 20.0
FIRST_SHARD_ID = 75  # existing shards are 0000-0074


def _extract_youtube_id(filename: str) -> str:
    """Extract YouTube ID from GIANTMIDI filename.

    Filenames follow: "Composer, Title, YouTubeID.mid"
    The YouTube ID is the last comma-separated field before .mid.

    Args:
        filename: MIDI filename (with .mid extension).

    Returns:
        YouTube ID string.

    Raises:
        ValueError: If filename doesn't contain a comma separator.
    """
    stem = filename.removesuffix(".mid")
    parts = stem.rsplit(", ", 1)
    if len(parts) < 2:
        # Fallback: comma without trailing space
        parts = stem.rsplit(",", 1)
    if len(parts) < 2:
        raise ValueError(f"Cannot extract YouTube ID from filename: {filename}")
    return parts[1].strip()


def _get_rss_gb() -> float:
    """Get current process RSS in GB."""
    return psutil.Process().memory_info().rss / (1024 ** 3)


def _load_manifest() -> dict:
    """Load existing manifest for resumption, or return empty template."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {
        "processed_keys": [],
        "skipped": [],
        "failed": [],
        "shard_range": [FIRST_SHARD_ID, FIRST_SHARD_ID],
        "note_count_buckets": {
            "0-500": 0, "500-1000": 0, "1000-2000": 0,
            "2000-3000": 0, "3000-5000": 0, "5000+_skipped": 0,
        },
        "total_processing_time_seconds": 0.0,
    }


def _save_manifest(manifest: dict) -> None:
    """Save manifest to disk."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def _bucket_note_count(n: int) -> str:
    """Return histogram bucket key for a note count."""
    if n < 500:
        return "0-500"
    elif n < 1000:
        return "500-1000"
    elif n < 2000:
        return "1000-2000"
    elif n < 3000:
        return "2000-3000"
    elif n <= MAX_NOTES:
        return "3000-5000"
    else:
        return "5000+_skipped"


def _save_shard(shard_id: int, graphs: dict[str, object]) -> None:
    """Save a graph shard to disk."""
    path = SHARD_DIR / f"graphs_{shard_id:04d}.pt"
    torch.save(graphs, path)
    logger.info(
        "  Shard %04d saved (%d graphs, RSS=%.1f GB)",
        shard_id, len(graphs), _get_rss_gb(),
    )


def main() -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    processed_set = set(manifest["processed_keys"])
    skipped_set = {entry["key"] for entry in manifest["skipped"]}
    failed_set = {entry["key"] for entry in manifest["failed"]}
    already_handled = processed_set | skipped_set | failed_set

    # Discover MIDI files
    midi_files = sorted(f.name for f in MIDI_DIR.iterdir() if f.suffix == ".mid")
    logger.info("Found %d MIDI files in %s", len(midi_files), MIDI_DIR)

    # Filter to unprocessed files
    remaining = []
    for filename in midi_files:
        try:
            yt_id = _extract_youtube_id(filename)
        except ValueError as e:
            logger.warning("Skipping %s: %s", filename, e)
            continue
        key = f"giantmidi/{yt_id}"
        if key not in already_handled:
            remaining.append((filename, key))

    logger.info(
        "%d already processed, %d skipped, %d failed, %d remaining",
        len(processed_set), len(skipped_set), len(failed_set), len(remaining),
    )

    if not remaining:
        logger.info("Nothing to process.")
        return

    # Determine starting shard ID from manifest (authoritative source)
    next_shard_id = manifest["shard_range"][1] if processed_set else FIRST_SHARD_ID

    # Process files
    shard_buffer: dict[str, object] = {}
    t_start = time.time()
    processed_this_run = 0
    t_last_log = t_start

    for i, (filename, key) in enumerate(remaining):
        midi_path = MIDI_DIR / filename

        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            manifest["failed"].append({"key": key, "reason": str(e)})
            logger.warning("Parse error for %s: %s", filename, e)
            continue

        note_count = count_midi_notes(midi)

        # Update histogram
        bucket = _bucket_note_count(note_count)
        manifest["note_count_buckets"][bucket] = (
            manifest["note_count_buckets"].get(bucket, 0) + 1
        )

        # Skip if too many or zero notes
        if note_count > MAX_NOTES:
            manifest["skipped"].append({
                "key": key, "reason": f"too many notes ({note_count})",
            })
            continue

        if note_count == 0:
            manifest["skipped"].append({"key": key, "reason": "no notes"})
            continue

        # Build graph
        try:
            graph = parsed_midi_to_graph(midi)
        except Exception as e:
            manifest["failed"].append({"key": key, "reason": str(e)})
            logger.warning("Graph build error for %s: %s", filename, e)
            continue

        shard_buffer[key] = graph
        manifest["processed_keys"].append(key)
        processed_this_run += 1

        # Flush shard if buffer full or RSS too high
        rss_gb = _get_rss_gb()
        if len(shard_buffer) >= SHARD_SIZE or rss_gb > RSS_FLUSH_THRESHOLD_GB:
            if rss_gb > RSS_FLUSH_THRESHOLD_GB:
                logger.warning(
                    "RSS=%.1f GB exceeds threshold, force-flushing shard", rss_gb
                )
            _save_shard(next_shard_id, shard_buffer)
            manifest["shard_range"][1] = next_shard_id + 1
            _save_manifest(manifest)
            next_shard_id += 1
            shard_buffer = {}
            del graph, midi
            gc.collect()

        # Progress log every 30 seconds
        now = time.time()
        if now - t_last_log > 30:
            elapsed = now - t_start
            rate = processed_this_run / elapsed if elapsed > 0 else 0
            eta_remaining = (len(remaining) - i - 1) / rate if rate > 0 else 0
            logger.info(
                "  [%d/%d] %d processed, RSS=%.1f GB, "
                "%.1f files/s, ETA %.0f min",
                i + 1, len(remaining), processed_this_run,
                _get_rss_gb(), rate, eta_remaining / 60,
            )
            t_last_log = now

    # Flush remaining buffer
    if shard_buffer:
        _save_shard(next_shard_id, shard_buffer)
        manifest["shard_range"][1] = next_shard_id + 1
        _save_manifest(manifest)
        del shard_buffer
        gc.collect()

    # Finalize manifest
    elapsed = time.time() - t_start
    manifest["total_processing_time_seconds"] += elapsed

    _save_manifest(manifest)

    # Summary
    logger.info("\n=== GIANTMIDI Processing Complete ===")
    logger.info("Processed: %d", len(manifest["processed_keys"]))
    logger.info("Skipped: %d", len(manifest["skipped"]))
    logger.info("Failed: %d", len(manifest["failed"]))
    logger.info("Shards: %04d - %04d", FIRST_SHARD_ID, manifest["shard_range"][1] - 1)
    logger.info("Note count distribution: %s", json.dumps(manifest["note_count_buckets"]))
    logger.info("Time: %.0f seconds (%.1f min)", elapsed, elapsed / 60)
    logger.info("Manifest: %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify psutil is available**

Run: `cd model && uv run python -c "import psutil; print(psutil.Process().memory_info().rss / 1024**3)"`
Expected: Prints a number (RSS in GB). If `ModuleNotFoundError`, run `uv pip install psutil`.

- [ ] **Step 3: Dry-run on 5 files to verify the pipeline works**

Run a quick test by temporarily setting `remaining = remaining[:5]` or just running and interrupting after a few files:

```bash
cd model && uv run python -c "
from scripts.process_giantmidi_graphs import _extract_youtube_id, _get_rss_gb
import pretty_midi
from pathlib import Path
from model_improvement.graph import count_midi_notes, parsed_midi_to_graph

midi_dir = Path('data/giantmidi_raw/GiantMIDI-PIano/midis')
files = sorted(f.name for f in midi_dir.iterdir() if f.suffix == '.mid')[:5]

for f in files:
    yt_id = _extract_youtube_id(f)
    midi = pretty_midi.PrettyMIDI(str(midi_dir / f))
    nc = count_midi_notes(midi)
    print(f'{yt_id}: {nc} notes', end='')
    if nc > 0 and nc <= 5000:
        g = parsed_midi_to_graph(midi)
        print(f' -> {g.x.shape[0]} nodes, {g.edge_index.shape[1]} edges')
    else:
        print(' -> SKIP')
print(f'RSS: {_get_rss_gb():.2f} GB')
"
```

Expected: 5 files processed with node/edge counts printed, RSS under 1 GB.

- [ ] **Step 4: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai
git add model/scripts/process_giantmidi_graphs.py
git commit -m "add GIANTMIDI graph processing script

Memory-safe conversion of 10,854 GIANTMIDI MIDI files into PyG graph
shards for S2 pretraining. Features: single MIDI parse, 5K note cap,
50-graph shards, psutil RSS monitoring (20 GB flush threshold),
YouTube ID source keys, resumable via manifest."
```

---

## Chunk 3: Run processing and verify

### Task 3: Run the full processing pipeline

**Files:**
- Run: `model/scripts/process_giantmidi_graphs.py`
- Verify: `model/data/pretrain_cache/graphs/shards/graphs_0075.pt` and subsequent

- [ ] **Step 1: Run the full processing script**

Run: `cd model && uv run python scripts/process_giantmidi_graphs.py`

This will take 2-3 hours. Monitor the logs for:
- RSS staying under 22 GB
- Shard saves happening every 50 files
- Processing rate (~1-2 files/s expected)
- Note count distribution in final summary

Expected output at the end:
```
=== GIANTMIDI Processing Complete ===
Processed: ~8500-9000
Skipped: ~1500-2000
Failed: <100
Shards: 0075 - ~0245
```

- [ ] **Step 2: Verify shard integrity**

After processing completes, verify the output:

```bash
cd model && uv run python -c "
import torch
from pathlib import Path

shard_dir = Path('data/pretrain_cache/graphs/shards')

# Check new shards exist
new_shards = sorted(p for p in shard_dir.glob('graphs_*.pt') if int(p.stem.split('_')[1]) >= 75)
print(f'New shards: {len(new_shards)}')

# Verify first and last shard
for path in [new_shards[0], new_shards[-1]]:
    shard = torch.load(path, map_location='cpu', weights_only=False)
    print(f'{path.name}: {len(shard)} graphs')
    for key, data in list(shard.items())[:2]:
        print(f'  {key}: x={data.x.shape}, edges={data.edge_index.shape}, edge_type={data.edge_type.shape}')
        assert data.x.shape[1] == 6, 'Node features should be 6-dim'
        assert data.edge_index.shape[0] == 2, 'edge_index should be [2, E]'
        assert data.edge_type.shape[0] == data.edge_index.shape[1], 'edge_type length should match edges'
    del shard
print('All checks passed')
"
```

Expected: New shards load, each graph has `x=(N, 6)`, `edge_index=(2, E)`, `edge_type=(E,)`.

- [ ] **Step 3: Verify existing shards untouched**

```bash
cd model && uv run python -c "
import torch
from pathlib import Path

# Check first existing shard is unchanged
shard = torch.load('data/pretrain_cache/graphs/shards/graphs_0000.pt', map_location='cpu', weights_only=False)
print(f'Shard 0000: {len(shard)} graphs (should be ~200)')
key = list(shard.keys())[0]
print(f'  First key: {key}')
assert not key.startswith('giantmidi/'), 'Existing shards should not contain GIANTMIDI keys'
print('Existing shards untouched')
"
```

- [ ] **Step 4: Verify ShardedScoreGraphPretrainDataset sees all shards**

```bash
cd model && uv run python -c "
import torch
from pathlib import Path

shard_dir = Path('data/pretrain_cache/graphs/shards')
all_shards = sorted(shard_dir.glob('graphs_*.pt'))
print(f'Total shards: {len(all_shards)}')

# Build shard index (same as ShardedScoreGraphPretrainDataset)
shard_index = {}
total_keys = 0
for path in all_shards:
    shard = torch.load(path, map_location='cpu', weights_only=False)
    for key in shard.keys():
        shard_index[key] = str(path)
    total_keys += len(shard)
    del shard

giantmidi_keys = [k for k in shard_index if k.startswith('giantmidi/')]
other_keys = [k for k in shard_index if not k.startswith('giantmidi/')]
print(f'Total keys: {total_keys}')
print(f'  GIANTMIDI: {len(giantmidi_keys)}')
print(f'  Existing: {len(other_keys)}')
print(f'  Combined: {len(giantmidi_keys) + len(other_keys)} (target: ~23-24K)')
"
```

Expected: ~14.8K existing + ~8.5-9K GIANTMIDI = ~23-24K total keys.

- [ ] **Step 5: Check disk usage**

Run: `du -sh /Users/jdhiman/Documents/crescendai/model/data/pretrain_cache/graphs/ && df -h /Users/jdhiman/Documents/crescendai/model/data/`

Expected: ~16-18 GB total for graphs dir (was 10 GB, +6-8 GB new). 69+ GB free.

- [ ] **Step 6: Commit manifest and any script fixes**

```bash
cd /Users/jdhiman/Documents/crescendai
git add model/scripts/process_giantmidi_graphs.py
git commit -m "fix: any script fixes discovered during processing run"
```

(Only commit if there were script changes. The manifest and data shards are gitignored.)
