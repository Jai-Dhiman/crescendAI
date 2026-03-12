# GIANTMIDI Graph Processing Design

**Date:** 2026-03-11
**Status:** Approved
**Context:** Track B data preparation (Wave 1, item 2b). Expand S2 GNN pretraining corpus from 14.8K to ~24K graphs by processing the GIANTMIDI-Piano dataset.

## Problem

S2 (GNN symbolic encoder) was pretrained on 14,821 graphs from ASAP, MAESTRO, and ATEPP MIDI sources. GIANTMIDI-Piano provides 10,854 additional MIDI files -- critically, these are AMT-transcribed (ByteDance piano transcription), so pretraining on them makes S2 natively robust to transcription noise from the production AMT pipeline.

The existing `preprocess_graphs()` pipeline has memory concerns for M4 (32 GB RAM):
- Builds both homogeneous and heterogeneous graphs (hetero is unnecessary -- S2H lost)
- Double-parses each MIDI file (once to count notes, once to build graph)
- Merges all shards into a monolithic `all_graphs.pt` (would be ~40+ GB for 25K graphs)
- `_build_edges` has O(n * k) average complexity (k = local polyphony window, typically 5-20), but degrades toward O(n^2) for pathological inputs with sustained overlapping notes. Combined with memory for edge lists, large files (GIANTMIDI has files up to 23K notes) are risky on 32 GB RAM.

## Solution

New script `model/scripts/process_giantmidi_graphs.py` with memory-safe processing:

### Architecture

```
GIANTMIDI MIDI files (10,854 files, 280 MB)
    |
    v
process_giantmidi_graphs.py
    |-- Parse MIDI once with pretty_midi
    |-- Check note count (skip if > 5,000)
    |-- Build homogeneous graph via parsed_midi_to_graph()
    |-- Accumulate in shard buffer (50 graphs per shard)
    |-- Monitor RSS via psutil, force-flush if > 20 GB
    |-- Write shards to pretrain_cache/graphs/shards/graphs_0075.pt, ...
    |-- Save manifest (giantmidi_manifest.json)
    v
ShardedScoreGraphPretrainDataset picks up new shards automatically
```

### Key Design Decisions

**MAX_NOTES = 5,000 (conservative initial cap):** While `_build_edges` is typically O(n * k) with early-break, edge list memory and graph object size grow with note count. On 32 GB RAM, being conservative for the initial 10K-file batch is prudent. Sampling 50 GIANTMIDI files shows:
- Median: 2,254 notes
- 75th percentile: 4,122 notes
- ~20% exceed 5,000 notes, ~8% exceed 10,000
- A 5K cap keeps ~80% of files while staying memory-safe
- Skipped files are logged in the manifest for a second pass. Future options: raise cap to 8K-10K with a per-file timeout (60s) as the safety net, or implement chunked edge building.
- Note: existing `preprocessing.py` uses MAX_NOTES=10,000. The lower cap here is intentional for the initial GIANTMIDI run -- these are AMT-transcribed files with more noise, so the distribution shift from excluding the longest pieces is acceptable.

**Shard size = 50:** Smaller than the existing 200, giving more frequent disk flushes and finer-grained resumption.

**Shard numbering starts at 0075:** Existing shards are 0000-0074. New GIANTMIDI shards continue the sequence so both old and new coexist in the same directory.

**No monolithic merge:** Training uses `ShardedScoreGraphPretrainDataset` which lazy-loads from shards. No need to build `all_graphs.pt`.

**Homogeneous only:** S2H (heterogeneous GNN) lost to S2 (homogeneous). No `homo_to_hetero_graph()` calls.

### Code Changes

#### New file: `model/scripts/process_giantmidi_graphs.py`

Responsibilities:
- Scan `data/giantmidi_raw/GiantMIDI-PIano/midis/` for `.mid` files
- Generate source keys: `giantmidi/{youtube_id}` where youtube_id is the 11-character YouTube ID extracted from the filename (unique by construction). GIANTMIDI filenames follow the pattern `"Composer, Title, YouTubeID.mid"` -- the last comma-separated field before `.mid` is always the YouTube ID.
- For each file: parse MIDI -> count notes -> skip or build graph -> buffer in shard
- Flush shard to disk every 50 graphs or when RSS > 20 GB threshold
- Log RSS, processing rate, and note count distribution throughout
- Save `data/giantmidi_raw/giantmidi_manifest.json` with:
  - processed_keys: list of successfully processed source keys
  - skipped_keys: list of skipped files with reasons (too many notes, no notes, parse error)
  - note_count_histogram: distribution buckets for analysis
  - shard_range: [first_shard_id, last_shard_id]
  - total_processing_time_seconds

Resumability: on restart, load manifest, skip already-processed keys.

Error handling: each file is wrapped in a try/except. GIANTMIDI files are AMT-transcribed from YouTube audio and may have edge cases (zero-length notes, empty tracks, corrupt headers). Errors are logged to manifest with the exception message. Processing continues to the next file.

The manifest and all data artifacts are gitignored (under `model/data/` which is already in `.gitignore`).

#### Modified file: `model/src/model_improvement/graph.py`

Add two functions:
- `count_midi_notes(midi: PrettyMIDI) -> int` -- counts notes from a parsed MIDI object without building a graph. Replaces the parse-from-path pattern in `preprocessing.py:_count_midi_notes`.
- `parsed_midi_to_graph(midi: PrettyMIDI, max_voices: int = 8, follow_tolerance: float = 0.05) -> Data` -- builds a homogeneous graph from an already-parsed PrettyMIDI object. Same output contract as `midi_to_graph()`.

The existing `midi_to_graph(path)` is refactored to: parse MIDI, then call `parsed_midi_to_graph()`. API unchanged.

#### No changes to:
- `preprocessing.py` -- not used for this task
- `data.py` -- `ShardedScoreGraphPretrainDataset` already handles shards
- Training notebooks -- shard index rebuild happens at training time

### Expected Output

- ~8,500-9,000 new graphs (after 5K note cap)
- ~170-180 new shard files (shards 0075+)
- ~6-8 GB additional disk usage (existing 14.8K graphs = 10 GB across 75 shards, ~0.67 MB/graph average; GIANTMIDI graphs skew smaller due to the 5K note cap). 77 GB available.
- Estimated processing time: 2-3 hours on M4

### Verification

1. All shards load correctly: `torch.load(shard_path)` returns dict of PyG Data objects
2. Each graph has expected attributes: `x` (N, 6), `edge_index` (2, E), `edge_type` (E,)
3. Shard count matches manifest
4. RSS never exceeds 22 GB during processing (flush threshold 20 GB + ~2 GB headroom for one shard's worth of graphs)
5. Existing shards (0000-0074) are untouched
6. `ShardedScoreGraphPretrainDataset` can build index over all shards (old + new)
