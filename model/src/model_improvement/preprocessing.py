"""Batch preprocessing pipeline for multi-dataset symbolic encoder pretraining.

Tokenizes MIDI files, builds score graphs, and extracts continuous features
using existing infrastructure from model_improvement.tokenizer and model_improvement.graph.

Graph and feature pipelines use shard-based processing to keep memory bounded:
build in chunks, save each chunk to disk, free memory, then merge all shards
into the final .pt files at the end.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import pretty_midi
import torch

from model_improvement.datasets import MIDIFileEntry

logger = logging.getLogger(__name__)

CHECKPOINT_INTERVAL = 500
GRAPH_SHARD_SIZE = 200
FEATURE_SHARD_SIZE = 500
MAX_NOTES = 10_000


def _report_failures(
    stage: str,
    failures: list[tuple[str, str]],
    total: int,
) -> None:
    """Print a summary of processing failures."""
    if not failures:
        return
    print(f"{stage} failures ({len(failures)}/{total}):")
    for path, err in failures[:10]:
        print(f"  {path}: {err}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")


# ---------------------------------------------------------------------------
# Tokenization (unchanged -- token lists are lightweight ~120MB total)
# ---------------------------------------------------------------------------


def preprocess_tokens(
    entries: list[MIDIFileEntry],
    output_path: Path,
    max_seq_len: int = 2048,
) -> dict[str, list[int]]:
    """Tokenize MIDI files using PianoTokenizer.

    Saves {source_key: token_list} as a .pt file. Supports resuming from
    a partial checkpoint.

    Args:
        entries: List of MIDIFileEntry objects to process.
        output_path: Path for the output .pt file.
        max_seq_len: Maximum token sequence length.

    Returns:
        Dict mapping source_key to token list.
    """
    from model_improvement.tokenizer import PianoTokenizer

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If final output already exists, load and return
    if output_path.exists():
        result = torch.load(output_path, map_location="cpu", weights_only=False)
        print(f"Tokens already built: {len(result)} entries at {output_path}")
        return result

    # Resume from checkpoint if available
    checkpoint_path = output_path.with_suffix(".pt.partial")
    if checkpoint_path.exists():
        result = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"Resuming tokenization from checkpoint ({len(result)} entries)")
    else:
        result: dict[str, list[int]] = {}

    tokenizer = PianoTokenizer(max_seq_len=max_seq_len)
    failures: list[tuple[str, str]] = []

    for i, entry in enumerate(entries):
        if entry.source_key in result:
            continue

        try:
            tokens = tokenizer.encode(entry.midi_path)
            result[entry.source_key] = tokens
        except Exception as e:
            failures.append((str(entry.midi_path), str(e)))
            logger.warning("Failed to tokenize %s: %s", entry.midi_path, e)

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(result, checkpoint_path)
            print(f"  Tokenization checkpoint: {len(result)}/{len(entries)}")

    torch.save(result, output_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    _report_failures("Tokenization", failures, len(entries))
    print(f"Tokenized {len(result)} / {len(entries)} MIDI files -> {output_path}")
    return result


# ---------------------------------------------------------------------------
# Graph building (shard-based, memory-safe)
# ---------------------------------------------------------------------------


def _count_midi_notes(midi_path: str) -> int:
    """Count notes in a MIDI file without building a full graph.

    Used as a pre-check to skip files that would cause O(n^2) blow-up
    in edge construction.
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    return sum(len(inst.notes) for inst in midi.instruments if not inst.is_drum)


def _save_graph_shard(
    shard_dir: Path,
    shard_id: int,
    graphs: dict,
    hetero: dict,
) -> None:
    """Save a pair of graph shards to disk."""
    torch.save(graphs, shard_dir / f"graphs_{shard_id:04d}.pt")
    torch.save(hetero, shard_dir / f"hetero_{shard_id:04d}.pt")


def _scan_graph_shards(shard_dir: Path) -> tuple[set[str], int]:
    """Scan existing graph shards and return (processed_keys, next_shard_id)."""
    done_keys: set[str] = set()
    max_id = -1

    for path in sorted(shard_dir.glob("graphs_*.pt")):
        shard_id = int(path.stem.split("_")[1])
        max_id = max(max_id, shard_id)
        shard = torch.load(path, map_location="cpu", weights_only=False)
        done_keys.update(shard.keys())
        del shard

    return done_keys, max_id + 1


def _migrate_legacy_graph_checkpoints(
    graphs_output: Path,
    hetero_output: Path,
    shard_dir: Path,
) -> None:
    """Migrate old .partial checkpoint files into shard format.

    Splits the monolithic checkpoint dicts into GRAPH_SHARD_SIZE-entry
    shards so subsequent runs can resume without loading everything.
    """
    graphs_ckpt = graphs_output.with_suffix(".pt.partial")
    hetero_ckpt = hetero_output.with_suffix(".pt.partial")

    if not (graphs_ckpt.exists() and hetero_ckpt.exists()):
        return

    # Don't re-migrate if shards already exist
    if any(shard_dir.glob("graphs_*.pt")):
        return

    print("Migrating legacy graph checkpoints to shard format...")
    graphs = torch.load(graphs_ckpt, map_location="cpu", weights_only=False)
    hetero = torch.load(hetero_ckpt, map_location="cpu", weights_only=False)

    keys = list(graphs.keys())
    shard_id = 0
    for start in range(0, len(keys), GRAPH_SHARD_SIZE):
        chunk_keys = keys[start : start + GRAPH_SHARD_SIZE]
        g_shard = {k: graphs[k] for k in chunk_keys}
        h_shard = {k: hetero[k] for k in chunk_keys}
        _save_graph_shard(shard_dir, shard_id, g_shard, h_shard)
        shard_id += 1

    del graphs, hetero
    gc.collect()

    graphs_ckpt.rename(graphs_ckpt.with_suffix(".partial.migrated"))
    hetero_ckpt.rename(hetero_ckpt.with_suffix(".partial.migrated"))
    print(f"  Migrated {len(keys)} entries into {shard_id} shards")


def merge_graph_shards(
    shard_dir: Path,
    graphs_output: Path,
    hetero_output: Path,
) -> tuple[dict, dict]:
    """Merge all graph shards into final output files.

    Loads shards one at a time and builds the merged dicts incrementally.

    Args:
        shard_dir: Directory containing graph shard .pt files.
        graphs_output: Path for the final homogeneous graphs .pt file.
        hetero_output: Path for the final heterogeneous graphs .pt file.

    Returns:
        Tuple of (graphs_dict, hetero_graphs_dict).
    """
    graphs: dict = {}
    hetero: dict = {}

    for path in sorted(shard_dir.glob("graphs_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        graphs.update(shard)
        del shard

    for path in sorted(shard_dir.glob("hetero_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        hetero.update(shard)
        del shard

    torch.save(graphs, graphs_output)
    torch.save(hetero, hetero_output)
    print(f"Merged {len(graphs)} graphs, {len(hetero)} hetero -> {graphs_output.parent}")

    return graphs, hetero


def preprocess_graphs(
    entries: list[MIDIFileEntry],
    graphs_output: Path,
    hetero_output: Path,
    max_notes: int = MAX_NOTES,
    shard_size: int = GRAPH_SHARD_SIZE,
) -> tuple[dict, dict]:
    """Build score graphs and heterogeneous graphs from MIDI files.

    Uses shard-based processing to keep memory bounded:

    - Processes entries in the main thread (no multiprocessing fork overhead).
    - Saves a shard every ``shard_size`` entries and frees memory.
    - Pre-checks note count to skip huge MIDI files that would cause
      O(n^2) blow-up in edge construction.
    - Migrates legacy .partial checkpoints into shard format automatically.
    - Merges all shards into final output files at the end.

    Args:
        entries: List of MIDIFileEntry objects to process.
        graphs_output: Path for the final homogeneous graphs .pt file.
        hetero_output: Path for the final heterogeneous graphs .pt file.
        max_notes: Skip MIDI files with more notes than this.
        shard_size: Number of entries per shard file.

    Returns:
        Tuple of (graphs_dict, hetero_graphs_dict).
    """
    from model_improvement.graph import homo_to_hetero_graph, midi_to_graph

    graphs_output = Path(graphs_output)
    hetero_output = Path(hetero_output)

    # If final outputs already exist, load and return
    if graphs_output.exists() and hetero_output.exists():
        graphs = torch.load(graphs_output, map_location="cpu", weights_only=False)
        hetero = torch.load(hetero_output, map_location="cpu", weights_only=False)
        print(f"Graphs already built: {len(graphs)} homo, {len(hetero)} hetero")
        return graphs, hetero

    shard_dir = graphs_output.parent / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Migrate legacy .partial checkpoints
    _migrate_legacy_graph_checkpoints(graphs_output, hetero_output, shard_dir)

    # Find already-processed keys from existing shards
    done_keys, next_shard_id = _scan_graph_shards(shard_dir)
    remaining = [e for e in entries if e.source_key not in done_keys]
    print(f"Graph building: {len(done_keys)} done, {len(remaining)} remaining")

    if not remaining:
        return merge_graph_shards(shard_dir, graphs_output, hetero_output)

    shard_graphs: dict = {}
    shard_hetero: dict = {}
    failures: list[tuple[str, str]] = []

    for i, entry in enumerate(remaining):
        try:
            note_count = _count_midi_notes(str(entry.midi_path))
            if note_count > max_notes:
                failures.append(
                    (str(entry.midi_path), f"too many notes ({note_count})")
                )
                logger.warning(
                    "Skipping %s: %d notes exceeds max %d",
                    entry.midi_path,
                    note_count,
                    max_notes,
                )
                continue
            if note_count == 0:
                failures.append((str(entry.midi_path), "no notes"))
                continue

            g = midi_to_graph(str(entry.midi_path))
            hg = homo_to_hetero_graph(g)
            shard_graphs[entry.source_key] = g
            shard_hetero[entry.source_key] = hg
        except Exception as e:
            failures.append((str(entry.midi_path), str(e)))
            logger.warning("Failed to build graph for %s: %s", entry.midi_path, e)

        # Flush shard to disk and free memory
        if len(shard_graphs) >= shard_size:
            _save_graph_shard(shard_dir, next_shard_id, shard_graphs, shard_hetero)
            done_keys.update(shard_graphs.keys())
            print(
                f"  Shard {next_shard_id} saved "
                f"({len(done_keys)}/{len(entries)} total, "
                f"{i + 1}/{len(remaining)} this run)"
            )
            next_shard_id += 1
            shard_graphs = {}
            shard_hetero = {}
            gc.collect()

    # Save any remaining entries
    if shard_graphs:
        _save_graph_shard(shard_dir, next_shard_id, shard_graphs, shard_hetero)
        done_keys.update(shard_graphs.keys())
        print(
            f"  Shard {next_shard_id} saved "
            f"({len(done_keys)}/{len(entries)} total)"
        )
        del shard_graphs, shard_hetero
        gc.collect()

    _report_failures("Graph building", failures, len(entries))

    # Merge all shards into final output files
    graphs, hetero = merge_graph_shards(shard_dir, graphs_output, hetero_output)
    print(f"Built graphs for {len(graphs)} / {len(entries)} MIDI files")
    print(f"  Homogeneous -> {graphs_output}")
    print(f"  Heterogeneous -> {hetero_output}")
    return graphs, hetero


# ---------------------------------------------------------------------------
# Continuous feature extraction (shard-based, memory-safe)
# ---------------------------------------------------------------------------


def _save_feature_shard(shard_dir: Path, shard_id: int, features: dict) -> None:
    """Save a feature shard to disk."""
    torch.save(features, shard_dir / f"features_{shard_id:04d}.pt")


def _scan_feature_shards(shard_dir: Path) -> tuple[set[str], int]:
    """Scan existing feature shards and return (processed_keys, next_shard_id)."""
    done_keys: set[str] = set()
    max_id = -1

    for path in sorted(shard_dir.glob("features_*.pt")):
        shard_id = int(path.stem.split("_")[1])
        max_id = max(max_id, shard_id)
        shard = torch.load(path, map_location="cpu", weights_only=False)
        done_keys.update(shard.keys())
        del shard

    return done_keys, max_id + 1


def _migrate_legacy_feature_checkpoint(
    output_path: Path,
    shard_dir: Path,
) -> None:
    """Migrate old .partial feature checkpoint into shard format."""
    ckpt = output_path.with_suffix(".pt.partial")
    if not ckpt.exists():
        return
    if any(shard_dir.glob("features_*.pt")):
        return

    print("Migrating legacy feature checkpoint to shard format...")
    result = torch.load(ckpt, map_location="cpu", weights_only=False)
    keys = list(result.keys())
    shard_id = 0
    for start in range(0, len(keys), FEATURE_SHARD_SIZE):
        chunk = {k: result[k] for k in keys[start : start + FEATURE_SHARD_SIZE]}
        _save_feature_shard(shard_dir, shard_id, chunk)
        shard_id += 1

    del result
    gc.collect()
    ckpt.rename(ckpt.with_suffix(".partial.migrated"))
    print(f"  Migrated {len(keys)} entries into {shard_id} shards")


def merge_feature_shards(
    shard_dir: Path,
    output_path: Path,
) -> dict[str, torch.Tensor]:
    """Merge all feature shards into a single output file.

    Args:
        shard_dir: Directory containing feature shard .pt files.
        output_path: Path for the final features .pt file.

    Returns:
        Dict mapping source_key to feature tensor.
    """
    result: dict[str, torch.Tensor] = {}
    for path in sorted(shard_dir.glob("features_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        result.update(shard)
        del shard

    torch.save(result, output_path)
    print(f"Merged {len(result)} feature entries -> {output_path}")
    return result


def preprocess_continuous_features(
    entries: list[MIDIFileEntry],
    output_path: Path,
    frame_rate: int = 50,
    shard_size: int = FEATURE_SHARD_SIZE,
) -> dict[str, torch.Tensor]:
    """Extract continuous features from MIDI files.

    Uses shard-based processing to keep memory bounded.

    Args:
        entries: List of MIDIFileEntry objects to process.
        output_path: Path for the final output .pt file.
        frame_rate: Frames per second for feature extraction.
        shard_size: Number of entries per shard file.

    Returns:
        Dict mapping source_key to feature tensor [T, 5].
    """
    from model_improvement.tokenizer import extract_continuous_features

    output_path = Path(output_path)

    # If final output already exists, load and return
    if output_path.exists():
        result = torch.load(output_path, map_location="cpu", weights_only=False)
        print(f"Features already built: {len(result)} entries at {output_path}")
        return result

    shard_dir = output_path.parent / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Migrate legacy checkpoint
    _migrate_legacy_feature_checkpoint(output_path, shard_dir)

    # Find already-processed keys
    done_keys, next_shard_id = _scan_feature_shards(shard_dir)
    remaining = [e for e in entries if e.source_key not in done_keys]
    print(f"Feature extraction: {len(done_keys)} done, {len(remaining)} remaining")

    if not remaining:
        return merge_feature_shards(shard_dir, output_path)

    shard_features: dict[str, torch.Tensor] = {}
    failures: list[tuple[str, str]] = []

    for i, entry in enumerate(remaining):
        try:
            features = extract_continuous_features(
                entry.midi_path, frame_rate=frame_rate
            )
            shard_features[entry.source_key] = torch.from_numpy(features).float()
        except Exception as e:
            failures.append((str(entry.midi_path), str(e)))
            logger.warning(
                "Failed to extract features for %s: %s", entry.midi_path, e
            )

        # Flush shard to disk and free memory
        if len(shard_features) >= shard_size:
            _save_feature_shard(shard_dir, next_shard_id, shard_features)
            done_keys.update(shard_features.keys())
            print(
                f"  Shard {next_shard_id} saved "
                f"({len(done_keys)}/{len(entries)} total, "
                f"{i + 1}/{len(remaining)} this run)"
            )
            next_shard_id += 1
            shard_features = {}
            gc.collect()

    # Save any remaining entries
    if shard_features:
        _save_feature_shard(shard_dir, next_shard_id, shard_features)
        done_keys.update(shard_features.keys())
        print(
            f"  Shard {next_shard_id} saved "
            f"({len(done_keys)}/{len(entries)} total)"
        )
        del shard_features
        gc.collect()

    _report_failures("Feature extraction", failures, len(entries))

    result = merge_feature_shards(shard_dir, output_path)
    print(
        f"Extracted features for {len(result)} / {len(entries)} "
        f"MIDI files -> {output_path}"
    )
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def preprocess_all(
    entries: list[MIDIFileEntry],
    output_dir: Path,
) -> dict:
    """Run all preprocessing pipelines and save manifest.

    Produces:
        output_dir/tokens/all_tokens.pt
        output_dir/graphs/all_graphs.pt
        output_dir/graphs/all_hetero_graphs.pt
        output_dir/features/all_features.pt
        output_dir/manifest.json

    Args:
        entries: List of MIDIFileEntry objects to process.
        output_dir: Base output directory for all preprocessed data.

    Returns:
        Manifest dict with counts and timing information.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "total_entries": len(entries),
        "sources": {},
    }

    # Count per source
    source_counts: dict[str, int] = {}
    for entry in entries:
        source_counts[entry.source] = source_counts.get(entry.source, 0) + 1
    manifest["sources"] = source_counts

    # Tokenization
    print("\n--- Tokenization ---")
    t0 = time.time()
    tokens = preprocess_tokens(
        entries, output_dir / "tokens" / "all_tokens.pt"
    )
    t_tokens = time.time() - t0
    manifest["tokens"] = {"count": len(tokens), "seconds": round(t_tokens, 1)}

    # Graphs
    print("\n--- Graph Building ---")
    t0 = time.time()
    graphs, hetero = preprocess_graphs(
        entries,
        output_dir / "graphs" / "all_graphs.pt",
        output_dir / "graphs" / "all_hetero_graphs.pt",
    )
    t_graphs = time.time() - t0
    manifest["graphs"] = {"count": len(graphs), "seconds": round(t_graphs, 1)}
    manifest["hetero_graphs"] = {"count": len(hetero), "seconds": round(t_graphs, 1)}

    # Continuous features
    print("\n--- Continuous Feature Extraction ---")
    t0 = time.time()
    features = preprocess_continuous_features(
        entries, output_dir / "features" / "all_features.pt"
    )
    t_features = time.time() - t0
    manifest["features"] = {"count": len(features), "seconds": round(t_features, 1)}

    total_time = t_tokens + t_graphs + t_features
    manifest["total_seconds"] = round(total_time, 1)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nPreprocessing complete in {total_time:.0f}s")
    print(f"Manifest saved to {manifest_path}")
    print(json.dumps(manifest, indent=2))

    return manifest
