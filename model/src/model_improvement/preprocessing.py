"""Batch preprocessing pipeline for multi-dataset symbolic encoder pretraining.

Tokenizes MIDI files, builds score graphs, and extracts continuous features
using existing infrastructure from model_improvement.tokenizer and model_improvement.graph.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch

from model_improvement.datasets import MIDIFileEntry

logger = logging.getLogger(__name__)

CHECKPOINT_INTERVAL = 500


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

    if failures:
        print(f"Tokenization failures ({len(failures)}/{len(entries)}):")
        for path, err in failures[:10]:
            print(f"  {path}: {err}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    print(f"Tokenized {len(result)} / {len(entries)} MIDI files -> {output_path}")
    return result


def preprocess_graphs(
    entries: list[MIDIFileEntry],
    graphs_output: Path,
    hetero_output: Path,
    timeout_per_file: int = 60,
) -> tuple[dict, dict]:
    """Build score graphs and heterogeneous graphs from MIDI files.

    Saves {source_key: Data} and {source_key: HeteroData} as .pt files.
    Supports resuming from partial checkpoints.

    Args:
        entries: List of MIDIFileEntry objects to process.
        graphs_output: Path for the homogeneous graphs .pt file.
        hetero_output: Path for the heterogeneous graphs .pt file.
        timeout_per_file: Max seconds per file before skipping (edge
            construction is O(n^2) and can hang on very large files).

    Returns:
        Tuple of (graphs_dict, hetero_graphs_dict).
    """
    import signal

    from model_improvement.graph import midi_to_graph, midi_to_hetero_graph

    class _Timeout(Exception):
        pass

    def _timeout_handler(_signum, _frame):
        raise _Timeout()

    graphs_output = Path(graphs_output)
    hetero_output = Path(hetero_output)
    graphs_output.parent.mkdir(parents=True, exist_ok=True)
    hetero_output.parent.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoints
    graphs_ckpt = graphs_output.with_suffix(".pt.partial")
    hetero_ckpt = hetero_output.with_suffix(".pt.partial")

    if graphs_ckpt.exists() and hetero_ckpt.exists():
        graphs = torch.load(graphs_ckpt, map_location="cpu", weights_only=False)
        hetero = torch.load(hetero_ckpt, map_location="cpu", weights_only=False)
        print(f"Resuming graph building from checkpoint ({len(graphs)} entries)")
    else:
        graphs: dict = {}
        hetero: dict = {}

    failures: list[tuple[str, str]] = []
    old_handler = signal.getsignal(signal.SIGALRM)

    for i, entry in enumerate(entries):
        if entry.source_key in graphs:
            continue

        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_per_file)
            g = midi_to_graph(entry.midi_path)
            hg = midi_to_hetero_graph(entry.midi_path)
            signal.alarm(0)
            graphs[entry.source_key] = g
            hetero[entry.source_key] = hg
        except _Timeout:
            signal.alarm(0)
            failures.append((str(entry.midi_path), f"timeout ({timeout_per_file}s)"))
            logger.warning("Timeout building graph for %s", entry.midi_path)
        except Exception as e:
            signal.alarm(0)
            failures.append((str(entry.midi_path), str(e)))
            logger.warning("Failed to build graph for %s: %s", entry.midi_path, e)

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(graphs, graphs_ckpt)
            torch.save(hetero, hetero_ckpt)
            print(f"  Graph checkpoint: {len(graphs)}/{len(entries)}")

    signal.signal(signal.SIGALRM, old_handler)
    torch.save(graphs, graphs_output)
    torch.save(hetero, hetero_output)
    for ckpt in [graphs_ckpt, hetero_ckpt]:
        if ckpt.exists():
            ckpt.unlink()

    if failures:
        print(f"Graph building failures ({len(failures)}/{len(entries)}):")
        for path, err in failures[:10]:
            print(f"  {path}: {err}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    print(f"Built graphs for {len(graphs)} / {len(entries)} MIDI files")
    print(f"  Homogeneous -> {graphs_output}")
    print(f"  Heterogeneous -> {hetero_output}")
    return graphs, hetero


def preprocess_continuous_features(
    entries: list[MIDIFileEntry],
    output_path: Path,
    frame_rate: int = 50,
) -> dict[str, torch.Tensor]:
    """Extract continuous features from MIDI files.

    Saves {source_key: Tensor[T, 5]} as a .pt file.
    Supports resuming from partial checkpoints.

    Args:
        entries: List of MIDIFileEntry objects to process.
        output_path: Path for the output .pt file.
        frame_rate: Frames per second for feature extraction.

    Returns:
        Dict mapping source_key to feature tensor [T, 5].
    """
    from model_improvement.tokenizer import extract_continuous_features

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_path.with_suffix(".pt.partial")
    if checkpoint_path.exists():
        result = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"Resuming feature extraction from checkpoint ({len(result)} entries)")
    else:
        result: dict[str, torch.Tensor] = {}

    failures: list[tuple[str, str]] = []

    for i, entry in enumerate(entries):
        if entry.source_key in result:
            continue

        try:
            features = extract_continuous_features(entry.midi_path, frame_rate=frame_rate)
            result[entry.source_key] = torch.from_numpy(features).float()
        except Exception as e:
            failures.append((str(entry.midi_path), str(e)))
            logger.warning(
                "Failed to extract features for %s: %s", entry.midi_path, e
            )

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(result, checkpoint_path)
            print(f"  Feature extraction checkpoint: {len(result)}/{len(entries)}")

    torch.save(result, output_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    if failures:
        print(f"Feature extraction failures ({len(failures)}/{len(entries)}):")
        for path, err in failures[:10]:
            print(f"  {path}: {err}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    print(f"Extracted features for {len(result)} / {len(entries)} MIDI files -> {output_path}")
    return result


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
