"""Generate reference performance profiles from MAESTRO recordings.

For each piece in the score library that has MAESTRO recordings:
1. Load the score JSON
2. For each MAESTRO recording of that piece:
   a. Load the performance MIDI
   b. Align performance to score via onset-based DTW (full DTW, not subsequence)
   c. Compute per-bar statistics (velocity, onset deviation, pedal, duration ratio)
3. Aggregate across all recordings
4. Save as references/v1/{piece_id}.json

Usage:
    uv run python -m src.score_library.reference_cache \
        --score-dir data/score_library \
        --maestro-dir data/maestro_cache \
        --output-dir data/reference_profiles
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mido
import numpy as np

# dtw-python exposes dtw() at top-level
from dtw import dtw


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BarStats:
    bar_number: int
    velocity_mean: float = 0.0
    velocity_std: float = 0.0
    onset_deviation_mean_ms: float = 0.0
    onset_deviation_std_ms: float = 0.0
    pedal_duration_mean_beats: float | None = None
    pedal_changes: int | None = None
    note_duration_ratio_mean: float = 1.0
    performer_count: int = 0


@dataclass
class ReferenceProfile:
    piece_id: str
    performer_count: int
    bars: list[BarStats] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MIDI loading
# ---------------------------------------------------------------------------

_DEFAULT_TEMPO = 500_000  # 120 BPM in microseconds per beat


def _build_tempo_map(midi: mido.MidiFile) -> list[dict]:
    """Extract tempo change events as a sorted list of {tick, tempo} dicts."""
    tempo_map: list[dict] = []
    for track in midi.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo_map.append({"tick": abs_tick, "tempo": msg.tempo})
    tempo_map.sort(key=lambda x: x["tick"])
    return tempo_map


def _ticks_to_seconds(tick: int, tempo_map: list[dict], ticks_per_beat: int) -> float:
    """Convert an absolute tick to seconds using a pre-built tempo map."""
    if not tempo_map:
        tempo_map = [{"tick": 0, "tempo": _DEFAULT_TEMPO}]

    seconds = 0.0
    prev_tick = 0
    current_tempo = _DEFAULT_TEMPO

    for tm in tempo_map:
        if tm["tick"] >= tick:
            break
        delta = tm["tick"] - prev_tick
        seconds += delta * current_tempo / (ticks_per_beat * 1_000_000)
        prev_tick = tm["tick"]
        current_tempo = tm["tempo"]

    delta = tick - prev_tick
    seconds += delta * current_tempo / (ticks_per_beat * 1_000_000)
    return seconds


def load_performance_midi(midi_path: str | Path) -> list[dict]:
    """Load a performance MIDI and return a list of note dicts.

    Each dict has keys:
        pitch, velocity, onset_s, offset_s, duration_s
    """
    midi = mido.MidiFile(str(midi_path))
    ticks_per_beat = midi.ticks_per_beat
    tempo_map = _build_tempo_map(midi)

    pending: dict[tuple[int, int], dict] = {}  # (track, pitch) -> note_on
    notes: list[dict] = []

    for track_idx, track in enumerate(midi.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                key = (track_idx, msg.note)
                onset_s = _ticks_to_seconds(abs_tick, tempo_map, ticks_per_beat)
                pending[key] = {
                    "pitch": msg.note,
                    "velocity": msg.velocity,
                    "onset_s": onset_s,
                    "onset_tick": abs_tick,
                }
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                key = (track_idx, msg.note)
                if key in pending:
                    note = pending.pop(key)
                    offset_s = _ticks_to_seconds(abs_tick, tempo_map, ticks_per_beat)
                    note["offset_s"] = offset_s
                    note["duration_s"] = offset_s - note["onset_s"]
                    notes.append(note)

    notes.sort(key=lambda n: n["onset_s"])
    return notes


def _extract_pedal_events(midi_path: str | Path) -> list[dict]:
    """Extract sustain pedal CC64 events with timestamps in seconds."""
    midi = mido.MidiFile(str(midi_path))
    ticks_per_beat = midi.ticks_per_beat
    tempo_map = _build_tempo_map(midi)

    events: list[dict] = []
    for track in midi.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "control_change" and msg.control == 64:
                events.append(
                    {
                        "type": "on" if msg.value >= 64 else "off",
                        "seconds": _ticks_to_seconds(abs_tick, tempo_map, ticks_per_beat),
                    }
                )

    events.sort(key=lambda e: e["seconds"])
    return events


# ---------------------------------------------------------------------------
# DTW alignment
# ---------------------------------------------------------------------------


def align_to_score(
    perf_notes: list[dict],
    score_data: dict,
) -> tuple[dict[int, list[dict]], float]:
    """Align performance notes to score bars via onset-based full DTW.

    Uses 1D arrays of note onset times (in seconds) for both the performance
    and the score, runs full DTW, then maps performance notes to score bars
    using the warping path.

    Returns a tuple of:
        - dict mapping bar_number -> list of performance note dicts, each
          augmented with score_onset_s, deviation_ms, score_duration_s
        - normalized DTW distance (total cost / path length)
    """
    if not perf_notes:
        return {}, 0.0

    # Collect all score notes across bars, sorted by onset.
    score_notes: list[dict] = []
    for bar in score_data.get("bars", []):
        for note in bar.get("notes", []):
            score_notes.append(
                {
                    "pitch": note["pitch"],
                    "onset_s": note["onset_seconds"],
                    "duration_s": note["duration_seconds"],
                    "bar_number": bar["bar_number"],
                }
            )
    score_notes.sort(key=lambda n: n["onset_s"])

    if not score_notes:
        return {}, 0.0

    # Build 1D onset arrays for DTW.
    perf_onsets = np.array([n["onset_s"] for n in perf_notes], dtype=np.float64)
    score_onsets = np.array([n["onset_s"] for n in score_notes], dtype=np.float64)

    # Normalize to [0, 1] to make distance metric scale-invariant.
    perf_max = perf_onsets[-1] if perf_onsets[-1] > 0 else 1.0
    score_max = score_onsets[-1] if score_onsets[-1] > 0 else 1.0
    perf_norm = perf_onsets / perf_max
    score_norm = score_onsets / score_max

    # Full DTW: query=performance, reference=score.
    alignment = dtw(
        perf_norm.reshape(-1, 1),
        score_norm.reshape(-1, 1),
        keep_internals=True,
    )

    normalized_distance = float(alignment.normalizedDistance)

    # alignment.index1 and alignment.index2 are the warping path indices
    # (performance index -> score index).
    path_perf = alignment.index1
    path_score = alignment.index2

    # Map each performance note to the aligned score note.
    # A performance note may align to multiple score positions -- use the
    # first occurrence in the warping path.
    perf_to_score: dict[int, int] = {}
    for pi, si in zip(path_perf, path_score):
        if pi not in perf_to_score:
            perf_to_score[pi] = si

    # Group performance notes by score bar.
    bar_to_perf: dict[int, list[dict]] = {}
    for perf_idx, perf_note in enumerate(perf_notes):
        score_idx = perf_to_score.get(perf_idx)
        if score_idx is None:
            continue
        matched_score_note = score_notes[score_idx]
        bar_num = matched_score_note["bar_number"]

        augmented = dict(perf_note)
        augmented["score_onset_s"] = matched_score_note["onset_s"]
        augmented["deviation_ms"] = (
            perf_note["onset_s"] - matched_score_note["onset_s"]
        ) * 1000.0
        augmented["score_duration_s"] = matched_score_note["duration_s"]

        if bar_num not in bar_to_perf:
            bar_to_perf[bar_num] = []
        bar_to_perf[bar_num].append(augmented)

    return bar_to_perf, normalized_distance


# ---------------------------------------------------------------------------
# Per-bar statistics
# ---------------------------------------------------------------------------


def _compute_pedal_stats(
    pedal_events: list[dict],
    bar_start_s: float,
    bar_end_s: float,
    beats_per_bar: float,
) -> tuple[float | None, int | None]:
    """Compute pedal hold duration (in beats) and change count for one bar."""
    if not pedal_events:
        return None, None

    # Filter events in this bar's time window.
    bar_events = [
        e for e in pedal_events if bar_start_s <= e["seconds"] < bar_end_s
    ]
    if not bar_events:
        return None, None

    pedal_changes = len(bar_events)

    # Compute total pedal-on duration within the bar.
    total_on_s = 0.0
    is_on = False
    pedal_on_start = 0.0

    for ev in bar_events:
        if ev["type"] == "on" and not is_on:
            is_on = True
            pedal_on_start = ev["seconds"]
        elif ev["type"] == "off" and is_on:
            is_on = False
            total_on_s += ev["seconds"] - pedal_on_start

    # If pedal is still on at end of bar.
    if is_on:
        total_on_s += bar_end_s - pedal_on_start

    bar_duration_s = bar_end_s - bar_start_s
    if bar_duration_s <= 0 or beats_per_bar <= 0:
        return None, pedal_changes

    # Convert to beats: fraction of bar held * beats_per_bar.
    pedal_beats = (total_on_s / bar_duration_s) * beats_per_bar
    return pedal_beats, pedal_changes


def compute_bar_stats(
    bar_num: int,
    perf_notes: list[dict],
    score_bar: dict,
    pedal_events: list[dict],
) -> BarStats:
    """Compute BarStats for a single bar from one performance.

    perf_notes: performance notes assigned to this bar (from align_to_score),
                each augmented with deviation_ms, score_duration_s.
    score_bar: the Bar dict from score JSON for this bar number.
    pedal_events: all pedal events for the full performance (filtered internally).
    """
    stats = BarStats(bar_number=bar_num, performer_count=1)

    if not perf_notes:
        return stats

    velocities = [n["velocity"] for n in perf_notes]
    stats.velocity_mean = statistics.mean(velocities)
    stats.velocity_std = statistics.stdev(velocities) if len(velocities) > 1 else 0.0

    deviations = [n["deviation_ms"] for n in perf_notes]
    stats.onset_deviation_mean_ms = statistics.mean(deviations)
    stats.onset_deviation_std_ms = (
        statistics.stdev(deviations) if len(deviations) > 1 else 0.0
    )

    # Duration ratio: actual / score duration (skip notes where score_duration_s is 0).
    duration_ratios = []
    for n in perf_notes:
        score_dur = n.get("score_duration_s", 0.0)
        perf_dur = n.get("duration_s", 0.0)
        if score_dur > 0:
            duration_ratios.append(perf_dur / score_dur)
    if duration_ratios:
        stats.note_duration_ratio_mean = statistics.mean(duration_ratios)

    # Pedal stats -- need bar start/end seconds.
    bar_start_s = score_bar.get("start_seconds", 0.0)
    # Estimate bar end from start_seconds of the note with latest onset, or
    # derive from time signature.
    ts_str = score_bar.get("time_signature", "4/4")
    try:
        num, denom = ts_str.split("/")
        beats_per_bar = float(num)
    except ValueError:
        beats_per_bar = 4.0

    # Estimate bar duration from score note data -- use the latest note offset.
    score_notes = score_bar.get("notes", [])
    if score_notes:
        bar_end_s = max(
            n["onset_seconds"] + n["duration_seconds"] for n in score_notes
        )
        bar_end_s = max(bar_end_s, bar_start_s + 0.1)
    else:
        bar_end_s = bar_start_s + 1.0

    pedal_duration, pedal_changes = _compute_pedal_stats(
        pedal_events, bar_start_s, bar_end_s, beats_per_bar
    )
    stats.pedal_duration_mean_beats = pedal_duration
    stats.pedal_changes = pedal_changes

    return stats


# ---------------------------------------------------------------------------
# Aggregation across performers
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_bar_stats(all_stats: list[list[BarStats]]) -> list[BarStats]:
    """Average per-bar statistics across multiple performers.

    all_stats: list of per-performance bar lists.
    Returns one BarStats per bar, averaged across performances.
    """
    if not all_stats:
        return []

    # Collect all bar numbers seen.
    bar_numbers: set[int] = set()
    for perf_bars in all_stats:
        for bs in perf_bars:
            bar_numbers.add(bs.bar_number)

    # Index by bar number for quick lookup.
    by_bar: dict[int, list[BarStats]] = {bn: [] for bn in bar_numbers}
    for perf_bars in all_stats:
        for bs in perf_bars:
            by_bar[bs.bar_number].append(bs)

    aggregated: list[BarStats] = []
    for bar_num in sorted(bar_numbers):
        entries = by_bar[bar_num]
        if not entries:
            continue

        performer_count = len(entries)

        velocity_means = [e.velocity_mean for e in entries if e.performer_count > 0]
        velocity_stds = [e.velocity_std for e in entries if e.performer_count > 0]
        dev_means = [e.onset_deviation_mean_ms for e in entries if e.performer_count > 0]
        dev_stds = [e.onset_deviation_std_ms for e in entries if e.performer_count > 0]
        dur_ratios = [
            e.note_duration_ratio_mean for e in entries if e.performer_count > 0
        ]

        pedal_durations = [
            e.pedal_duration_mean_beats
            for e in entries
            if e.pedal_duration_mean_beats is not None
        ]
        pedal_changes_list = [
            e.pedal_changes for e in entries if e.pedal_changes is not None
        ]

        aggregated.append(
            BarStats(
                bar_number=bar_num,
                velocity_mean=_safe_mean(velocity_means),
                velocity_std=_safe_mean(velocity_stds),
                onset_deviation_mean_ms=_safe_mean(dev_means),
                onset_deviation_std_ms=_safe_mean(dev_stds),
                note_duration_ratio_mean=_safe_mean(dur_ratios) if dur_ratios else 1.0,
                pedal_duration_mean_beats=(
                    _safe_mean(pedal_durations) if pedal_durations else None
                ),
                pedal_changes=(
                    round(_safe_mean(pedal_changes_list))
                    if pedal_changes_list
                    else None
                ),
                performer_count=performer_count,
            )
        )

    return aggregated


# ---------------------------------------------------------------------------
# Piece-level orchestration
# ---------------------------------------------------------------------------


def load_score(path: str | Path) -> dict:
    """Load a score JSON file and return it as a plain dict."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_reference_for_piece(
    piece_id: str,
    score_path: Path,
    midi_paths: list[Path],
) -> ReferenceProfile | None:
    """Build a ReferenceProfile for one piece from its MAESTRO MIDI recordings.

    Returns None if no MIDI paths are provided or all fail to process.
    """
    if not midi_paths:
        return None

    print(f"  Loading score: {score_path.name}")
    score_data = load_score(score_path)

    # Build a bar lookup dict: bar_number -> bar dict.
    bar_lookup: dict[int, dict] = {}
    for bar in score_data.get("bars", []):
        bar_lookup[bar["bar_number"]] = bar

    all_bar_stats: list[list[BarStats]] = []

    for midi_path in midi_paths:
        print(f"  Processing: {midi_path.name}")
        try:
            perf_notes = load_performance_midi(midi_path)
            pedal_events = _extract_pedal_events(midi_path)
            bar_to_notes, _ = align_to_score(perf_notes, score_data)

            perf_bar_stats: list[BarStats] = []
            for bar_num, notes in bar_to_notes.items():
                score_bar = bar_lookup.get(bar_num, {})
                bs = compute_bar_stats(bar_num, notes, score_bar, pedal_events)
                perf_bar_stats.append(bs)

            if perf_bar_stats:
                all_bar_stats.append(perf_bar_stats)
        except Exception as exc:
            print(f"    WARNING: failed to process {midi_path.name}: {exc}")

    if not all_bar_stats:
        return None

    aggregated_bars = aggregate_bar_stats(all_bar_stats)
    performer_count = len(all_bar_stats)

    return ReferenceProfile(
        piece_id=piece_id,
        performer_count=performer_count,
        bars=aggregated_bars,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_match(args: argparse.Namespace) -> None:
    """Run the match subcommand."""
    import csv as csv_mod
    from .maestro_matcher import run_match_pipeline

    maestro_csv = Path(args.maestro_csv)
    score_dir = Path(args.score_dir)
    output_path = Path(args.output)

    if not maestro_csv.exists():
        raise FileNotFoundError(f"MAESTRO CSV not found: {maestro_csv}")
    if not score_dir.exists():
        raise FileNotFoundError(f"Score directory not found: {score_dir}")

    # Load titles.json from score_dir
    titles_path = score_dir / "titles.json"
    if not titles_path.exists():
        raise FileNotFoundError(f"titles.json not found in {score_dir}")
    with open(titles_path, encoding="utf-8") as fh:
        titles_map: dict[str, str] = json.load(fh)

    # Extract known ASAP composers from score filenames
    asap_composers = sorted(
        {f.stem.split(".")[0] for f in score_dir.glob("*.json") if f.stem != "titles"}
    )

    print(f"MAESTRO CSV: {maestro_csv}")
    print(f"Score dir: {score_dir} ({len(titles_map)} pieces)")
    print(f"ASAP composers: {', '.join(asap_composers)}")
    print()

    maestro_content = maestro_csv.read_text(encoding="utf-8")
    matches, unmatched = run_match_pipeline(
        maestro_content, titles_map, asap_composers
    )

    # Write matches CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "maestro_composer", "maestro_title", "midi_filename", "duration_s",
        "asap_piece_id", "asap_title", "confidence", "multi_piece", "status",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        # Sort by confidence descending for easier review
        for row in sorted(matches, key=lambda r: float(r["confidence"]), reverse=True):
            writer.writerow(row)

    # Write unmatched CSV
    unmatched_path = output_path.parent / "unmatched_maestro.csv"
    unmatched_fields = ["maestro_composer", "maestro_title", "midi_filename", "reason"]
    with open(unmatched_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=unmatched_fields)
        writer.writeheader()
        for row in unmatched:
            writer.writerow(row)

    # Summary
    unique_pieces = {r["asap_piece_id"] for r in matches}
    print(f"Matched: {len(matches)} recordings -> {len(unique_pieces)} unique pieces")
    print(f"Unmatched: {len(unmatched)} recordings")
    print(f"Output: {output_path}")
    print(f"Unmatched: {unmatched_path}")


def _cmd_generate(args: argparse.Namespace) -> None:
    """Run the generate subcommand."""
    import csv as csv_mod

    matches_path = Path(args.matches)
    maestro_dir = Path(args.maestro_dir)
    score_dir = Path(args.score_dir)
    output_dir = Path(args.output_dir)

    if not matches_path.exists():
        raise FileNotFoundError(f"Matches CSV not found: {matches_path}")
    if not maestro_dir.exists():
        raise FileNotFoundError(f"MAESTRO directory not found: {maestro_dir}")
    if not score_dir.exists():
        raise FileNotFoundError(f"Score directory not found: {score_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read approved matches
    with open(matches_path, encoding="utf-8") as fh:
        reader = csv_mod.DictReader(fh)
        all_rows = list(reader)

    approved = [r for r in all_rows if r.get("status", "").strip().lower() == "approved"]
    if not approved:
        raise ValueError("No approved rows in matches CSV. Review and mark rows as 'approved' first.")

    # Group by piece_id
    by_piece: dict[str, list[str]] = {}
    for row in approved:
        piece_id = row["asap_piece_id"]
        midi_filename = row["midi_filename"]
        if piece_id not in by_piece:
            by_piece[piece_id] = []
        by_piece[piece_id].append(midi_filename)

    print(f"Processing {len(by_piece)} pieces from {len(approved)} approved recordings")
    print()

    # Generation report rows
    report_rows: list[dict] = []
    min_coverage = args.min_coverage

    for piece_id, midi_filenames in sorted(by_piece.items()):
        print(f"[{piece_id}] ({len(midi_filenames)} recording(s))")

        score_path = score_dir / f"{piece_id}.json"
        if not score_path.exists():
            print(f"  Score not found: {score_path} -- skipping")
            report_rows.append({
                "piece_id": piece_id,
                "total_recordings": len(midi_filenames),
                "passed_validation": 0,
                "rejected_coverage": 0,
                "rejected_dtw_cost": 0,
                "performer_count": 0,
                "mean_coverage": "",
                "mean_dtw_cost": "",
                "errors": "score_not_found",
            })
            continue

        score_data = load_score(score_path)
        total_bars = len(score_data.get("bars", []))
        bar_lookup: dict[int, dict] = {}
        for bar in score_data.get("bars", []):
            bar_lookup[bar["bar_number"]] = bar

        all_bar_stats: list[list[BarStats]] = []
        coverages: list[float] = []
        dtw_costs: list[float] = []
        rejected_coverage_count = 0
        errors: list[str] = []

        for midi_filename in midi_filenames:
            midi_path = maestro_dir / midi_filename
            if not midi_path.exists():
                errors.append(f"{midi_filename}: file_not_found")
                print(f"  MIDI not found: {midi_path}")
                continue

            try:
                perf_notes = load_performance_midi(midi_path)
                pedal_events = _extract_pedal_events(midi_path)

                # Single DTW call: align_to_score returns (bar_mapping, dtw_cost)
                bar_to_notes, dtw_cost = align_to_score(perf_notes, score_data)
                dtw_costs.append(dtw_cost)

                # Coverage check
                if total_bars > 0:
                    coverage = len(bar_to_notes) / total_bars
                else:
                    coverage = 0.0
                coverages.append(coverage)

                if coverage < min_coverage:
                    rejected_coverage_count += 1
                    print(f"  {Path(midi_filename).name}: coverage {coverage:.1%} < {min_coverage:.0%} -- rejected")
                    continue

                # Compute per-bar stats
                perf_bar_stats: list[BarStats] = []
                for bar_num, notes in bar_to_notes.items():
                    score_bar = bar_lookup.get(bar_num, {})
                    bs = compute_bar_stats(bar_num, notes, score_bar, pedal_events)
                    perf_bar_stats.append(bs)

                if perf_bar_stats:
                    all_bar_stats.append(perf_bar_stats)
                    print(f"  {Path(midi_filename).name}: coverage {coverage:.1%}, dtw_cost {dtw_cost:.4f} -- OK")

            except Exception as exc:
                errors.append(f"{midi_filename}: {exc}")
                print(f"  {Path(midi_filename).name}: ERROR -- {exc}")

        # Aggregate and write
        performer_count = len(all_bar_stats)
        if performer_count == 0:
            print(f"  No valid recordings for {piece_id}")
        else:
            if performer_count == 1:
                print(f"  WARNING: single-performer reference for {piece_id}")

            aggregated_bars = aggregate_bar_stats(all_bar_stats)

            # Validate non-negative values before serialization
            for bar in aggregated_bars:
                if bar.pedal_changes is not None and bar.pedal_changes < 0:
                    bar.pedal_changes = 0

            profile = ReferenceProfile(
                piece_id=piece_id,
                performer_count=performer_count,
                bars=aggregated_bars,
            )

            out_path = output_dir / f"{piece_id}.json"
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(asdict(profile), fh, indent=2)
            print(f"  Saved: {out_path.name} ({performer_count} performer(s), {len(aggregated_bars)} bars)")

        report_rows.append({
            "piece_id": piece_id,
            "total_recordings": len(midi_filenames),
            "passed_validation": performer_count,
            "rejected_coverage": rejected_coverage_count,
            "rejected_dtw_cost": 0,  # Not enforced on first run
            "performer_count": performer_count,
            "mean_coverage": f"{statistics.mean(coverages):.3f}" if coverages else "",
            "mean_dtw_cost": f"{statistics.mean(dtw_costs):.4f}" if dtw_costs else "",
            "errors": "; ".join(errors) if errors else "",
        })

    # Write generation report
    report_path = matches_path.parent / "generation_report.csv"
    report_fields = [
        "piece_id", "total_recordings", "passed_validation", "rejected_coverage",
        "rejected_dtw_cost", "performer_count", "mean_coverage", "mean_dtw_cost", "errors",
    ]
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=report_fields)
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)

    total_generated = sum(1 for r in report_rows if int(r["performer_count"]) > 0)
    print()
    print(f"Generated {total_generated} reference profiles")
    print(f"Report: {report_path}")


def _cmd_upload(args: argparse.Namespace) -> None:
    """Run the upload subcommand."""
    import subprocess

    source_dir = Path(args.source_dir)
    bucket = args.bucket
    prefix = args.prefix

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {source_dir}")
        return

    print(f"Uploading {len(json_files)} files to {bucket}/{prefix}/")

    uploaded = 0
    for json_file in json_files:
        r2_key = f"{prefix}/{json_file.name}"
        cmd = [
            "wrangler", "r2", "object", "put",
            f"{bucket}/{r2_key}",
            f"--file={json_file}",
            "--content-type=application/json",
        ]
        print(f"  {r2_key}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Upload failed for {json_file.name}: {result.stderr.strip()}"
            )
        uploaded += 1

    print(f"\nUploaded {uploaded} files")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reference cache pipeline: match, generate, upload."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # match
    match_parser = subparsers.add_parser("match", help="Match MAESTRO recordings to ASAP pieces")
    match_parser.add_argument("--maestro-csv", type=str, required=True)
    match_parser.add_argument("--score-dir", type=str, required=True)
    match_parser.add_argument("--output", type=str, required=True)

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate reference profiles from approved matches")
    gen_parser.add_argument("--matches", type=str, required=True)
    gen_parser.add_argument("--maestro-dir", type=str, required=True)
    gen_parser.add_argument("--score-dir", type=str, required=True)
    gen_parser.add_argument("--output-dir", type=str, required=True)
    gen_parser.add_argument("--min-coverage", type=float, default=0.75,
                            help="Minimum bar coverage to accept a recording (default: 0.75)")

    # upload
    upload_parser = subparsers.add_parser("upload", help="Upload reference profiles to R2")
    upload_parser.add_argument("--source-dir", type=str, required=True)
    upload_parser.add_argument("--bucket", type=str, required=True)
    upload_parser.add_argument("--prefix", type=str, required=True)

    args = parser.parse_args()

    if args.command == "match":
        _cmd_match(args)
    elif args.command == "generate":
        _cmd_generate(args)
    elif args.command == "upload":
        _cmd_upload(args)


if __name__ == "__main__":
    main()
