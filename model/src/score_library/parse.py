"""MIDI parser that converts score MIDIs into bar-centric JSON."""

from __future__ import annotations

import bisect
from pathlib import Path

import mido

from score_library.schema import Bar, PedalEvent, ScoreData, ScoreNote

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_DEFAULT_TEMPO = 500_000  # 120 BPM in microseconds per beat


def _pitch_name(midi_note: int) -> str:
    """Convert a MIDI note number to a pitch name, e.g. 60 -> 'C4'."""
    octave = (midi_note // 12) - 1
    name = _NOTE_NAMES[midi_note % 12]
    return f"{name}{octave}"


def build_bar_grid(
    time_sigs: list[dict],
    total_ticks: int,
    ticks_per_beat: int,
) -> list[dict]:
    """Build a list of bar dicts from time signature events.

    Each bar dict has: bar_number, start_tick, time_signature.
    """
    if not time_sigs:
        time_sigs = [{"tick": 0, "numerator": 4, "denominator": 4}]

    bars: list[dict] = []
    bar_number = 1

    for i, ts in enumerate(time_sigs):
        numerator = ts["numerator"]
        denominator = ts["denominator"]
        ticks_per_bar = int(numerator * ticks_per_beat * 4 / denominator)
        ts_str = f"{numerator}/{denominator}"

        start_tick = ts["tick"]
        # End tick is the start of the next time signature, or total_ticks.
        if i + 1 < len(time_sigs):
            end_tick = time_sigs[i + 1]["tick"]
        else:
            end_tick = total_ticks

        tick = start_tick
        while tick < end_tick:
            bars.append(
                {
                    "bar_number": bar_number,
                    "start_tick": tick,
                    "time_signature": ts_str,
                }
            )
            bar_number += 1
            tick += ticks_per_bar

    return bars


def ticks_to_seconds(
    tick: int,
    tempo_map: list[dict],
    ticks_per_beat: int,
) -> float:
    """Convert an absolute tick position to seconds using the tempo map."""
    if not tempo_map:
        tempo_map = [{"tick": 0, "tempo": _DEFAULT_TEMPO}]

    seconds = 0.0
    prev_tick = 0
    current_tempo = _DEFAULT_TEMPO

    for tm in tempo_map:
        if tm["tick"] >= tick:
            break
        # Accumulate time from prev_tick to this tempo change.
        delta_ticks = tm["tick"] - prev_tick
        seconds += delta_ticks * current_tempo / (ticks_per_beat * 1_000_000)
        prev_tick = tm["tick"]
        current_tempo = tm["tempo"]

    # Accumulate remaining ticks at current tempo.
    delta_ticks = tick - prev_tick
    seconds += delta_ticks * current_tempo / (ticks_per_beat * 1_000_000)

    return seconds


def assign_notes_to_bars(
    notes: list[dict],
    bar_grid: list[dict],
) -> dict[int, list[dict]]:
    """Assign notes to bars based on their onset tick.

    Returns a dict mapping bar_number -> list of notes.
    """
    result: dict[int, list[dict]] = {bar["bar_number"]: [] for bar in bar_grid}

    bar_starts = [bar["start_tick"] for bar in bar_grid]

    for note in notes:
        idx = bisect.bisect_right(bar_starts, note["onset_tick"]) - 1
        if idx < 0:
            # Note before first bar -- assign to bar 1 if it exists.
            if bar_grid:
                result[bar_grid[0]["bar_number"]].append(note)
        else:
            bar_num = bar_grid[idx]["bar_number"]
            result[bar_num].append(note)

    return result


def _assign_pedal_to_bars(
    pedal_events: list[dict],
    bar_grid: list[dict],
) -> dict[int, list[dict]]:
    """Assign pedal events to bars based on their tick."""
    result: dict[int, list[dict]] = {bar["bar_number"]: [] for bar in bar_grid}

    bar_starts = [bar["start_tick"] for bar in bar_grid]

    for event in pedal_events:
        idx = bisect.bisect_right(bar_starts, event["tick"]) - 1
        if idx < 0:
            if bar_grid:
                result[bar_grid[0]["bar_number"]].append(event)
        else:
            bar_num = bar_grid[idx]["bar_number"]
            result[bar_num].append(event)

    return result


def parse_score_midi(
    midi_path: str | Path,
    piece_id: str,
    composer: str,
    title: str,
) -> ScoreData:
    """Parse a score MIDI file into bar-centric ScoreData.

    Full pipeline: read MIDI, extract events, build bar grid, assign notes/pedal
    to bars, convert ticks to seconds, compute per-bar summaries.
    """
    midi = mido.MidiFile(str(midi_path))
    ticks_per_beat = midi.ticks_per_beat

    # -- Extract events from all tracks --
    time_sigs: list[dict] = []
    key_sigs: list[dict] = []
    tempo_map: list[dict] = []
    raw_notes: list[dict] = []  # note_on events waiting for note_off
    pending: dict[tuple[int, int], dict] = {}  # (track, pitch) -> note_on dict
    pedal_events: list[dict] = []
    total_ticks = 0

    for track_idx, track in enumerate(midi.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time

            if msg.type == "time_signature":
                time_sigs.append(
                    {
                        "tick": abs_tick,
                        "numerator": msg.numerator,
                        "denominator": msg.denominator,
                    }
                )
            elif msg.type == "key_signature":
                key_sigs.append({"tick": abs_tick, "key": msg.key})
            elif msg.type == "set_tempo":
                tempo_map.append({"tick": abs_tick, "tempo": msg.tempo})
            elif msg.type == "note_on" and msg.velocity > 0:
                pending[(track_idx, msg.note)] = {
                    "pitch": msg.note,
                    "velocity": msg.velocity,
                    "onset_tick": abs_tick,
                    "track": track_idx,
                }
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                key = (track_idx, msg.note)
                if key in pending:
                    note_dict = pending.pop(key)
                    note_dict["duration_ticks"] = abs_tick - note_dict["onset_tick"]
                    raw_notes.append(note_dict)
            elif msg.type == "control_change" and msg.control == 64:
                pedal_events.append(
                    {
                        "type": "on" if msg.value >= 64 else "off",
                        "tick": abs_tick,
                    }
                )

        total_ticks = max(total_ticks, abs_tick)

    # Sort time sigs and tempo map by tick.
    time_sigs.sort(key=lambda x: x["tick"])
    tempo_map.sort(key=lambda x: x["tick"])

    # Deduplicate time sigs at same tick (keep last).
    if time_sigs:
        deduped_ts: list[dict] = [time_sigs[0]]
        for ts in time_sigs[1:]:
            if ts["tick"] == deduped_ts[-1]["tick"]:
                deduped_ts[-1] = ts
            else:
                deduped_ts.append(ts)
        time_sigs = deduped_ts

    # -- Build bar grid --
    bar_grid = build_bar_grid(time_sigs, total_ticks, ticks_per_beat)

    # -- Assign notes and pedal to bars --
    notes_by_bar = assign_notes_to_bars(raw_notes, bar_grid)
    pedal_by_bar = _assign_pedal_to_bars(pedal_events, bar_grid)

    # -- Build Bar models --
    bars: list[Bar] = []
    for bar_dict in bar_grid:
        bn = bar_dict["bar_number"]
        bar_notes = notes_by_bar.get(bn, [])
        bar_pedal = pedal_by_bar.get(bn, [])

        # Convert notes to ScoreNote with seconds.
        score_notes: list[ScoreNote] = []
        for n in sorted(bar_notes, key=lambda x: (x["onset_tick"], x["pitch"])):
            onset_sec = ticks_to_seconds(n["onset_tick"], tempo_map, ticks_per_beat)
            dur_sec = ticks_to_seconds(
                n["onset_tick"] + n["duration_ticks"], tempo_map, ticks_per_beat
            ) - onset_sec
            score_notes.append(
                ScoreNote(
                    pitch=n["pitch"],
                    pitch_name=_pitch_name(n["pitch"]),
                    velocity=n["velocity"],
                    onset_tick=n["onset_tick"],
                    onset_seconds=round(onset_sec, 6),
                    duration_ticks=n["duration_ticks"],
                    duration_seconds=round(dur_sec, 6),
                    track=n["track"],
                )
            )

        # Convert pedal events to PedalEvent with seconds.
        score_pedal: list[PedalEvent] = []
        for p in sorted(bar_pedal, key=lambda x: x["tick"]):
            p_sec = ticks_to_seconds(p["tick"], tempo_map, ticks_per_beat)
            score_pedal.append(
                PedalEvent(
                    type=p["type"],
                    tick=p["tick"],
                    seconds=round(p_sec, 6),
                )
            )

        # Per-bar summaries.
        pitches = [n.pitch for n in score_notes]
        velocities = [n.velocity for n in score_notes]

        bars.append(
            Bar(
                bar_number=bn,
                start_tick=bar_dict["start_tick"],
                start_seconds=round(
                    ticks_to_seconds(
                        bar_dict["start_tick"], tempo_map, ticks_per_beat
                    ),
                    6,
                ),
                time_signature=bar_dict["time_signature"],
                notes=score_notes,
                pedal_events=score_pedal,
                note_count=len(score_notes),
                pitch_range=[min(pitches), max(pitches)] if pitches else [],
                mean_velocity=round(sum(velocities) / len(velocities))
                if velocities
                else 0,
            )
        )

    # -- Key signature: use the first one if available --
    key_signature = key_sigs[0]["key"] if key_sigs else None

    # -- Build tempo markings with BPM for output --
    tempo_markings_out = []
    for tm in tempo_map:
        bpm = round(60_000_000 / tm["tempo"], 1) if tm["tempo"] > 0 else 0
        tempo_markings_out.append(
            {"tick": tm["tick"], "tempo_usec": tm["tempo"], "bpm": bpm}
        )

    # -- Time signatures for output --
    time_sigs_out = [
        {
            "tick": ts["tick"],
            "numerator": ts["numerator"],
            "denominator": ts["denominator"],
        }
        for ts in time_sigs
    ]

    return ScoreData(
        piece_id=piece_id,
        composer=composer,
        title=title,
        key_signature=key_signature,
        time_signatures=time_sigs_out,
        tempo_markings=tempo_markings_out,
        total_bars=len(bars),
        bars=bars,
    )
