"""Transkun-unlocked difficulty features (MIREX Track A, #137).

The 37-feature `candidate_features` superset (phase3c_explore) deliberately EXCLUDES
note offsets, because it was designed against aria-amt, whose offset F1 is 0.37. That
exclusion is why every one of #104's six converging nulls landed near the same ~0.76
wall: they were all measured on the SAME offset-blind view of the music.

Transkun changes the premise. It recovers offsets accurately (offset F1 0.79) and emits
real sustain-pedal CC64 (plus CC67 una corda). That unlocks a feature FAMILY the
difficulty pipeline has never had:

  articulation   duration/IOI ratios -- staccato vs legato touch, the classic
                 articulation measure, impossible without offsets.
  duration       entropy / LZ / dispersion of note lengths -- rhythmic-notation
                 variety that onset IOIs alone cannot see (a whole note and a
                 staccato quarter can share an IOI).
  voicing        TRUE time-weighted concurrency. The existing `polyphony_per_onset`
                 counts notes struck together within 30 ms; it is blind to a held
                 bass sustaining under a running melody. Sustained voice count is
                 a different quantity and a core difficulty driver.
  release        offset dispersion within a chord -- releasing a chord's notes
                 independently (voice-leading) vs together (block chord).
  pedal          CC64 change rate, depth distribution, and HALF-pedalling. Binary
                 on/off pedal is beginner technique; graded continuous values are
                 a late-grade skill, so the value DISTRIBUTION carries more signal
                 than pedal presence.

CONFOUND DISCIPLINE. Piece length already enters the model through `note_count`, and
length correlates with grade. So every feature here is a rate (per second), a fraction,
or a shape statistic (entropy / CV / percentile). No raw counts -- a raw CC64-event
count would look predictive while merely re-measuring duration.

A note is a dict {"pitch","onset","offset","velocity"} as produced by
`psyllabus.notes_from_midi_bytes`. Pedal events are dicts {"time","value"} from
`pedal_from_midi_bytes`. Functions raise loudly on empty input; features that are
genuinely undefined for a degenerate piece are nan (LightGBM handles nan natively),
never a silent zero -- a zero would be indistinguishable from real "no pedal at all".
"""
from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np

from difficulty_features import pitch_lz_complexity

Note = dict

# Log2-duration bin edges (seconds) -> symbolic "note value" alphabet for entropy/LZ.
# Spans demisemiquaver-at-speed (~30 ms) to a long held whole note (~4 s).
DUR_EDGES = np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0])
# Articulation ratio (duration / IOI-to-next-onset) bin edges. <0.5 staccato, ~1 legato,
# >1 overlapping/pedalled.
ARTIC_EDGES = np.array([0.25, 0.5, 0.75, 0.95, 1.1, 1.5, 2.5])
PEDAL_ON = 64          # standard MIDI sustain threshold
CHORD_TOL_S = 0.03     # same onset tolerance the 37-feature set uses for chord clusters


def _entropy(symbols) -> float:
    s = np.asarray(symbols)
    if s.size < 1:
        return float("nan")
    _, counts = np.unique(s, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def pedal_from_midi_bytes(raw: bytes) -> dict[str, list[dict]]:
    """Sustain (CC64) and una-corda (CC67) control changes from raw MIDI bytes.

    `notes_from_midi_bytes` drops control changes entirely, so pedal needs its own
    parse. Returns {"sustain": [...], "soft": [...]} with each event {"time","value"},
    sorted by time. Empty lists mean the transcription contained no pedal at all --
    a real observation about the performance, not a parse failure.
    """
    import pretty_midi  # lazy, matching psyllabus.notes_from_midi_bytes

    pm = pretty_midi.PrettyMIDI(io.BytesIO(raw))
    out: dict[str, list[dict]] = {"sustain": [], "soft": []}
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for cc in inst.control_changes:
            if cc.number == 64:
                out["sustain"].append({"time": float(cc.time), "value": int(cc.value)})
            elif cc.number == 67:
                out["soft"].append({"time": float(cc.time), "value": int(cc.value)})
    for events in out.values():
        events.sort(key=lambda e: e["time"])
    return out


def articulation_features(notes: Sequence[Note]) -> dict[str, float]:
    """Touch: how long each note is held relative to the time until the next attack.

    ratio = duration / IOI-to-next-distinct-onset. ~0.3 is crisp staccato, ~1.0 is
    legato, >1 means the note is still sounding when the next is struck (overlapping
    legato or pedalled texture). This is THE articulation measure and is structurally
    unavailable to an offset-blind feature set.

    Also reports melodic-legato overlap restricted to pitch-adjacent successive notes
    (<=12 semitones apart), which approximates within-a-single-line legato rather than
    the trivial cross-hand overlap that any dense texture produces.
    """
    ordered = sorted(notes, key=lambda n: (n["onset"], n["pitch"]))
    n = len(ordered)
    keys = ("artic_ratio_median", "artic_ratio_iqr", "artic_entropy",
            "staccato_fraction", "legato_fraction", "overlap_fraction",
            "melodic_legato_fraction", "artic_lz")
    if n < 2:
        return dict.fromkeys(keys, float("nan"))

    onsets = np.array([x["onset"] for x in ordered], float)
    offsets = np.array([x["offset"] for x in ordered], float)
    pitches = np.array([x["pitch"] for x in ordered], np.int64)
    durs = np.clip(offsets - onsets, 0.0, None)

    # IOI from each note to the next STRICTLY LATER onset (chord-mates share an IOI).
    uniq = np.unique(onsets)
    nxt = np.searchsorted(uniq, onsets, side="right")
    has_next = nxt < uniq.size
    if not has_next.any():
        return dict.fromkeys(keys, float("nan"))
    ioi = np.full(n, np.nan)
    ioi[has_next] = uniq[nxt[has_next]] - onsets[has_next]

    valid = has_next & (ioi > 1e-3)
    if valid.sum() < 2:
        return dict.fromkeys(keys, float("nan"))
    ratio = durs[valid] / ioi[valid]

    sym = np.digitize(ratio, ARTIC_EDGES)
    # Successive-pair overlap, and the same restricted to nearby pitches (one "line").
    prev_over_next = offsets[:-1] - onsets[1:]
    near = np.abs(np.diff(pitches)) <= 12
    return {
        "artic_ratio_median": float(np.median(ratio)),
        "artic_ratio_iqr": float(np.percentile(ratio, 75) - np.percentile(ratio, 25)),
        "artic_entropy": _entropy(sym),
        "artic_lz": float(pitch_lz_complexity(sym.tolist())),
        "staccato_fraction": float(np.mean(ratio < 0.5)),
        "legato_fraction": float(np.mean((ratio >= 0.95) & (ratio <= 1.1))),
        "overlap_fraction": float(np.mean(ratio > 1.1)),
        "melodic_legato_fraction": (float(np.mean(prev_over_next[near] > 0))
                                    if near.any() else float("nan")),
    }


def duration_features(notes: Sequence[Note]) -> dict[str, float]:
    """Note-length variety: how many distinct rhythmic VALUES the piece asks for.

    Distinct from the existing IOI features: IOI measures the gap between attacks, so a
    held whole note and a staccato quarter followed by a rest are identical to it.
    Duration entropy/LZ separate them, which is exactly the notational complexity a
    grader reads off the page.
    """
    durs = np.array([float(x["offset"]) - float(x["onset"]) for x in notes], float)
    durs = durs[durs > 1e-4]
    keys = ("dur_entropy", "dur_lz", "dur_cv", "dur_median",
            "dur_range_log", "unique_dur_ratio")
    if durs.size < 2:
        return dict.fromkeys(keys, float("nan"))
    sym = np.digitize(durs, DUR_EDGES)
    mean = float(np.mean(durs))
    return {
        "dur_entropy": _entropy(sym),
        "dur_lz": float(pitch_lz_complexity(sym.tolist())),
        "dur_cv": float(np.std(durs) / mean) if mean > 0 else float("nan"),
        "dur_median": float(np.median(durs)),
        "dur_range_log": float(np.log2(durs.max() / durs.min())),
        "unique_dur_ratio": float(np.unique(sym).size / sym.size),
    }


def _concurrency_timeline(notes: Sequence[Note]) -> tuple[np.ndarray, np.ndarray]:
    """(voice_counts, segment_durations) from an onset/offset sweep.

    Sorting +1/-1 events by time gives the number of simultaneously SOUNDING notes on
    each inter-event segment. Weighting by segment length makes the statistics
    time-weighted, so a long held chord counts for its actual duration rather than
    once. This is the quantity `polyphony_per_onset` cannot compute.
    """
    starts = np.array([float(x["onset"]) for x in notes], float)
    ends = np.array([float(x["offset"]) for x in notes], float)
    keep = ends > starts
    starts, ends = starts[keep], ends[keep]
    if starts.size < 1:
        return np.array([]), np.array([])
    times = np.concatenate([starts, ends])
    deltas = np.concatenate([np.ones(starts.size), -np.ones(ends.size)])
    order = np.lexsort((deltas, times))   # releases before attacks at equal time
    times, deltas = times[order], deltas[order]
    counts = np.cumsum(deltas)
    seg_dur = np.diff(times)
    return counts[:-1], seg_dur


def voicing_features(notes: Sequence[Note]) -> dict[str, float]:
    """True sustained polyphony, time-weighted, plus how fast the voice count churns.

    Voice statistics are normalized over SOUNDING time, not wall-clock time, and rests
    are reported separately as `frac_time_silent`. Normalizing over wall-clock would
    multiply two independent difficulty signals together -- a sparse monophonic line
    with rests would report the same mean voice count as a continuous one -- and the
    model cannot undo that product afterwards.
    """
    keys = ("concurrency_mean", "concurrency_p90", "concurrency_max",
            "frac_time_polyphonic", "frac_time_ge3_voices", "frac_time_ge5_voices",
            "voice_change_rate", "frac_time_silent")
    counts, seg = _concurrency_timeline(notes)
    wall = float(seg.sum()) if seg.size else 0.0
    if counts.size < 1 or wall <= 0:
        return dict.fromkeys(keys, float("nan"))

    sounding = counts >= 1
    total = float(seg[sounding].sum())
    if total <= 0:
        return dict.fromkeys(keys, float("nan"))
    silent_fraction = float(seg[~sounding].sum() / wall)
    counts, seg = counts[sounding], seg[sounding]
    w = seg / total

    def frac(threshold: int) -> float:
        return float(w[counts >= threshold].sum())

    # Weighted percentile of the voice count: sort by count, walk the weight mass.
    idx = np.argsort(counts)
    cw = np.cumsum(w[idx])
    p90 = float(counts[idx][np.searchsorted(cw, 0.9)]) if cw.size else float("nan")
    # How often the sounding-voice count changes per second -- texture churn, which is
    # a rate (length-invariant) rather than a count.
    changes = int(np.count_nonzero(np.diff(counts))) if counts.size >= 2 else 0
    return {
        "concurrency_mean": float(np.sum(counts * w)),
        "concurrency_p90": p90,
        "concurrency_max": float(counts.max()),
        "frac_time_polyphonic": frac(2),
        "frac_time_ge3_voices": frac(3),
        "frac_time_ge5_voices": frac(5),
        "voice_change_rate": float(changes / wall),
        "frac_time_silent": silent_fraction,
    }


def release_features(notes: Sequence[Note]) -> dict[str, float]:
    """Do a chord's notes release together, or independently?

    Simultaneous release = block chord. Dispersed release = independent voice-leading
    (holding an inner voice while the others lift), which is markedly harder. Measured
    as offset spread within each >=2-note onset cluster.
    """
    keys = ("release_dispersion_median", "release_dispersion_p90",
            "frac_chords_released_together", "chord_release_ratio_median")
    ordered = sorted(notes, key=lambda x: (x["onset"], x["pitch"]))
    if len(ordered) < 2:
        return dict.fromkeys(keys, float("nan"))

    spreads, ratios = [], []
    cluster = [ordered[0]]
    for x in ordered[1:]:
        if x["onset"] - cluster[0]["onset"] <= CHORD_TOL_S:
            cluster.append(x)
        else:
            if len(cluster) >= 2:
                offs = np.array([c["offset"] for c in cluster], float)
                durs = np.array([c["offset"] - c["onset"] for c in cluster], float)
                spreads.append(float(offs.max() - offs.min()))
                md = float(np.median(durs))
                if md > 1e-4:
                    ratios.append(float(offs.max() - offs.min()) / md)
            cluster = [x]
    if len(cluster) >= 2:
        offs = np.array([c["offset"] for c in cluster], float)
        durs = np.array([c["offset"] - c["onset"] for c in cluster], float)
        spreads.append(float(offs.max() - offs.min()))
        md = float(np.median(durs))
        if md > 1e-4:
            ratios.append(float(offs.max() - offs.min()) / md)

    if not spreads:
        return dict.fromkeys(keys, float("nan"))
    sp = np.asarray(spreads, float)
    return {
        "release_dispersion_median": float(np.median(sp)),
        "release_dispersion_p90": float(np.percentile(sp, 90)),
        # "together" = all voices lift within the same 30 ms window used for attacks.
        "frac_chords_released_together": float(np.mean(sp <= CHORD_TOL_S)),
        "chord_release_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
    }


def pedal_features(sustain: Sequence[dict], soft: Sequence[dict],
                   start_s: float, end_s: float) -> dict[str, float]:
    """Sustain-pedal TIMING from CC64 (+ una corda usage from CC67).

    Deliberately timing-only. The obvious richer idea -- half-pedalling, measured from
    CC64 values strictly inside the extremes -- is NOT EXPRESSIBLE from Transkun output:
    Transkun's pedal head is a binary detector that emits only 0 and 127, so depth mean
    (always 63.5), value entropy (always 1.0) and half-pedal fraction (always 0.0) are
    constants across every piece in the corpus, carrying zero information. They are
    omitted rather than shipped as dead columns. Recovering pedal DEPTH would require a
    different transcriber, not a different feature. What Transkun does supply honestly
    is WHEN and HOW OFTEN the pedal moves, which is what remains here.

    The window [start_s, end_s] must be the note window in the SAME absolute clock as
    the pedal event times. Pedal events are timestamped from the start of the file while
    the first note may begin much later, so normalizing by the note span alone lets
    hold-times sum past the span and produces an "on fraction" above 1.

    A piece with genuinely no CC64 gets 0.0 for the usage terms (a true observation)
    but nan for segment shape, which is undefined with no events.
    """
    keys = ("pedal_change_rate", "pedal_on_fraction", "pedal_segment_mean_s",
            "pedal_segment_cv", "soft_pedal_rate", "soft_pedal_used")
    window = float(end_s) - float(start_s)
    if window <= 0:
        return dict.fromkeys(keys, float("nan"))

    soft_rate = float(len(soft) / window)
    if not sustain:
        return {"pedal_change_rate": 0.0, "pedal_on_fraction": 0.0,
                "pedal_segment_mean_s": float("nan"), "pedal_segment_cv": float("nan"),
                "soft_pedal_rate": soft_rate, "soft_pedal_used": float(bool(soft))}

    times = np.array([e["time"] for e in sustain], float)
    values = np.array([e["value"] for e in sustain], float)

    # Time-weighted fraction of the piece with the pedal down: each event's value holds
    # until the next event, the last to end_s. Both edges are clipped into the note
    # window so the result is a genuine fraction of the sounding piece.
    edges = np.clip(np.append(times, end_s), start_s, end_s)
    holds = np.clip(np.diff(edges), 0.0, None)
    down = values >= PEDAL_ON
    on_fraction = float(holds[down].sum() / window)

    # Sustained "pedal down" segments -> how long each pedalling gesture lasts.
    seg_lengths = []
    run = 0.0
    for is_down, h in zip(down, holds):
        if is_down:
            run += h
        elif run > 0:
            seg_lengths.append(run)
            run = 0.0
    if run > 0:
        seg_lengths.append(run)
    seg = np.asarray(seg_lengths, float)
    seg_mean = float(seg.mean()) if seg.size else float("nan")
    seg_cv = float(seg.std() / seg.mean()) if seg.size >= 2 and seg.mean() > 0 else float("nan")

    return {
        "pedal_change_rate": float(len(sustain) / window),
        "pedal_on_fraction": on_fraction,
        "pedal_segment_mean_s": seg_mean,
        "pedal_segment_cv": seg_cv,
        "soft_pedal_rate": soft_rate,
        "soft_pedal_used": float(bool(soft)),
    }


def transkun_features(notes: Sequence[Note], sustain: Sequence[dict],
                      soft: Sequence[dict]) -> dict[str, float]:
    """All Transkun-unlocked features for one piece. Prefixed `tk_` so an ablation can
    select the new family by name without maintaining a parallel list."""
    if len(notes) < 2:
        raise ValueError("need >=2 notes for transkun features")
    onsets = [float(x["onset"]) for x in notes]
    offsets = [float(x["offset"]) for x in notes]
    # The note window in absolute file time -- pedal events share this clock.
    start = min(onsets)
    end = max(max(offsets), max(onsets))
    out: dict[str, float] = {}
    out.update(articulation_features(notes))
    out.update(duration_features(notes))
    out.update(voicing_features(notes))
    out.update(release_features(notes))
    out.update(pedal_features(sustain, soft, start, end))
    return {f"tk_{k}": v for k, v in out.items()}
