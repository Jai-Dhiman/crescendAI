# model/src/follower_eval/calibration_recall.py
"""G-OOD-0: reference-channel Transkun note recall on calibration takes (#148).

This is the issue's one BLOCKING gate. Its bar is reference-channel Transkun
note recall vs the known score >= 0.95, and its stated consequence is that
failure "stops Phase 2; remedy is mic placement or the retrofit, not a caveat".
Nothing else in the repo measures it, and the recording session is not cheaply
repeatable, so it has to be scoreable before the microphone goes up.

WHAT A MISS ACTUALLY MEANS
--------------------------
A score note counted as missing can come from three different places, and they
do not have the same remedy:

  1. Transkun failed to transcribe a note that was played -- the quantity the
     gate exists to measure, and the one mic placement can fix.
  2. The performer did not play the note -- a performance error.
  3. The MATCHER failed to pair a note Transkun transcribed correctly.

Source 3 is the dangerous one. Phase 2's truth (provenance B) is parangonar
output, so scoring this gate with parangonar alone would let a matcher weakness
either cancel the session for the wrong reason or pass the gate and then quietly
degrade the truth it qualified. That is the soft form of #101's gate1 mistake --
parangonar scored against parangonar is agreement, not accuracy.

So recall is reported as TWO ARMS, never one number:

  * ``recall_parangonar`` -- the real Phase-2 pipeline. The gate's bar applies
    to this arm, because this is the matcher truth will actually come from.
  * ``recall_sequence`` -- a TIMING-FREE arm: the longest common subsequence of
    the two pitch streams. It never looks at a clock, so it cannot be defeated
    by rubato, tempo choice, or the match-scatter that ``score_align`` already
    documented in parangonar's global matcher. It is not a proof-grade upper
    bound (parangonar can emit a non-monotone pair that LCS forbids), but a
    large gap between the arms localizes the loss to the matcher rather than to
    the channel.

Combining the arms with the single existing 0.95 bar -- no second threshold is
invented -- gives an ACTIONABLE verdict instead of a bare fail:

  PASS          parangonar arm >= bar.
  FAIL_MATCHER  parangonar arm < bar but the timing-free arm >= bar. The notes
                are in the transcription; the aligner is losing them. Re-siting
                a microphone will not help.
  FAIL_CHANNEL  both arms < bar. The notes are not in the transcription. This is
                the failure the issue's remedy is written for.
  UNINFORMATIVE the matcher floor itself is < bar (see below), so the gate
                cannot discriminate at all and its numbers mean nothing. Same
                precedent as G-OOD-6, which this issue already recorded as
                non-discriminating rather than quietly reporting its pass.

THE MATCHER FLOOR
-----------------
``matcher_floor`` runs the score against ITSELF through the identical parangonar
call. A perfect matcher returns 1.0. Whatever it actually returns is the ceiling
every take in the session is measured against, and it isolates source 3 with no
audio, no piano and no performer. It is computed on every run and reported
whether or not anyone looks at it.

SOURCE 2 IS NOT REMOVED, IT IS DECLARED
---------------------------------------
There is no code fix for a note the performer did not play; separating it would
need the symbolic sensor retrofit this issue explicitly deferred. A performer
omission therefore counts against Transkun, which makes every recall here a
LOWER BOUND on Transkun's true note recall. The bias has a direction and the
direction is safe: a PASS is conservative, and only a FAIL is ambiguous -- which
is exactly what the two arms are there to disambiguate.

Because the bias is one-directional, the gate is scored ONLY on takes the
manifest declares ``"behavior": "calibration"``. A practice take contains
deliberate wrong notes and restarts; scoring the gate there would fail it on the
performance, not the channel.

RUNNING (from the PRIMARY checkout -- data/ is gitignored in worktrees):

  cd /Users/jdhiman/Documents/crescendai/model
  PYTHONPATH=<worktree>/model/src .venv/bin/python \
    -m follower_eval.calibration_recall \
    --manifest sessions/cal_01/take.json sessions/cal_02/take.json \
    --out reports/g_ood_0.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from follower_eval.asap_audio import SAMPLE_RATE, TRANSCRIBER_ID
from follower_eval.take_capture import TakeCaptureError, load_manifest

# The gate bar, fixed ex ante in the issue body. One bar, reused by every
# comparison in this module; nothing here introduces a second threshold.
G_OOD_0_BAR = 0.95

# parangonar needs a beat axis and mis-initializes when the score's nominal
# tempo is used, because a calibration take runs slower than the render. Both
# arrays therefore use seconds-as-beats -- the convention chroma_dtw_eval's
# _amt_to_perf_na established and measured (score coverage 0.41 -> 0.59).
BEAT_SEC = 1.0

# Notes whose onsets fall within this of each other are one chord, and are
# ordered by pitch rather than by arrival so both streams present a chord the
# same way. Without it the timing-free arm reads 0.75 on a PERFECT transcription
# (measured): transcribed onsets carry ~20 ms of jitter, which inverts roughly
# half of all two-note chords, and LCS then loses a note on each inversion. That
# would put the arm permanently under the bar, make FAIL_MATCHER unreachable and
# route every failure to FAIL_CHANNEL -- the exact misattribution the two arms
# exist to prevent. 50 ms is a physical statement about what "simultaneous"
# means on a piano, not a tuned parameter; a run fast enough to be chained by it
# would be 33 notes per second.
CHORD_WINDOW_S = 0.05

# The timing-free arm is an O(n*m) DP. Above this many cells it is refused
# rather than left to run for minutes with no output -- a calibration take is
# short by construction, so hitting this means the wrong file was passed.
MAX_LCS_CELLS = 6_000_000

_PERF_DTYPE = [
    ("onset_sec", float),
    ("onset_beat", float),
    ("duration_sec", float),
    ("duration_beat", float),
    ("pitch", int),
    ("velocity", int),
    ("id", "U32"),
]


class CalibrationRecallError(RuntimeError):
    """Raised when the gate cannot be scored as specified. Loud by design: an
    unscoreable calibration take must stop the session, not produce a number
    whose provenance nobody can reconstruct afterwards."""


@dataclass(frozen=True)
class TakeRecall:
    """One calibration take's reference-channel recall, both arms."""

    take_id: str
    score_midi: str
    n_score_notes: int
    n_transcribed_notes: int
    n_matched_parangonar: int
    n_matched_sequence: int
    recall_parangonar: float
    recall_sequence: float
    # Transcribed notes parangonar paired with no score note. Descriptive only:
    # the reference channel also captures the clap slates and room noise, and a
    # spurious note is not a recall miss.
    unmatched_transcribed_frac: float


# --- note arrays ------------------------------------------------------------


def _chord_ordered(notes: list[dict]) -> list[dict]:
    """Notes in onset order, with each chord's members ordered by pitch.

    A chord is a run of consecutive notes no more than ``CHORD_WINDOW_S`` apart.
    Ordering inside it by pitch rather than by arrival makes the score stream and
    the transcribed stream present the same chord identically, which is what the
    timing-free arm needs; see CHORD_WINDOW_S for the measurement that forced it.
    """
    ordered = sorted(notes, key=lambda n: float(n["onset"]))
    out: list[dict] = []
    group: list[dict] = []
    for n in ordered:
        if group and float(n["onset"]) - float(group[-1]["onset"]) > CHORD_WINDOW_S:
            out.extend(sorted(group, key=lambda g: int(g["pitch"])))
            group = []
        group.append(n)
    out.extend(sorted(group, key=lambda g: int(g["pitch"])))
    return out


def _note_array(notes: list[dict], prefix: str) -> np.ndarray:
    """(onset, offset, pitch) dicts -> the structured array parangonar expects,
    in chord-canonical order, with stable ids of the form ``<prefix><index>``."""
    if not notes:
        raise CalibrationRecallError(f"{prefix}: zero notes; nothing to match")
    notes = _chord_ordered(notes)
    arr = np.empty(len(notes), dtype=_PERF_DTYPE)
    for i, n in enumerate(notes):
        onset = float(n["onset"])
        dur = max(float(n["offset"]) - onset, 0.001)
        arr[i] = (
            onset,
            onset / BEAT_SEC,
            dur,
            dur / BEAT_SEC,
            int(n["pitch"]),
            int(n.get("velocity", 80)),
            f"{prefix}{i}",
        )
    # No re-sort here: _chord_ordered already fixed the order, and sorting the
    # array would undo the chord grouping and desynchronise the ids from it.
    return arr


def load_score_notes(score_midi: Path) -> list[dict]:
    """Score MIDI -> note dicts. The score is the KNOWN quantity in this gate;
    it is read straight off disk and never derived from any audio."""
    if not score_midi.exists():
        raise CalibrationRecallError(f"score MIDI missing: {score_midi}")
    import partitura as pa

    na = pa.load_performance_midi(str(score_midi)).note_array()
    return [
        {
            "onset": float(r["onset_sec"]),
            "offset": float(r["onset_sec"]) + float(r["duration_sec"]),
            "pitch": int(r["pitch"]),
            "velocity": int(r["velocity"]),
        }
        for r in na
    ]


# --- arm 1: parangonar ------------------------------------------------------


def _match(score_na: np.ndarray, perf_na: np.ndarray) -> list[dict]:
    """The parangonar seam, isolated so tests can replace it without the
    library, exactly as score_align's hermetic tests do."""
    import parangonar as pa

    return list(pa.AutomaticNoteMatcher()(score_na, perf_na))


def parangonar_matched_ids(
    score_na: np.ndarray, perf_na: np.ndarray, match=_match
) -> tuple[set[str], set[str]]:
    """(matched score ids, matched performance ids) from ONE parangonar call.

    One call, not one per side: ``extract_cli`` guards this library with a
    timeout because it can blow up combinatorially on real transcriptions, and
    matching twice for two views of the same result would double both the
    runtime and that exposure on a gate a human is waiting on.

    Sets, not counts: parangonar can emit the same score id twice, and counting
    entries would then report recall above 1.0 on a duplicate.
    """
    known_s = {str(s["id"]) for s in score_na}
    known_p = {str(p["id"]) for p in perf_na}
    matched_s: set[str] = set()
    matched_p: set[str] = set()
    for e in match(score_na, perf_na):
        if e.get("label") != "match":
            continue
        s_id, p_id = str(e.get("score_id")), str(e.get("performance_id"))
        if s_id in known_s:
            matched_s.add(s_id)
        if p_id in known_p:
            matched_p.add(p_id)
    return matched_s, matched_p


# --- arm 2: timing-free -----------------------------------------------------


def sequence_matched(score_pitches: list[int], perf_pitches: list[int]) -> int:
    """Longest common subsequence of the two pitch streams.

    No clock is consulted, so tempo, rubato and drift cannot cost a match. Both
    streams arrive sorted by (onset, pitch), which puts the notes of a chord in
    the same order on both sides so a chord is not split by arrival order.

    This bounds what a monotone matcher could achieve; it is not a bound on
    parangonar in the strict sense, because parangonar's global matcher can pair
    non-monotonically. It is here to localize a failure, not to replace the
    other arm.
    """
    n, m = len(score_pitches), len(perf_pitches)
    if n == 0 or m == 0:
        return 0
    if n * m > MAX_LCS_CELLS:
        raise CalibrationRecallError(
            f"timing-free arm refused: {n} score notes x {m} transcribed notes "
            f"= {n * m:,} cells, over the {MAX_LCS_CELLS:,} limit. A calibration "
            f"take is short by construction -- check the score and audio are the "
            f"same excerpt rather than a whole-recital file."
        )
    prev = [0] * (m + 1)
    for a in score_pitches:
        cur = [0] * (m + 1)
        for j, b in enumerate(perf_pitches, start=1):
            cur[j] = prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


# --- matcher floor ----------------------------------------------------------


def matcher_floor(score_notes: list[dict], match=_match) -> float:
    """Recall of the score against ITSELF through the identical parangonar call.

    A perfect matcher returns 1.0. Anything less is matcher loss with no audio,
    no piano and no performer involved, and it is the ceiling every take in the
    session is measured against. Reported unconditionally.
    """
    score_na = _note_array(score_notes, "s")
    perf_na = _note_array(score_notes, "p")
    matched, _ = parangonar_matched_ids(score_na, perf_na, match)
    return len(matched) / len(score_na)


# --- one take ---------------------------------------------------------------


def to_transcriber_wav(src: Path, dst: Path) -> Path:
    """Downsample a 48 kHz/24-bit session recording to the 16 kHz mono the
    production transcriber consumes -- the same preprocessing asap_audio feeds
    the 279-clip corpus. Going in through a different path would introduce a
    channel difference that is not the one being measured."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(src),
           "-ac", "1", "-ar", str(SAMPLE_RATE), str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        raise CalibrationRecallError(
            f"ffmpeg failed downsampling {src}: {r.stderr[-400:]}"
        )
    return dst


@dataclass(frozen=True)
class CalibrationTake:
    """One validated calibration manifest, read once."""

    take_id: str
    ref_wav: Path
    score_midi: Path


def load_calibration_take(manifest_path: Path) -> CalibrationTake:
    """Validate and resolve one calibration manifest in a single read.

    Every refusal below is a refusal to score, not a value to fall back on --
    the same precedent take_capture set when it declined to substitute the
    reference channel for a missing one.
    """
    body = load_manifest(manifest_path)
    behavior = body.get("behavior")
    if behavior != "calibration":
        raise CalibrationRecallError(
            f"{manifest_path}: behavior is {behavior!r}, not 'calibration'. "
            f"G-OOD-0 is defined on calibration takes only; a take with wrong "
            f"notes in it would fail the gate on the performance, not the "
            f"reference channel."
        )
    if "score_midi" not in body:
        raise CalibrationRecallError(
            f"{manifest_path}: no 'score_midi'. G-OOD-0 is recall against a "
            f"KNOWN score; without one there is nothing to be recalled."
        )
    return CalibrationTake(
        take_id=body["take_id"],
        ref_wav=body["channels"][body["reference_channel"]],
        score_midi=manifest_path.parent / body["score_midi"],
    )


def reference_channel_path(manifest_path: Path) -> Path:
    """The reference channel's audio, and a refusal if the manifest is not a
    calibration take.

    The gate charges performer omissions to Transkun (see the module docstring),
    which is only conservative on a take played straight. On a practice take --
    deliberate wrong notes, restarts, repeats -- the same arithmetic would fail
    the gate on the performance and cancel Phase 2 for a reason that has nothing
    to do with the channel.
    """
    return load_calibration_take(manifest_path).ref_wav


def score_midi_path(manifest_path: Path) -> Path:
    return load_calibration_take(manifest_path).score_midi


def score_take(
    take_id: str,
    ref_wav: Path,
    score_notes: list[dict],
    transcribe_wav,
    score_midi_name: str = "",
) -> TakeRecall:
    """Transcribe one calibration take's reference channel and score both arms.

    ``transcribe_wav`` is injected rather than imported, matching
    ``asap_audio.build_one``: it keeps the whole harness testable with no model
    weights and no audio.
    """
    with tempfile.TemporaryDirectory() as td:
        wav = to_transcriber_wav(ref_wav, Path(td) / "ref16k.wav")
        notes = transcribe_wav(wav)[0]
    if not notes:
        raise CalibrationRecallError(
            f"{take_id}: {TRANSCRIBER_ID} returned zero notes from {ref_wav}. "
            f"That is a dead channel or a failed transcription, not a recall of 0."
        )

    score_na = _note_array(score_notes, "s")
    perf_na = _note_array(notes, "p")

    matched_score, matched_perf = parangonar_matched_ids(score_na, perf_na)
    n_seq = sequence_matched(
        [int(p) for p in score_na["pitch"]], [int(p) for p in perf_na["pitch"]]
    )

    n_score = len(score_na)
    return TakeRecall(
        take_id=take_id,
        score_midi=score_midi_name,
        n_score_notes=n_score,
        n_transcribed_notes=len(perf_na),
        n_matched_parangonar=len(matched_score),
        n_matched_sequence=n_seq,
        recall_parangonar=round(len(matched_score) / n_score, 4),
        recall_sequence=round(n_seq / n_score, 4),
        unmatched_transcribed_frac=round(
            1.0 - len(matched_perf) / len(perf_na), 4
        ),
    )


# --- gate -------------------------------------------------------------------


def verdict(
    recall_parangonar: float, recall_sequence: float, floor: float
) -> tuple[str, str]:
    """(verdict, why). Uses only G_OOD_0_BAR -- no second threshold."""
    if floor < G_OOD_0_BAR:
        return (
            "UNINFORMATIVE",
            f"the matcher scores the score against ITSELF at {floor:.4f}, below "
            f"the {G_OOD_0_BAR} bar. Every take is measured against that ceiling, "
            f"so neither arm's number carries information about the channel. Fix "
            f"the matcher before reading this gate; do not report a pass or a "
            f"fail from it.",
        )
    if recall_parangonar >= G_OOD_0_BAR:
        return (
            "PASS",
            f"reference-channel recall {recall_parangonar:.4f} >= {G_OOD_0_BAR} "
            f"through the same matcher Phase 2's truth will use.",
        )
    if recall_sequence >= G_OOD_0_BAR:
        return (
            "FAIL_MATCHER",
            f"recall is {recall_parangonar:.4f} through parangonar but "
            f"{recall_sequence:.4f} with timing ignored. The notes ARE in the "
            f"transcription and the aligner is losing them, which is the "
            f"match-scatter score_align already documents. Re-siting a "
            f"microphone will not move this number; Phase 2's truth is what is "
            f"at risk.",
        )
    return (
        "FAIL_CHANNEL",
        f"recall is {recall_parangonar:.4f} through parangonar and "
        f"{recall_sequence:.4f} with timing ignored, so the notes are not in the "
        f"transcription at all. This is the failure the issue's remedy is "
        f"written for: mic placement or the retrofit, not a caveat. Phase 2 "
        f"stops.",
    )


def run(manifest_paths: list[Path], transcribe_wav) -> dict:
    """Score G-OOD-0 over a session's calibration takes.

    No bootstrap interval. A calibration session is a handful of takes, and a
    resampling CI over three of them would be decoration, not inference; the
    per-take numbers are printed instead so a single bad take is visible rather
    than averaged away.
    """
    if not manifest_paths:
        raise CalibrationRecallError("no calibration manifests given")

    takes: list[TakeRecall] = []
    floors: dict[str, float] = {}
    # The floor is a full parangonar run over the score; several takes of one
    # piece share it, so it is computed once per score rather than once per take.
    notes_by_score: dict[Path, list[dict]] = {}
    for mp in manifest_paths:
        try:
            spec = load_calibration_take(mp)
            if spec.score_midi not in notes_by_score:
                notes_by_score[spec.score_midi] = load_score_notes(spec.score_midi)
                floors[spec.score_midi.name] = round(
                    matcher_floor(notes_by_score[spec.score_midi]), 4
                )
            takes.append(
                score_take(
                    take_id=spec.take_id,
                    ref_wav=spec.ref_wav,
                    score_notes=notes_by_score[spec.score_midi],
                    transcribe_wav=transcribe_wav,
                    score_midi_name=spec.score_midi.name,
                )
            )
        except TakeCaptureError as exc:
            raise CalibrationRecallError(f"{mp}: {exc}") from exc

    n_score = sum(t.n_score_notes for t in takes)
    pooled_par = sum(t.n_matched_parangonar for t in takes) / n_score
    pooled_seq = sum(t.n_matched_sequence for t in takes) / n_score
    floor = min(floors.values())

    v, why = verdict(pooled_par, pooled_seq, floor)
    return {
        "gate": "G-OOD-0",
        "bar": G_OOD_0_BAR,
        "system_under_test": f"{TRANSCRIBER_ID} on the reference channel, "
        f"{SAMPLE_RATE} Hz mono",
        "known_quantity": "score MIDI read from disk; never derived from audio",
        "bias": (
            "A note the performer did not play counts as a Transkun miss -- "
            "separating the two needs the symbolic retrofit this issue deferred. "
            "Every recall here is therefore a LOWER BOUND on Transkun's true note "
            "recall, so a PASS is conservative and only a FAIL is ambiguous."
        ),
        "n_takes": len(takes),
        "n_score_notes": n_score,
        "matcher_floor": floor,
        "matcher_floor_by_score": floors,
        "recall_parangonar": round(pooled_par, 4),
        "recall_sequence": round(pooled_seq, 4),
        "verdict": v,
        "why": why,
        "takes": [asdict(t) for t in takes],
    }


def _format(result: dict) -> str:
    rows = [
        f"  {t['take_id']:<24} {t['recall_parangonar']:.4f}  "
        f"{t['recall_sequence']:.4f}   {t['n_score_notes']:>6,} score notes"
        for t in result["takes"]
    ]
    return "\n".join(
        [
            "=" * 78,
            "G-OOD-0 -- REFERENCE-CHANNEL TRANSKUN NOTE RECALL (#148, BLOCKING)",
            "=" * 78,
            f"system under test : {result['system_under_test']}",
            f"takes {result['n_takes']}   score notes {result['n_score_notes']:,}"
            f"   bar {result['bar']}",
            f"matcher floor     : {result['matcher_floor']:.4f} "
            f"(score matched against itself)",
            "",
            "  take                     parangonar  timing-free",
            *rows,
            "",
            f"pooled parangonar : {result['recall_parangonar']:.4f}",
            f"pooled timing-free: {result['recall_sequence']:.4f}",
            "",
            f"VERDICT: {result['verdict']}",
            f"  {result['why']}",
            "",
            f"NOTE: {result['bias']}",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="G-OOD-0 calibration-take recall gate (#148)"
    )
    ap.add_argument("--manifest", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from follower_eval.build_corpus import _import_transcribe_wav

    result = run(args.manifest, _import_transcribe_wav())
    print(_format(result))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    if result["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
