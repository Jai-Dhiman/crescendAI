# model/src/follower_eval/take_capture.py
"""Multi-channel take capture and sync for the OOD position eval (issue #148).

A **take** is one performance recorded simultaneously on N devices: a reference
channel (directional mic, or the re-amp source) and one or more phone channels
in the positions under test. #148's central abstraction is that truth attaches
to the *take*, not to any one recording -- so the take's channels must first be
put on one clock. That is this module's whole job.

Each recorder starts at its own moment and runs on its own crystal, so a channel
differs from the reference by an **offset** (when it started) and a **drift
rate** (how fast its clock runs). A single slate gives only the offset; a head
and a tail slate give both. This module detects the slates, measures each
channel's offset at each slate by cross-correlation, and fits the affine map
that puts the channel on reference time.

TWO SLATES DEFINE THE FIT BUT CANNOT TEST IT
--------------------------------------------
A line through two points fits both points exactly. With head and tail slates
alone, residual non-linearity in the clock relationship is not merely unmeasured
-- it is unidentifiable, and the fit will look perfect whatever the truth is.
So ``sync_channel`` accepts optional **mid-take slates**: the line is fit on
head+tail only, and each mid slate's residual is reported as an independent
check on linearity. One extra clap per take converts the linearity assumption
into a measurement. ``ChannelSync.max_mid_residual_s`` is None when no mid slate
was recorded, which is a statement that linearity is UNTESTED, not that it holds.

A MISSING CHANNEL RAISES
------------------------
Every function here fails loudly. A channel named by the manifest but absent
from disk, a slate that cannot be found, a cross-correlation peak below
``MIN_SLATE_CORR`` -- each raises ``TakeCaptureError``. There is deliberately no
path that falls back to the reference channel for a missing phone channel: that
would silently report a clean-channel number as if it had come through a phone,
which is the whole quantity the eval exists to measure. Same precedent as
``asap_audio.py`` refusing to fall back to MIDI, and as ``derive_shift``
refusing to guess an offset it cannot verify.

INTAKE RUNS WHILE THE RIG IS STILL UP
-------------------------------------
``sync_take`` already refuses every take it cannot put on one clock. But it is
normally run at a desk, hours after the session, and by then a channel that
never recorded, a phone left at 44.1 kHz, or a tail clap that was never struck
are all unrecoverable. ``intake_session`` runs those same refusals -- plus the
naming and completeness checks a manifest can be wrong about -- against the raw
exports minutes after each take, while re-recording still costs one more take
instead of one more session. It is a scheduling fix, not a second validator:
the deep check IS ``sync_take``, called per take on the files it just wrote.

SCOPE
-----
Sync is trustworthy at beat resolution, not note-onset resolution. Mic distance
alone puts a few milliseconds of genuine acoustic delay between channels, and
that delay is real rather than an error to remove. #148 makes no note-level
offset claims for this reason.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Envelope smoothing for cross-correlation, in seconds. Two mics in a room see
# very different spectra but the same broadband transient, so correlation runs
# on a smoothed rectified envelope rather than the raw waveform.
ENVELOPE_SMOOTH_S = 0.002

# Half-width of the audio window correlated around each slate.
SLATE_WINDOW_S = 0.5

# Widest start-time difference between recorders we will search for, in seconds.
# Recorders started by hand, so tens of seconds is realistic; beyond this a
# "match" is more likely a different transient than the slate.
MAX_OFFSET_S = 60.0

# Normalized correlation peak below which a slate match is not trustworthy.
# Below this we raise rather than accept a lag that may be a different clap.
MIN_SLATE_CORR = 0.5

# A slate must stand this many times above the median envelope of its search
# window to count as a clap rather than loud playing.
SLATE_PROMINENCE = 6.0

# Head and tail slates closer together than this make the drift rate a ratio of
# two small numbers and amplify slate-detection noise without bound.
MIN_SLATE_SPAN_S = 30.0

# Largest drift rate any real recorder can exhibit, in ppm. Consumer crystals
# run within ~100 ppm and even a bad one stays under ~500; 1000 ppm (0.1%) is
# already an order of magnitude beyond plausible, so anything past it is a
# mis-locked slate rather than a fast clock.
#
# This bound exists because MIN_SLATE_CORR is NOT sufficient on its own.
# Measured on the synthetic fixtures: when a channel stops before the tail clap
# is struck, the correlator locks onto an unrelated transient and scores 0.512
# -- above the 0.50 floor -- yielding an offset 47 s away from the head offset.
# Genuine matches score 0.82-0.85. Rather than tune the correlation floor on
# synthetic audio, the mis-lock is caught by a physical bound on clock rate,
# which needs no tuning and does not soften on real room recordings.
MAX_PLAUSIBLE_DRIFT_PPM = 1000.0


class TakeCaptureError(RuntimeError):
    """Raised whenever a take cannot be synced as specified. Loud by design: a
    take that cannot be put on one clock is excluded and reported, never
    silently degraded to the reference channel."""


@dataclass(frozen=True)
class ChannelSync:
    """One channel's affine map onto reference time.

    ``channel_time = head_offset_s + (1 + drift_rate) * (ref_time - ref_head_s)
                     + ref_head_s`` -- see ``to_channel_time``.
    """

    name: str
    ref_head_s: float
    ref_tail_s: float
    head_offset_s: float
    tail_offset_s: float
    drift_rate: float  # dimensionless; ppm = drift_rate * 1e6
    head_corr: float
    tail_corr: float
    mid_residuals_s: tuple[float, ...] = ()

    @property
    def drift_ppm(self) -> float:
        return self.drift_rate * 1e6

    @property
    def max_mid_residual_s(self) -> float | None:
        """Largest |residual| at a mid-take slate, or None when no mid slate was
        recorded. None means linearity is UNTESTED -- not that it holds."""
        if not self.mid_residuals_s:
            return None
        return max(abs(r) for r in self.mid_residuals_s)

    def to_channel_time(self, ref_time: float | np.ndarray):
        """Reference-clock time -> this channel's clock time."""
        return (
            ref_time
            + self.head_offset_s
            + self.drift_rate * (np.asarray(ref_time) - self.ref_head_s)
        )

    def to_ref_time(self, channel_time: float | np.ndarray):
        """This channel's clock time -> reference-clock time (the inverse)."""
        ct = np.asarray(channel_time)
        return (ct - self.head_offset_s + self.drift_rate * self.ref_head_s) / (
            1.0 + self.drift_rate
        )


@dataclass(frozen=True)
class Take:
    """One performance on one clock: every channel mapped to reference time."""

    take_id: str
    reference_channel: str
    channels: dict[str, Path]
    syncs: dict[str, ChannelSync] = field(default_factory=dict)


# --- audio io ---------------------------------------------------------------


def load_channel(path: Path) -> tuple[np.ndarray, int]:
    """Load one channel as mono float32.

    Raises:
        TakeCaptureError: the file is absent or unreadable. There is no fallback
            -- see the module docstring.
    """
    if not path.exists():
        raise TakeCaptureError(
            f"channel audio missing: {path}. A missing channel is an error; this "
            f"take is excluded. Refusing to substitute the reference channel -- "
            f"that would report a clean-channel result as a phone-channel one."
        )
    import soundfile as sf

    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise TakeCaptureError(f"could not read {path}: {type(exc).__name__}: {exc}")
    if x.size == 0:
        raise TakeCaptureError(f"{path}: zero samples")
    return x.mean(axis=1), int(sr)


def envelope(x: np.ndarray, sr: int) -> np.ndarray:
    """Smoothed rectified envelope. Correlating this rather than the raw
    waveform survives the very different spectra two mics give the same event,
    while keeping sample-resolution timing on transients."""
    n = max(1, int(round(ENVELOPE_SMOOTH_S * sr)))
    kernel = np.ones(n, dtype=np.float64) / n
    return np.convolve(np.abs(x.astype(np.float64)), kernel, mode="same")


# --- slate detection --------------------------------------------------------


def detect_slate(
    x: np.ndarray, sr: int, search_start_s: float, search_end_s: float
) -> float:
    """Time of the clap slate inside [search_start_s, search_end_s), in seconds.

    A slate is the sharpest ONSET in the window -- the largest positive jump in
    the envelope, not the largest amplitude, so a loud chord does not outrank a
    clap. Requires the winner to stand SLATE_PROMINENCE times above the window's
    median rise.

    Raises:
        TakeCaptureError: the window is empty or holds nothing clap-like.
    """
    a = max(0, int(round(search_start_s * sr)))
    b = min(len(x), int(round(search_end_s * sr)))
    if b - a < int(0.05 * sr):
        raise TakeCaptureError(
            f"slate search window [{search_start_s:.1f}, {search_end_s:.1f})s is "
            f"empty or shorter than 50 ms in a {len(x) / sr:.1f}s channel"
        )
    env = envelope(x[a:b], sr)
    rise = np.diff(env, prepend=env[0])
    rise[rise < 0] = 0.0
    med = float(np.median(rise[rise > 0])) if np.any(rise > 0) else 0.0
    peak = int(np.argmax(rise))
    if med <= 0 or rise[peak] < SLATE_PROMINENCE * med:
        raise TakeCaptureError(
            f"no clap slate in [{search_start_s:.1f}, {search_end_s:.1f})s: "
            f"sharpest onset is {rise[peak] / med if med > 0 else float('inf'):.1f}x "
            f"the median rise, below the {SLATE_PROMINENCE}x floor. Refusing to "
            f"sync on a transient that is probably playing, not a slate."
        )
    return (a + peak) / sr


# --- cross-correlation ------------------------------------------------------


def _sliding_energy(x: np.ndarray, width: int) -> np.ndarray:
    """Sum of ``x**2`` over every length-``width`` window, as a prefix-sum.

    This is the normalizing denominator of the cross-correlation below, and it
    is a sliding window sum -- so computing it as ``np.convolve(x**2,
    ones(width))`` pays a general O(N*width) convolution for something a prefix
    sum does in O(N). Measured on this machine at the sizes intake actually
    sees (120 s search band, 1 s template): 4.7 s at 8 kHz, 14.4 s at 16 kHz,
    and **105 s at the 48 kHz the rig records at** -- roughly 100x the FFT
    correlation it normalizes. At 3 slates and 3 phone channels that is ~16
    minutes per take, which would put intake's answer hours after the rig came
    down, i.e. after the point where re-recording is still possible.
    """
    c = np.cumsum(x.astype(np.float64) ** 2)
    return np.concatenate(([c[width - 1]], c[width:] - c[:-width]))


def estimate_offset(
    ref: np.ndarray,
    chan: np.ndarray,
    sr: int,
    ref_slate_s: float,
    max_offset_s: float = MAX_OFFSET_S,
) -> tuple[float, float]:
    """(offset_s, normalized_corr) for one channel at one slate.

    ``offset_s`` is how far the channel's copy of this event sits AFTER the
    reference's: ``channel_time = ref_time + offset_s`` at this slate.

    Correlates a SLATE_WINDOW_S window of the reference envelope against the
    channel envelope over a +/-max_offset_s search band.

    Raises:
        TakeCaptureError: the peak correlation is below MIN_SLATE_CORR, i.e. the
            match is not trustworthy enough to sync on.
    """
    half = int(round(SLATE_WINDOW_S * sr))
    c = int(round(ref_slate_s * sr))
    lo, hi = max(0, c - half), min(len(ref), c + half)
    if hi - lo < half:
        raise TakeCaptureError(
            f"reference slate at {ref_slate_s:.2f}s is too close to the edge to "
            f"cut a {2 * SLATE_WINDOW_S:.1f}s correlation window"
        )
    template = envelope(ref[lo:hi], sr)
    template = template - template.mean()

    band = int(round(max_offset_s * sr))
    slo, shi = max(0, lo - band), min(len(chan), hi + band)
    segment = envelope(chan[slo:shi], sr)
    if len(segment) < len(template):
        raise TakeCaptureError(
            f"channel is shorter than the correlation window at slate "
            f"{ref_slate_s:.2f}s ({len(segment)} < {len(template)} samples)"
        )
    segment = segment - segment.mean()

    from scipy.signal import correlate

    corr = correlate(segment, template, mode="valid")
    denom = np.linalg.norm(template) * np.sqrt(_sliding_energy(segment, len(template)))
    with np.errstate(invalid="ignore", divide="ignore"):
        ncorr = np.where(denom > 0, corr / denom, 0.0)

    k = int(np.argmax(ncorr))
    peak = float(ncorr[k])
    if peak < MIN_SLATE_CORR:
        raise TakeCaptureError(
            f"slate at {ref_slate_s:.2f}s: best correlation {peak:.2f} is below "
            f"the {MIN_SLATE_CORR} floor. The lag found is more likely a "
            f"different transient than the slate. Refusing to sync on it."
        )
    return (slo + k - lo) / sr, peak


# --- the affine fit ---------------------------------------------------------


def sync_channel(
    name: str,
    ref: np.ndarray,
    chan: np.ndarray,
    sr: int,
    head_slate_s: float,
    tail_slate_s: float,
    mid_slates_s: tuple[float, ...] = (),
) -> ChannelSync:
    """Fit one channel's affine map to reference time from head + tail slates.

    Mid slates, when present, are NOT fit -- they are held out and their
    residuals reported, which is the only way two-slate linearity can be
    checked at all (a line through two points fits both exactly).

    Raises:
        TakeCaptureError: the slates are closer together than MIN_SLATE_SPAN_S,
            which makes the drift rate a ratio of two small numbers; or the
            fitted drift exceeds MAX_PLAUSIBLE_DRIFT_PPM, which means a slate
            mis-locked rather than that the clock is fast.
    """
    span = tail_slate_s - head_slate_s
    if span < MIN_SLATE_SPAN_S:
        raise TakeCaptureError(
            f"{name}: head and tail slates are {span:.1f}s apart, below the "
            f"{MIN_SLATE_SPAN_S}s floor. The drift rate would be slate-detection "
            f"noise divided by a small number. Re-slate or drop the take."
        )
    head_off, head_corr = estimate_offset(ref, chan, sr, head_slate_s)
    tail_off, tail_corr = estimate_offset(ref, chan, sr, tail_slate_s)
    drift = (tail_off - head_off) / span

    if abs(drift) * 1e6 > MAX_PLAUSIBLE_DRIFT_PPM:
        raise TakeCaptureError(
            f"{name}: fitted drift {drift * 1e6:,.0f} ppm exceeds the "
            f"{MAX_PLAUSIBLE_DRIFT_PPM:,.0f} ppm physical bound (head offset "
            f"{head_off:.3f}s at corr {head_corr:.2f}, tail offset "
            f"{tail_off:.3f}s at corr {tail_corr:.2f}). No recorder clock runs "
            f"this far off -- one of the slates mis-locked, most likely because "
            f"the channel stopped before the slate was struck. Refusing to sync."
        )

    residuals = []
    for m in mid_slates_s:
        measured, _ = estimate_offset(ref, chan, sr, m)
        predicted = head_off + drift * (m - head_slate_s)
        residuals.append(measured - predicted)

    return ChannelSync(
        name=name,
        ref_head_s=head_slate_s,
        ref_tail_s=tail_slate_s,
        head_offset_s=head_off,
        tail_offset_s=tail_off,
        drift_rate=drift,
        head_corr=head_corr,
        tail_corr=tail_corr,
        mid_residuals_s=tuple(residuals),
    )


# --- manifest -> take -------------------------------------------------------


def load_manifest(path: Path) -> dict:
    """Read a session manifest. Schema:

    {"take_id": "...", "reference_channel": "ref",
     "channels": {"ref": "ref.wav", "phone_a": "a.wav"},
     "head_search_s": [0, 20], "tail_search_s": [-20, null],
     "mid_search_s": [[140, 160]]}      # optional

    Channel paths are resolved relative to the manifest's own directory, so a
    session folder moves without breaking.
    """
    if not path.exists():
        raise TakeCaptureError(f"manifest missing: {path}")
    body = json.loads(path.read_text())
    for key in ("take_id", "reference_channel", "channels"):
        if key not in body:
            raise TakeCaptureError(f"{path}: manifest has no {key!r}")
    if body["reference_channel"] not in body["channels"]:
        raise TakeCaptureError(
            f"{path}: reference_channel {body['reference_channel']!r} is not in "
            f"channels {sorted(body['channels'])}"
        )
    body["channels"] = {
        name: (path.parent / rel) for name, rel in body["channels"].items()
    }
    return body


def sync_take(manifest_path: Path) -> Take:
    """Manifest -> a Take with every non-reference channel mapped onto the
    reference clock.

    Raises:
        TakeCaptureError: any channel is missing, any slate is undetectable, any
            correlation is below the floor, or sample rates disagree. The take
            is excluded whole -- partial takes are not produced, because a take
            missing the channel under test is not a take.
    """
    body = load_manifest(manifest_path)
    ref_name = body["reference_channel"]

    loaded = {name: load_channel(p) for name, p in body["channels"].items()}
    rates = {name: sr for name, (_, sr) in loaded.items()}
    if len(set(rates.values())) != 1:
        raise TakeCaptureError(
            f"{manifest_path}: channels disagree on sample rate: {rates}. "
            f"Resample before syncing rather than letting the lag units drift."
        )
    sr = next(iter(rates.values()))
    ref = loaded[ref_name][0]
    ref_dur = len(ref) / sr

    def _window(spec, default):
        lo, hi = spec if spec else default
        lo = ref_dur + lo if lo is not None and lo < 0 else (lo or 0.0)
        hi = ref_dur if hi is None else (ref_dur + hi if hi < 0 else hi)
        return lo, hi

    head_lo, head_hi = _window(body.get("head_search_s"), (0.0, 20.0))
    tail_lo, tail_hi = _window(body.get("tail_search_s"), (-20.0, None))

    head = detect_slate(ref, sr, head_lo, head_hi)
    tail = detect_slate(ref, sr, tail_lo, tail_hi)
    mids = tuple(
        detect_slate(ref, sr, *_window(w, (0.0, ref_dur)))
        for w in body.get("mid_search_s", [])
    )

    syncs = {}
    for name, (chan, _) in loaded.items():
        if name == ref_name:
            continue
        syncs[name] = sync_channel(name, ref, chan, sr, head, tail, mids)

    return Take(
        take_id=body["take_id"],
        reference_channel=ref_name,
        channels=body["channels"],
        syncs=syncs,
    )


# --- intake: raw session exports -> validated takes -------------------------

# A position channel's name carries the POSITION and nothing else. The recording
# guide's flat filename scheme (``p2_s01_t007__p1_phone.wav``) used ``p2`` for
# the PHASE and ``p1`` for the POSITION in a single name, which is one reading
# slip away from attributing a take to the wrong position -- and position is the
# factor #148 is subtracting on. Positions are named here and only here.
POSITION_NAME_RE = re.compile(r"^pos[1-9][0-9]*_[a-z0-9]+$")

# take_id and session_id become directory names.
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# Head, tail, and at least one mid. Two slates fit a line exactly and cannot
# test it -- see the module docstring. The third clap cannot be added later.
MIN_SLATES_PER_TAKE = 3

_TAKE_KEYS = ("take_id", "piece", "behavior", "sources", "mid_search_s")


def _materialize_channel(name: str, src: Path, dst: Path) -> bool:
    """Put one raw export at ``dst`` as a WAV. Returns True if it was converted.

    A WAV source is copied byte-for-byte, so the reference channel reaches sync
    and the transcriber exactly as the rig wrote it. Anything else -- phone
    voice memos are m4a -- goes through ffmpeg with **no** ``-ar`` and **no**
    ``-ac``, so a wrong sample rate SURVIVES intake and is caught by the rate
    check below. Resampling here would quietly repair the one rig
    misconfiguration that is still cheap to fix while the rig is up.
    """
    if not src.exists():
        raise TakeCaptureError(
            f"channel {name!r}: no file at {src}. Every channel the session "
            f"manifest names must exist before the rig comes down -- a channel "
            f"first discovered missing at sync time cannot be re-recorded."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".wav":
        shutil.copyfile(src, dst)
        return False
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(src), "-c:a", "pcm_s24le", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        raise TakeCaptureError(
            f"channel {name!r}: ffmpeg failed converting {src}: {r.stderr[-400:]}"
        )
    return True


def _channel_rate(path: Path) -> int:
    import soundfile as sf

    try:
        return int(sf.info(str(path)).samplerate)
    except Exception as exc:
        raise TakeCaptureError(
            f"could not read {path}: {type(exc).__name__}: {exc}"
        ) from exc


def load_session_manifest(path: Path) -> dict:
    """Read and validate a SESSION manifest -- the recorder's notes for a whole
    sitting, written before intake and pointing at the raw exports:

    {"session_id": "s01",
     "rig_hash": "sha256:9f2c...",
     "sample_rate": 48000,
     "reference_channel": "ref",
     "channels": ["ref", "pos1_phone", "pos2_ipad", "pos3_laptop"],
     "takes": [
       {"take_id": "phase1_t007",
        "piece": "chopin_op28_no4",
        "behavior": "calibration",
        "score_midi": "scores/op28_4.mid",
        "sources": {"ref": "raw/GB_007_ref.wav",
                    "pos1_phone": "raw/memo_12.m4a",
                    "pos2_ipad": "raw/rec-0007.m4a",
                    "pos3_laptop": "raw/laptop_007.wav"},
        "mid_search_s": [[80, 110]]}]}

    Every path is relative to the manifest's own directory.

    These are structural refusals: a manifest typo is fixed in a second and the
    session re-run, so the FIRST one raises rather than being collected. Disk
    and audio failures, which need someone to walk back to the piano, are
    collected instead -- see ``intake_session``.
    """
    if not path.exists():
        raise TakeCaptureError(f"session manifest missing: {path}")
    body = json.loads(path.read_text())
    for key in (
        "session_id",
        "rig_hash",
        "sample_rate",
        "reference_channel",
        "channels",
        "takes",
    ):
        if key not in body:
            raise TakeCaptureError(f"{path}: session manifest has no {key!r}")

    if not SAFE_ID_RE.match(str(body["session_id"])):
        raise TakeCaptureError(
            f"{path}: session_id {body['session_id']!r} is not a safe directory "
            f"name (letters, digits, '_' and '-')"
        )
    if not str(body["rig_hash"]).strip():
        raise TakeCaptureError(
            f"{path}: rig_hash is empty. Every take carries the hash of the rig "
            f"that recorded it; a subtraction across two rigs is not a "
            f"subtraction across two positions."
        )

    channels = list(body["channels"])
    ref_name = body["reference_channel"]
    if ref_name not in channels:
        raise TakeCaptureError(
            f"{path}: reference_channel {ref_name!r} is not in channels {channels}"
        )
    for name in channels:
        if name == ref_name:
            if not SAFE_ID_RE.match(name):
                raise TakeCaptureError(
                    f"{path}: reference channel {name!r} is not a safe name"
                )
            continue
        if not POSITION_NAME_RE.match(name):
            raise TakeCaptureError(
                f"{path}: channel {name!r} does not name a position. Positions "
                f"are 'pos1_phone' / 'pos2_ipad' / 'pos3_laptop' -- a 'p1'-style "
                f"name collides with the 'p1'/'p2' phase tag and one reading "
                f"slip attributes a take to the wrong position."
            )

    seen: set[str] = set()
    for entry in body["takes"]:
        for key in _TAKE_KEYS:
            if key not in entry:
                raise TakeCaptureError(
                    f"{path}: take {entry.get('take_id', '?')!r} has no {key!r}"
                )
        tid = entry["take_id"]
        if not SAFE_ID_RE.match(str(tid)):
            raise TakeCaptureError(
                f"{path}: take_id {tid!r} is not a safe directory name"
            )
        if tid in seen:
            raise TakeCaptureError(
                f"{path}: take_id {tid!r} appears twice. Two takes under one id "
                f"overwrite each other's audio, and the survivor is silent "
                f"about which performance it was."
            )
        seen.add(tid)
        if sorted(entry["sources"]) != sorted(channels):
            raise TakeCaptureError(
                f"{path}: take {tid!r} declares channels "
                f"{sorted(entry['sources'])}, but the session records "
                f"{sorted(channels)}. A take short one position is not a take."
            )
        if len(entry["mid_search_s"]) < MIN_SLATES_PER_TAKE - 2:
            raise TakeCaptureError(
                f"{path}: take {tid!r} declares no mid-slate window. "
                f"{MIN_SLATES_PER_TAKE} claps per take: head and tail fit the "
                f"drift line exactly and cannot test it, and the mid clap "
                f"cannot be added after the session."
            )
        if entry["behavior"] == "calibration" and "score_midi" not in entry:
            raise TakeCaptureError(
                f"{path}: calibration take {tid!r} has no 'score_midi'. G-OOD-0 "
                f"is recall against a KNOWN score; calibration_recall refuses to "
                f"score it without one."
            )
    return body


def intake_take(entry: dict, session: dict, src_root: Path, dest_root: Path) -> dict:
    """One raw take -> a take directory ``sync_take`` and ``calibration_recall``
    can both read, with every channel synced as proof it is usable.

    Writes ``<dest_root>/<take_id>/`` holding one WAV per channel named for its
    position, the calibration score MIDI if there is one, and ``take.json``.
    The directory is self-contained so a session folder moves without breaking,
    the same property ``load_manifest`` already gives.

    Raises:
        TakeCaptureError: any channel is missing or unreadable, any channel is
            not at the session's declared sample rate, or the take does not
            sync -- slates too close, correlation below the floor, implausible
            drift. Each is unrecoverable once the rig is packed up.
    """
    tid = entry["take_id"]
    ref_name = session["reference_channel"]
    take_dir = dest_root / tid
    take_dir.mkdir(parents=True, exist_ok=True)

    channels: dict[str, dict] = {}
    for name, rel in sorted(entry["sources"].items()):
        dst = take_dir / f"{name}.wav"
        converted = _materialize_channel(name, src_root / rel, dst)
        rate = _channel_rate(dst)
        if rate != session["sample_rate"]:
            raise TakeCaptureError(
                f"{tid}: channel {name!r} is {rate} Hz, not the session's "
                f"{session['sample_rate']} Hz (from {rel}). Set the rate on the "
                f"DEVICE and re-record this take -- intake will not resample it "
                f"into agreement, because that hides a rig setting that is still "
                f"cheap to fix. iOS voice-memo apps default to 44100."
            )
        channels[name] = {
            "source": str(rel),
            "sample_rate": rate,
            "converted": converted,
        }

    body = {
        "take_id": tid,
        "session_id": session["session_id"],
        "rig_hash": session["rig_hash"],
        "piece": entry["piece"],
        "behavior": entry["behavior"],
        "reference_channel": ref_name,
        "channels": {name: f"{name}.wav" for name in entry["sources"]},
        "mid_search_s": entry["mid_search_s"],
    }
    for key in ("head_search_s", "tail_search_s"):
        if key in entry:
            body[key] = entry[key]
    if entry["behavior"] == "calibration":
        score_src = src_root / entry["score_midi"]
        if not score_src.exists():
            raise TakeCaptureError(
                f"{tid}: score MIDI missing at {score_src}. A calibration take "
                f"without its score cannot be scored against anything."
            )
        shutil.copyfile(score_src, take_dir / "score.mid")
        body["score_midi"] = "score.mid"

    manifest = take_dir / "take.json"
    manifest.write_text(json.dumps(body, indent=1) + "\n")

    take = sync_take(manifest)
    return {
        "take_id": tid,
        "piece": entry["piece"],
        "behavior": entry["behavior"],
        "manifest": str(manifest),
        "channels": channels,
        "syncs": {
            name: {
                "head_offset_s": s.head_offset_s,
                "drift_ppm": s.drift_ppm,
                "head_corr": s.head_corr,
                "tail_corr": s.tail_corr,
                # None means linearity is UNTESTED. Reported, never gated on: a
                # residual bar picked before any real room recording exists
                # would be invented, not measured.
                "max_mid_residual_s": s.max_mid_residual_s,
            }
            for name, s in take.syncs.items()
        },
    }


def intake_session(manifest_path: Path, dest_root: Path) -> dict:
    """Whole session in, validated takes out. Writes ``intake_report.json``.

    Every take is attempted even after one fails, because the operator wants
    the entire punch-list before walking back to the piano rather than one
    problem per round trip. The report is written either way, and a session
    with any failure then RAISES: a partially intaken session is not a session,
    and a silent failure list is how a missing position becomes a hole in the
    subtraction weeks later.
    """
    session = load_session_manifest(manifest_path)
    src_root = manifest_path.parent
    dest_root.mkdir(parents=True, exist_ok=True)

    takes, failures = [], []
    for entry in session["takes"]:
        try:
            takes.append(intake_take(entry, session, src_root, dest_root))
        except TakeCaptureError as exc:
            failures.append({"take_id": entry["take_id"], "error": str(exc)})

    report = {
        "session_id": session["session_id"],
        "rig_hash": session["rig_hash"],
        "sample_rate": session["sample_rate"],
        "reference_channel": session["reference_channel"],
        "dest_root": str(dest_root),
        "n_takes_declared": len(session["takes"]),
        "n_takes_ok": len(takes),
        "takes": takes,
        "failures": failures,
    }
    report_path = dest_root / "intake_report.json"
    report_path.write_text(json.dumps(report, indent=1) + "\n")

    if failures:
        raise TakeCaptureError(
            f"{len(failures)} of {len(session['takes'])} takes failed intake "
            f"(report: {report_path}):\n"
            + "\n".join(f"  {f['take_id']}: {f['error']}" for f in failures)
        )
    return report


def _format_intake(report: dict) -> str:
    lines = [
        f"session {report['session_id']}  rig {report['rig_hash']}",
        f"{report['n_takes_ok']}/{report['n_takes_declared']} takes intaken "
        f"at {report['sample_rate']} Hz -> {report['dest_root']}",
        "",
    ]
    for t in report["takes"]:
        lines.append(f"{t['take_id']}  {t['behavior']}  {t['piece']}")
        for name, s in sorted(t["syncs"].items()):
            resid = s["max_mid_residual_s"]
            resid_s = "UNTESTED" if resid is None else f"{resid * 1e3:.1f} ms"
            lines.append(
                f"    {name:<16} offset {s['head_offset_s']:+7.3f}s  "
                f"drift {s['drift_ppm']:+8.1f} ppm  "
                f"corr {s['head_corr']:.2f}/{s['tail_corr']:.2f}  "
                f"mid residual {resid_s}"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Intake a raw recording session into validated takes (#148)"
    )
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args()
    print(_format_intake(intake_session(args.session, args.dest)))


if __name__ == "__main__":
    main()
