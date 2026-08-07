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

SCOPE
-----
Sync is trustworthy at beat resolution, not note-onset resolution. Mic distance
alone puts a few milliseconds of genuine acoustic delay between channels, and
that delay is real rather than an error to remove. #148 makes no note-level
offset claims for this reason.
"""

from __future__ import annotations

import json
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
    denom = np.linalg.norm(template) * np.sqrt(
        np.convolve(segment**2, np.ones(len(template)), mode="valid")
    )
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
