# model/tests/follower_eval/test_take_capture.py
"""Sync validation for take_capture (issue #148).

Every accuracy claim here is a RECOVERY test: a channel is synthesized from the
reference with a KNOWN offset and a KNOWN drift rate, and the test asserts on
the error against those injected values. Nothing is checked against the
module's own output.

The channel is not a copy of the reference. It is band-shaped and gain-varied
(a crude stand-in for a different mic and for phone AGC) so a correlator that
only works on identical waveforms fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from follower_eval import take_capture as tc

SR = 8000
DURATION_S = 180.0
HEAD_SLATE_S = 3.0
MID_SLATE_S = 90.0
TAIL_SLATE_S = 175.0


def _clap(sr: int, length_s: float = 0.08) -> np.ndarray:
    """Broadband transient with a sharp attack and exponential decay."""
    n = int(sr * length_s)
    rng = np.random.default_rng(0)
    return (rng.standard_normal(n) * np.exp(-np.linspace(0, 12, n))).astype(np.float64)


def make_reference(
    sr: int = SR, duration_s: float = DURATION_S, slates=(HEAD_SLATE_S, TAIL_SLATE_S)
) -> np.ndarray:
    """A reference channel: quiet 'playing' plus clap slates at known times."""
    rng = np.random.default_rng(133)
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    # sustained tones with note-rate amplitude modulation -- loud, but never as
    # sharp-onset as a clap, so it must not win slate detection.
    x = 0.10 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    x += 0.06 * np.sin(2 * np.pi * 330 * t)
    x += 0.01 * rng.standard_normal(n)
    clap = _clap(sr)
    for s in slates:
        i = int(s * sr)
        x[i : i + len(clap)] += clap
    return x


def make_channel(
    ref: np.ndarray, sr: int, offset_s: float, drift_rate: float, seed: int = 7
) -> np.ndarray:
    """Resample the reference onto a channel clock with a KNOWN offset and drift.

    The forward map being injected is, by definition,

        t_chan = t_ref + offset_s + drift_rate * (t_ref - HEAD_SLATE_S)

    so the channel's sample k must be read from reference time

        t_ref = (k/sr - offset_s + drift_rate*HEAD_SLATE_S) / (1 + drift_rate)

    which is derived here from that definition rather than borrowed from the
    module, so a sign error in either one shows up as a recovery failure. It
    did: an earlier version of this fixture flipped the drift term's sign,
    injecting an extra 2*drift*HEAD_SLATE_S of offset (3.1 ms at 500 ppm) and
    failing the 2 ms bar.

    Also applies a first-order lowpass (different mic) and a slow gain
    modulation (crude AGC), so recovery cannot rely on identical waveforms.
    """
    rng = np.random.default_rng(seed)
    n_ref = len(ref)
    # The channel must run long enough on ITS clock to contain the reference's
    # last slate. A channel started 12 s late and truncated to the reference's
    # length would simply not have recorded the tail clap -- an earlier version
    # did that and the correlator locked onto a spurious transient instead.
    n = n_ref + int(sr * (abs(offset_s) + 2.0))
    k = np.arange(n)
    ref_time = (k / sr - offset_s + drift_rate * HEAD_SLATE_S) / (1.0 + drift_rate)
    y = np.interp(ref_time, np.arange(n_ref) / sr, ref, left=0.0, right=0.0)

    # different mic: one-pole lowpass
    a = 0.35
    out = np.empty_like(y)
    acc = 0.0
    for i in range(n):  # noqa: B007 - explicit, n is small in tests
        acc = a * y[i] + (1 - a) * acc
        out[i] = acc
    # crude AGC: slow gain wobble + a little independent noise
    t = k / sr
    out *= 0.8 + 0.35 * np.sin(2 * np.pi * 0.05 * t)
    out += 0.004 * rng.standard_normal(n)
    return out


@pytest.fixture(scope="module")
def reference():
    return make_reference(slates=(HEAD_SLATE_S, MID_SLATE_S, TAIL_SLATE_S))


# --- slate detection --------------------------------------------------------


def test_detect_slate_finds_the_head_clap_not_the_loud_playing(reference):
    found = tc.detect_slate(reference, SR, 0.0, 20.0)
    assert abs(found - HEAD_SLATE_S) < 0.01


def test_detect_slate_finds_the_tail_clap(reference):
    found = tc.detect_slate(reference, SR, DURATION_S - 20.0, DURATION_S)
    assert abs(found - TAIL_SLATE_S) < 0.01


def test_detect_slate_raises_when_the_window_holds_only_playing(reference):
    with pytest.raises(tc.TakeCaptureError, match="no clap slate"):
        tc.detect_slate(reference, SR, 30.0, 60.0)


def test_detect_slate_raises_on_an_empty_window(reference):
    with pytest.raises(tc.TakeCaptureError, match="empty or shorter"):
        tc.detect_slate(reference, SR, 10.0, 10.01)


# --- offset and drift recovery, against the injected truth -------------------


@pytest.mark.parametrize(
    "offset_s,drift_ppm",
    [
        (0.0, 0.0),  # identical clocks
        (2.5, 0.0),  # pure offset
        (0.0, 200.0),  # pure drift (200 ppm ~ 34 ms over 180 s)
        (-1.25, -150.0),  # channel started early, clock runs slow
        (12.0, 500.0),  # large hand-start gap, sloppy crystal
    ],
)
def test_recovers_injected_offset_and_drift(reference, offset_s, drift_ppm):
    drift = drift_ppm * 1e-6
    chan = make_channel(reference, SR, offset_s, drift)
    sync = tc.sync_channel("phone", reference, chan, SR, HEAD_SLATE_S, TAIL_SLATE_S)

    offset_err_ms = abs(sync.head_offset_s - offset_s) * 1000
    drift_err_ppm = abs(sync.drift_ppm - drift_ppm)
    # 2 ms and 25 ppm: 25 ppm over a 180 s take is 4.5 ms of accumulated skew,
    # far inside the beat resolution #148 claims.
    assert offset_err_ms < 2.0, f"offset error {offset_err_ms:.2f} ms"
    assert drift_err_ppm < 25.0, f"drift error {drift_err_ppm:.1f} ppm"


def test_recovered_map_puts_a_known_event_on_the_reference_clock(reference):
    """End-to-end: the fitted map must move a mid-take event to within a few ms
    of where the reference has it."""
    offset_s, drift = 4.0, 300e-6
    chan = make_channel(reference, SR, offset_s, drift)
    sync = tc.sync_channel("phone", reference, chan, SR, HEAD_SLATE_S, TAIL_SLATE_S)

    # the mid clap sits at MID_SLATE_S on the reference clock
    chan_time_of_mid = tc.detect_slate(chan, SR, MID_SLATE_S - 5, MID_SLATE_S + 15)
    assert abs(float(sync.to_ref_time(chan_time_of_mid)) - MID_SLATE_S) < 0.01


def test_to_channel_time_and_to_ref_time_are_inverses():
    sync = tc.ChannelSync(
        name="p",
        ref_head_s=3.0,
        ref_tail_s=175.0,
        head_offset_s=2.5,
        tail_offset_s=2.55,
        drift_rate=2.9e-4,
        head_corr=0.9,
        tail_corr=0.9,
    )
    for t in (0.0, 3.0, 90.0, 175.0):
        assert float(sync.to_ref_time(sync.to_channel_time(t))) == pytest.approx(
            t, abs=1e-6
        )


# --- the mid-slate linearity check ------------------------------------------


def test_mid_slate_residual_is_small_when_the_clock_really_is_linear(reference):
    chan = make_channel(reference, SR, 2.0, 250e-6)
    sync = tc.sync_channel(
        "phone",
        reference,
        chan,
        SR,
        HEAD_SLATE_S,
        TAIL_SLATE_S,
        mid_slates_s=(MID_SLATE_S,),
    )
    assert sync.max_mid_residual_s is not None
    assert sync.max_mid_residual_s < 0.005


def test_max_mid_residual_is_none_without_a_mid_slate_meaning_untested(reference):
    """None must mean 'linearity was never tested', not 'linearity holds'. Two
    slates fit a line through two points exactly, so a zero residual would be
    an arithmetic identity rather than evidence."""
    chan = make_channel(reference, SR, 2.0, 250e-6)
    sync = tc.sync_channel("phone", reference, chan, SR, HEAD_SLATE_S, TAIL_SLATE_S)
    assert sync.max_mid_residual_s is None


def test_mid_slate_residual_exposes_a_nonlinear_clock(reference):
    """Inject a clock that is NOT affine: the head+tail fit still passes through
    both slates perfectly, and only the held-out mid slate can reveal it."""
    n = len(reference)
    k = np.arange(n)
    t = k / SR
    # a sinusoidal timing wobble that vanishes at both slates but not between
    span = TAIL_SLATE_S - HEAD_SLATE_S
    wobble = 0.05 * np.sin(np.pi * (t - HEAD_SLATE_S) / span)
    ref_time = t - 2.0 - wobble
    chan = np.interp(ref_time, t, reference, left=0.0, right=0.0)

    sync = tc.sync_channel(
        "phone",
        reference,
        chan,
        SR,
        HEAD_SLATE_S,
        TAIL_SLATE_S,
        mid_slates_s=(MID_SLATE_S,),
    )
    # head and tail offsets agree (the wobble is zero there) -> drift ~ 0
    assert abs(sync.drift_ppm) < 100
    # but the mid slate is displaced by ~the wobble amplitude
    assert sync.max_mid_residual_s > 0.03


# --- refusals ---------------------------------------------------------------


def test_sync_channel_raises_when_slates_are_too_close(reference):
    chan = make_channel(reference, SR, 1.0, 0.0)
    with pytest.raises(tc.TakeCaptureError, match="below the .* floor"):
        tc.sync_channel("phone", reference, chan, SR, 3.0, 13.0)


def test_a_channel_that_stopped_before_the_tail_slate_raises(reference):
    """The silent-failure path this bound exists for. When the tail clap was
    never recorded, the correlator locks onto an unrelated transient and scores
    0.512 -- ABOVE the 0.50 floor -- returning an offset 47 s from the head
    offset. The correlation floor alone accepts it; the physical drift bound is
    what catches it."""
    truncated = make_channel(reference, SR, 12.0, 500e-6)[: len(reference)]
    with pytest.raises(tc.TakeCaptureError, match="physical bound"):
        tc.sync_channel("phone", reference, truncated, SR, HEAD_SLATE_S, TAIL_SLATE_S)


def test_the_correlation_floor_alone_would_have_accepted_that_mis_lock(reference):
    """Pins WHY the drift bound is needed rather than a higher MIN_SLATE_CORR:
    the mis-lock is only just under the genuine matches, so tuning the
    correlation floor on synthetic audio would be guesswork."""
    truncated = make_channel(reference, SR, 12.0, 500e-6)[: len(reference)]
    _, corr = tc.estimate_offset(reference, truncated, SR, TAIL_SLATE_S)
    assert tc.MIN_SLATE_CORR < corr < 0.6


def test_sliding_energy_equals_the_convolution_it_replaces():
    """The correlation denominator was a general O(N*width) convolution, which
    at the rig's 48 kHz costs ~105 s per slate per channel -- intake's answer
    would arrive hours after the rig came down. The prefix sum is O(N); this
    pins that it computes the SAME quantity, so the speedup did not buy a
    different correlation."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(20_000)
    for width in (1, 37, 4096, len(x)):
        expected = np.convolve(x**2, np.ones(width), mode="valid")
        got = tc._sliding_energy(x, width)
        assert got.shape == expected.shape
        assert np.allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_estimate_offset_raises_on_an_untrustworthy_correlation(reference):
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(len(reference)) * 0.5
    with pytest.raises(tc.TakeCaptureError, match="below the .* floor"):
        tc.estimate_offset(reference, noise, SR, HEAD_SLATE_S)


# --- manifest / missing channel ---------------------------------------------


def _write_session(tmp_path: Path, channels: dict, present: set, **extra) -> Path:
    ref = make_reference(slates=(HEAD_SLATE_S, TAIL_SLATE_S))
    for name, rel in channels.items():
        if name not in present:
            continue
        x = ref if name == "ref" else make_channel(ref, SR, 1.5, 100e-6)
        sf.write(str(tmp_path / rel), x.astype(np.float32), SR)
    manifest = tmp_path / "take.json"
    manifest.write_text(
        json.dumps(
            {
                "take_id": "t001",
                "reference_channel": "ref",
                "channels": channels,
                **extra,
            }
        )
    )
    return manifest


def test_sync_take_maps_every_phone_channel(tmp_path: Path):
    chans = {"ref": "ref.wav", "phone_a": "a.wav"}
    m = _write_session(tmp_path, chans, present=set(chans))
    take = sorted(tc.sync_take(m).syncs)
    assert take == ["phone_a"]


def test_missing_channel_raises_and_never_falls_back_to_the_reference(tmp_path: Path):
    """THE hard rule from #148: substituting the reference for an absent phone
    channel would report a clean-channel number as a phone-channel number --
    exactly the degradation the eval exists to measure. Same precedent as
    asap_audio.py refusing to fall back to MIDI."""
    chans = {"ref": "ref.wav", "phone_a": "a.wav"}
    m = _write_session(tmp_path, chans, present={"ref"})

    with pytest.raises(tc.TakeCaptureError) as exc:
        tc.sync_take(m)
    assert "missing" in str(exc.value)
    assert "Refusing to substitute the reference channel" in str(exc.value)


def test_manifest_raises_when_reference_channel_is_not_a_channel(tmp_path: Path):
    (tmp_path / "take.json").write_text(
        json.dumps(
            {
                "take_id": "t",
                "reference_channel": "nope",
                "channels": {"ref": "ref.wav"},
            }
        )
    )
    with pytest.raises(tc.TakeCaptureError, match="is not in channels"):
        tc.load_manifest(tmp_path / "take.json")


def test_sync_take_raises_on_mismatched_sample_rates(tmp_path: Path):
    ref = make_reference(slates=(HEAD_SLATE_S, TAIL_SLATE_S))
    sf.write(str(tmp_path / "ref.wav"), ref.astype(np.float32), SR)
    sf.write(
        str(tmp_path / "a.wav"),
        make_channel(ref, SR, 1.0, 0.0).astype(np.float32),
        SR * 2,
    )
    m = tmp_path / "take.json"
    m.write_text(
        json.dumps(
            {
                "take_id": "t",
                "reference_channel": "ref",
                "channels": {"ref": "ref.wav", "phone_a": "a.wav"},
            }
        )
    )
    with pytest.raises(tc.TakeCaptureError, match="disagree on sample rate"):
        tc.sync_take(m)


# --- intake -----------------------------------------------------------------
# Intake exists to move sync's refusals earlier in WALL-CLOCK time: each failure
# below is unrecoverable once the rig is packed up, so every test here asserts
# that the refusal happens against the raw exports rather than at sync time.

POS = "pos1_phone"
MID_WINDOW = [MID_SLATE_S - 10, MID_SLATE_S + 10]


def _write_raw_take(root: Path, tid: str, *, rates=None, skip=(), fmt="wav"):
    ref = make_reference(slates=(HEAD_SLATE_S, MID_SLATE_S, TAIL_SLATE_S))
    rates = rates or {}
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, x in (("ref", ref), (POS, make_channel(ref, SR, 1.5, 100e-6))):
        if name in skip:
            continue
        suffix = "wav" if name == "ref" else fmt
        sf.write(
            str(raw / f"{tid}_{name}.{suffix}"),
            x.astype(np.float32),
            rates.get(name, SR),
        )


def _take_entry(tid: str, *, behavior="practice", fmt="wav", **extra) -> dict:
    entry = {
        "take_id": tid,
        "piece": "test_piece",
        "behavior": behavior,
        "sources": {
            "ref": f"raw/{tid}_ref.wav",
            POS: f"raw/{tid}_{POS}.{fmt}",
        },
        "mid_search_s": [MID_WINDOW],
    }
    entry.update(extra)
    return entry


def _write_session_manifest(root: Path, entries: list[dict], **overrides) -> Path:
    body = {
        "session_id": "s01",
        "rig_hash": "sha256:abc123",
        "sample_rate": SR,
        "reference_channel": "ref",
        "channels": ["ref", POS],
        "takes": entries,
    }
    body.update(overrides)
    path = root / "session.json"
    path.write_text(json.dumps(body))
    return path


def test_intake_writes_a_take_directory_sync_take_can_read(tmp_path: Path):
    """The contract: manifest in, a self-contained take directory out that the
    downstream readers consume unchanged."""
    _write_raw_take(tmp_path, "t001")
    session = _write_session_manifest(tmp_path, [_take_entry("t001")])

    report = tc.intake_session(session, tmp_path / "takes")

    assert report["n_takes_ok"] == 1
    body = tc.load_manifest(tmp_path / "takes" / "t001" / "take.json")
    assert sorted(body["channels"]) == sorted(["ref", POS])
    assert all(p.exists() for p in body["channels"].values())
    assert body["rig_hash"] == "sha256:abc123"
    assert body["piece"] == "test_piece"


def test_intake_syncs_each_channel_and_reports_the_held_out_mid_residual(
    tmp_path: Path,
):
    """Intake's deep check IS sync_take -- a take that cannot be put on one
    clock fails while it can still be re-recorded. The injected drift is
    100 ppm and the mid slate is held out of the fit, so its residual is a
    measurement of linearity rather than None (which would mean UNTESTED)."""
    _write_raw_take(tmp_path, "t001")
    session = _write_session_manifest(tmp_path, [_take_entry("t001")])

    sync = tc.intake_session(session, tmp_path / "takes")["takes"][0]["syncs"][POS]

    assert abs(sync["drift_ppm"] - 100.0) < 20.0
    assert sync["max_mid_residual_s"] is not None
    assert sync["max_mid_residual_s"] < 0.005


def test_a_channel_with_no_file_fails_at_intake_not_hours_later_at_sync(
    tmp_path: Path,
):
    _write_raw_take(tmp_path, "t001", skip={POS})
    session = _write_session_manifest(tmp_path, [_take_entry("t001")])

    with pytest.raises(tc.TakeCaptureError) as exc:
        tc.intake_session(session, tmp_path / "takes")
    assert "cannot be re-recorded" in str(exc.value)
    report = json.loads((tmp_path / "takes" / "intake_report.json").read_text())
    assert [f["take_id"] for f in report["failures"]] == ["t001"]


def test_a_channel_at_the_wrong_rate_is_refused_not_resampled(tmp_path: Path):
    """iOS voice-memo apps default to 44.1 kHz while the rig runs at 48. sync
    would catch the disagreement, but only after the session; and resampling it
    into agreement here would repair the rig setting instead of reporting it."""
    _write_raw_take(tmp_path, "t001", rates={POS: 44100})
    session = _write_session_manifest(tmp_path, [_take_entry("t001")])

    with pytest.raises(tc.TakeCaptureError) as exc:
        tc.intake_session(session, tmp_path / "takes")
    assert "44100" in str(exc.value)
    assert "will not resample" in str(exc.value)


def test_a_converted_channel_is_not_resampled_into_agreement_either(tmp_path: Path):
    """The non-WAV branch goes through ffmpeg, which is exactly where an '-ar'
    would silently fix the rate. A wrong rate must survive the conversion."""
    _write_raw_take(tmp_path, "t001", rates={POS: 44100}, fmt="flac")
    session = _write_session_manifest(tmp_path, [_take_entry("t001", fmt="flac")])

    with pytest.raises(tc.TakeCaptureError, match="44100"):
        tc.intake_session(session, tmp_path / "takes")


def test_a_non_wav_export_is_converted_and_still_syncs(tmp_path: Path):
    _write_raw_take(tmp_path, "t001", fmt="flac")
    session = _write_session_manifest(tmp_path, [_take_entry("t001", fmt="flac")])

    take = tc.intake_session(session, tmp_path / "takes")["takes"][0]

    assert take["channels"][POS]["converted"] is True
    assert take["channels"]["ref"]["converted"] is False
    assert take["channels"][POS]["sample_rate"] == SR
    assert POS in take["syncs"]


def test_a_take_with_only_two_claps_is_refused_while_a_third_can_still_be_added(
    tmp_path: Path,
):
    _write_raw_take(tmp_path, "t001")
    session = _write_session_manifest(tmp_path, [_take_entry("t001", mid_search_s=[])])

    with pytest.raises(tc.TakeCaptureError, match="cannot be added after"):
        tc.intake_session(session, tmp_path / "takes")


def test_a_p1_style_position_name_is_refused(tmp_path: Path):
    """'p2_s01_t007__p1_phone.wav' uses p2 for the PHASE and p1 for the
    POSITION. Position is the factor #148 subtracts on, so the two must not be
    one reading slip apart."""
    session = _write_session_manifest(tmp_path, [], channels=["ref", "p1_phone"])

    with pytest.raises(tc.TakeCaptureError, match="does not name a position"):
        tc.intake_session(session, tmp_path / "takes")


def test_a_duplicate_take_id_is_refused_before_it_overwrites_audio(tmp_path: Path):
    _write_raw_take(tmp_path, "t001")
    session = _write_session_manifest(
        tmp_path, [_take_entry("t001"), _take_entry("t001")]
    )

    with pytest.raises(tc.TakeCaptureError, match="appears twice"):
        tc.intake_session(session, tmp_path / "takes")


def test_a_take_short_one_position_is_refused(tmp_path: Path):
    """Distinct from a missing FILE: the manifest never named the channel, so
    nothing downstream would have looked for it."""
    entry = _take_entry("t001")
    del entry["sources"][POS]
    session = _write_session_manifest(tmp_path, [entry])

    with pytest.raises(tc.TakeCaptureError, match="short one position"):
        tc.intake_session(session, tmp_path / "takes")


def test_an_empty_rig_hash_is_refused(tmp_path: Path):
    session = _write_session_manifest(tmp_path, [], rig_hash="  ")

    with pytest.raises(tc.TakeCaptureError, match="rig_hash is empty"):
        tc.intake_session(session, tmp_path / "takes")


def test_a_calibration_take_without_a_score_is_refused(tmp_path: Path):
    _write_raw_take(tmp_path, "c001")
    session = _write_session_manifest(
        tmp_path, [_take_entry("c001", behavior="calibration")]
    )

    with pytest.raises(tc.TakeCaptureError, match="no 'score_midi'"):
        tc.intake_session(session, tmp_path / "takes")


def test_a_calibration_take_directory_is_readable_by_the_g_ood_0_gate(
    tmp_path: Path,
):
    """The two manifest keys calibration_recall raises on -- 'behavior' and
    'score_midi' -- are written by intake, and the score travels with the take
    so the session folder stays self-contained."""
    from follower_eval import calibration_recall as cr

    _write_raw_take(tmp_path, "c001")
    (tmp_path / "scores").mkdir()
    (tmp_path / "scores" / "op28_4.mid").write_bytes(b"MThd-not-parsed-here")
    session = _write_session_manifest(
        tmp_path,
        [
            _take_entry(
                "c001", behavior="calibration", score_midi="scores/op28_4.mid"
            )
        ],
    )

    tc.intake_session(session, tmp_path / "takes")

    take = cr.load_calibration_take(tmp_path / "takes" / "c001" / "take.json")
    assert take.ref_wav.exists()
    assert take.score_midi.exists()


def test_the_report_lists_every_failed_take_not_only_the_first(tmp_path: Path):
    """One round trip to the piano, not one per problem. A practice take is
    still attempted after a calibration take fails."""
    _write_raw_take(tmp_path, "t001", skip={POS})
    _write_raw_take(tmp_path, "t002", skip={POS})
    _write_raw_take(tmp_path, "t003")
    session = _write_session_manifest(
        tmp_path, [_take_entry(t) for t in ("t001", "t002", "t003")]
    )

    with pytest.raises(tc.TakeCaptureError) as exc:
        tc.intake_session(session, tmp_path / "takes")
    assert "t001" in str(exc.value) and "t002" in str(exc.value)

    report = json.loads((tmp_path / "takes" / "intake_report.json").read_text())
    assert [f["take_id"] for f in report["failures"]] == ["t001", "t002"]
    assert [t["take_id"] for t in report["takes"]] == ["t003"]
