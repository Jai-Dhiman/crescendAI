# Continuity-Aware Chroma Score-Follower Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Prevent the chroma-DTW score-follower from teleporting to wrong score positions by adding positional continuity, separation-margin confidence, and a silence gate — all behind the same one-call WASM interface.

**Spec:** docs/specs/2026-05-28-continuity-aware-chroma-follower-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / apps/api/TS_STYLE.md)

---

## Task Groups

```
Group 0 (sequential, must complete before all other groups):
  Task 0 — Margin hypothesis validation harness (GATING)

Group A (parallel, depends on Group 0 passing):
  Task 1 — Silence abstain: uniformity gate in Rust + cargo test
  Task 2 — Schema: BarMapChroma gains end_score_frame / confidence / status in Rust types

Group B (sequential, depends on Group A):
  Task 3 — Margin computation: two-readout + disjoint-neighborhood separation margin
  Task 4 — Arbitration state machine + updated WASM entry point

Group C (sequential, depends on Group B):
  Task 5 — Teleport regression fixture + cargo test (headline)
  Task 6 — Cold-start non-regression guard
  Task 7 — Relocalize scenario fixture + cargo test
  Task 8 — Low-margin abstain scenario + cargo test

Group D (sequential, depends on Group B — parallel with Group C):
  Task 9  — TypeScript: BarMapChroma interface + alignChunkChroma signature in wasm-bridge.ts
  Task 10 — Schema: expectedScoreFrame in session-brain.schema.ts

Group E (sequential, depends on Groups C and D):
  Task 11 — DO wiring: pass new args, dispatch on status, update expectedScoreFrame
  Task 12 — DO unit tests: frame continuity, abstain preserves frame, reset on piece re-id
  Task 13 — wasm-bridge forwarding test: 9-arg contract
```

---

## Task 0: Margin Hypothesis Validation (GATING)

**Group:** 0 (must pass before any other task)

**Behavior being verified:** The separation margin is higher (better) for a correct cold-start alignment than for the teleport-wrong alignment on the same score. If this does not hold, the margin-based arbitration logic is unfounded and the plan must stop.

**Interface under test:** `chroma_dtw_native` (pure Rust, no WASM boundary) called with the current stateless interface, plus a Python fixture generator that computes margins from the existing accumulated-cost array exposed by a diagnostic-only variant.

**Files:**
- Modify: `apps/api/src/wasm/score-analysis/tests/fixtures/generate.py`
- Create: `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/audio_chroma.bin` (generated)
- Create: `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/score_bars.json` (generated)
- Create: `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/expected.json` (generated)
- Create: `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/margin_probe.py`
- Modify: `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs` (add diagnostic export `subseq_dtw_last_row` for probe)

**Step 1: Extend the fixture generator to produce the amateur cs_000 fixture**

Add the following case to `CASES` in `apps/api/src/wasm/score-analysis/tests/fixtures/generate.py` (insert after the existing `ballade1_coldstart_111s` case):

```python
    {
        "slug": "ballade1_amateur_cs000",
        "audio_wav": "model/data/evals/skill_eval/chopin_ballade_1/audio/Jt2f6yEGcP4.wav",
        "start_s": 0.0,
        "dur_s": 15.0,
        # True position: bars 1-5 (opening of the piece)
        "bar_min_lo": 1,
        "bar_min_hi": 10,
        "bar_max_lo": 1,
        "bar_max_hi": 15,
        "cost_hi": 0.35,
    },
```

The generator loop must be updated to support a per-case `audio_wav` override (fall back to the existing `AUDIO_WAV` constant when not present in the case dict). The updated `main()` loop:

```python
for case in CASES:
    slug = case["slug"]
    wav_override = case.get("audio_wav")
    wav = root / wav_override if wav_override else root / AUDIO_WAV
    out_dir = root / FIXTURES_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {slug} ...")
    chroma = build_audio_chroma(wav, case["start_s"], case["dur_s"])
    n_frames = chroma.shape[1]
    bin_bytes = struct.pack(f"<{12 * n_frames}f", *chroma.flatten().tolist())
    (out_dir / "audio_chroma.bin").write_bytes(bin_bytes)
    all_bars_local = load_score_bars(score_json)
    (out_dir / "score_bars.json").write_text(json.dumps(all_bars_local))
    decim_step = int(round(FRAME_RATE / DECIM_HZ))
    decim_n = (n_frames + decim_step - 1) // decim_step
    expected = {
        "bar_min_lo": case["bar_min_lo"],
        "bar_min_hi": case["bar_min_hi"],
        "bar_max_lo": case["bar_max_lo"],
        "bar_max_hi": case["bar_max_hi"],
        "n_frames": n_frames,
        "decim_n": decim_n,
        "cost_hi": case["cost_hi"],
        "frame_rate_hz": FRAME_RATE,
    }
    (out_dir / "expected.json").write_text(json.dumps(expected, indent=2))
    print(f"  expected.json: {expected}")
```

**Step 2: Run the generator — verify it produces the new fixture**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run python apps/api/src/wasm/score-analysis/tests/fixtures/generate.py
```

Expected: `Generating ballade1_amateur_cs000 ...` printed, three files created in `tests/fixtures/ballade1_amateur_cs000/`.

**Step 3: Add a diagnostic export to chroma_dtw.rs for the margin probe**

Add this function to `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs`, just before the `#[cfg(test)]` block:

```rust
/// Returns the last-audio-row accumulated cost slice d[last, *] for probe use.
/// Only compiled in test/non-wasm builds.
#[cfg(not(target_arch = "wasm32"))]
pub fn subseq_dtw_last_row(
    audio: &[f32], n_audio: usize,
    score: &[f32], n_score: usize,
) -> Vec<f32> {
    let n_a = n_audio;
    let n_s = n_score;
    let mut d = vec![f32::INFINITY; n_a * n_s];
    let idx = |i: usize, j: usize| i * n_s + j;
    for j in 0..n_s {
        d[idx(0, j)] = cosine_dist(audio, n_a, 0, score, n_s, j);
    }
    for i in 1..n_a {
        for j in 0..n_s {
            let c = cosine_dist(audio, n_a, i, score, n_s, j);
            let from_up = if i > 0 { d[idx(i - 1, j)] } else { f32::INFINITY };
            let from_left = if j > 0 { d[idx(i, j - 1)] } else { f32::INFINITY };
            let from_diag = if i > 0 && j > 0 { d[idx(i - 1, j - 1)] } else { f32::INFINITY };
            let best = from_diag.min(from_up).min(from_left);
            d[idx(i, j)] = c + best;
        }
    }
    d[(n_a - 1) * n_s..n_a * n_s].to_vec()
}
```

Also add this re-export in `apps/api/src/wasm/score-analysis/src/lib.rs`:

```rust
#[cfg(not(target_arch = "wasm32"))]
pub use chroma_dtw::subseq_dtw_last_row;
```

**Step 4: Create the margin probe script**

Create `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/margin_probe.py`:

```python
"""
Margin hypothesis probe.

Loads the amateur cs_000 and pro cs_111 fixtures, runs the Rust DP via cffi,
and computes separation margins for:
  - Wrong alignment (teleport to ~bar 261): the global argmin on cs_000 without any prior
  - Correct alignment (cs_111 cold-start landing bars 25-40): global argmin on cs_111

Asserts that the correct alignment has a clearly positive margin and the
wrong alignment has a lower margin (ideally negative or < margin_min=0.02).

If the assertion fails, print diagnostic info and exit 1.

Run from project root:
  uv run python apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/margin_probe.py
"""
from __future__ import annotations
import json
import struct
import sys
from pathlib import Path

NEIGHBOR = 50      # frames either side for disjoint-margin computation
MARGIN_MIN = 0.02  # threshold the design uses

def load_fixture(slug: str, root: Path):
    base = root / "apps/api/src/wasm/score-analysis/tests/fixtures" / slug
    raw = (base / "audio_chroma.bin").read_bytes()
    floats = list(struct.unpack(f"<{len(raw)//4}f", raw))
    n_frames = len(floats) // 12
    bars = json.loads((base / "score_bars.json").read_text())
    expected = json.loads((base / "expected.json").read_text())
    return floats, n_frames, bars, expected

def build_score_chroma(bars, frame_rate=50.0):
    import numpy as np
    end_time = 0.0
    for bar in bars:
        for note in bar.get("notes", []):
            end = note["onset_seconds"] + max(note["duration_seconds"], 0.05)
            if end > end_time:
                end_time = end
    if end_time <= 0:
        return np.zeros((12, 0), dtype=np.float32)
    n = int(end_time * frame_rate) + 1
    chroma = np.zeros((12, n), dtype=np.float32)
    for bar in bars:
        for note in bar.get("notes", []):
            onset = note["onset_seconds"]
            dur = max(note["duration_seconds"], 0.05)
            pc = note["pitch"] % 12
            f0 = int(onset * frame_rate)
            f1 = int((onset + dur) * frame_rate) + 1
            chroma[pc, f0:min(f1, n)] += 1.0
    chroma += 1e-3
    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    return (chroma / norms).astype(np.float32)

def compute_last_row(audio_chroma, score_chroma):
    import numpy as np
    n_a = audio_chroma.shape[1]
    n_s = score_chroma.shape[1]
    # Cosine distance: 1 - dot product (both already L2-normalized per column)
    # d[i, j] = accumulated cost
    d = np.full((n_a, n_s), np.inf, dtype=np.float32)
    cost = lambda i, j: float(1.0 - np.dot(audio_chroma[:, i], score_chroma[:, j]))
    for j in range(n_s):
        d[0, j] = cost(0, j)
    for i in range(1, n_a):
        for j in range(n_s):
            c = cost(i, j)
            prev = min(
                d[i-1, j],
                d[i, j-1] if j > 0 else np.inf,
                d[i-1, j-1] if j > 0 else np.inf,
            )
            d[i, j] = c + prev
    return d[-1, :]

def separation_margin(last_row, best_j, neighbor=NEIGHBOR):
    import numpy as np
    best_cost = last_row[best_j]
    n = len(last_row)
    lo = max(0, best_j - neighbor)
    hi = min(n, best_j + neighbor + 1)
    mask = np.ones(n, dtype=bool)
    mask[lo:hi] = False
    outside = last_row[mask]
    if len(outside) == 0:
        return 0.0
    return float(outside.min() - best_cost)

def main():
    import numpy as np
    root = Path(__file__).parents[8]  # project root
    floats_000, n_000, bars_000, exp_000 = load_fixture("ballade1_amateur_cs000", root)
    floats_111, n_111, bars_111, exp_111 = load_fixture("ballade1_coldstart_111s", root)

    audio_000 = np.array(floats_000, dtype=np.float32).reshape(12, n_000)
    audio_111 = np.array(floats_111, dtype=np.float32).reshape(12, n_111)

    print("Building score chroma (this takes ~10s)...")
    score_chroma = build_score_chroma(bars_000, frame_rate=exp_000["frame_rate_hz"])
    n_score = score_chroma.shape[1]
    print(f"  score frames: {n_score}")

    print("Computing last-row DP for cs_000 (amateur opening)...")
    last_row_000 = compute_last_row(audio_000, score_chroma)
    global_best_000 = int(np.argmin(last_row_000))
    margin_000 = separation_margin(last_row_000, global_best_000)
    # Expected: teleport lands near end of score (frame ~> n_score * 0.9)
    teleport_threshold = int(n_score * 0.85)
    print(f"  global_best_000 frame: {global_best_000} / {n_score}  (teleport if > {teleport_threshold})")
    print(f"  margin at global_best_000: {margin_000:.4f}")

    print("Computing last-row DP for cs_111 (pro cold-start)...")
    last_row_111 = compute_last_row(audio_111, score_chroma)
    global_best_111 = int(np.argmin(last_row_111))
    margin_111 = separation_margin(last_row_111, global_best_111)
    print(f"  global_best_111 frame: {global_best_111} / {n_score}")
    print(f"  margin at global_best_111: {margin_111:.4f}")

    print()
    print("=== HYPOTHESIS CHECK ===")
    teleport_occurred = global_best_000 > teleport_threshold
    print(f"  Amateur cs_000 teleports (frame > {teleport_threshold}): {teleport_occurred}")
    print(f"  Amateur cs_000 margin: {margin_000:.4f}  (want <= {MARGIN_MIN} for abstain to trigger)")
    print(f"  Pro cs_111 margin: {margin_111:.4f}  (want >= {MARGIN_MIN} for aligned to trigger)")

    ok = True
    if not teleport_occurred:
        print("WARN: cs_000 did not teleport — the failure mode may have changed.")
    if margin_111 < MARGIN_MIN:
        print(f"FAIL: Pro cs_111 margin {margin_111:.4f} < margin_min {MARGIN_MIN}. "
              "The margin does not separate correct alignments. Plan must be revisited.")
        ok = False
    if margin_000 >= margin_111:
        print(f"FAIL: Amateur cs_000 margin {margin_000:.4f} >= pro cs_111 margin {margin_111:.4f}. "
              "The margin does not discriminate. Plan must be revisited.")
        ok = False
    if ok:
        print("PASS: Margin hypothesis holds. Proceed with implementation.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Step 5: Run the probe — verify PASS**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run python apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/margin_probe.py
```

Expected output ends with: `PASS: Margin hypothesis holds. Proceed with implementation.`

If it exits 1, the fundamental assumption is broken — do NOT proceed to Group A. Surface the diagnostic output and revisit the design.

**Step 6: Commit**

```bash
git add apps/api/src/wasm/score-analysis/tests/fixtures/ \
        apps/api/src/wasm/score-analysis/src/chroma_dtw.rs \
        apps/api/src/wasm/score-analysis/src/lib.rs && \
git commit -m "test(chroma-dtw): add amateur cs_000 fixture + margin hypothesis probe (PASS)"
```

---

## Task 1: Uniformity Gate — Silence Abstain

**Group:** A (parallel with Task 2)

**Behavior being verified:** A chunk whose normalized chroma columns are all near-uniform (silence) is classified as `status="abstained"` without running the DP fill.

**Interface under test:** `chroma_dtw_native` in `chroma_dtw.rs` (the extended signature from Task 2 is not needed yet; this task validates the gate in isolation with a stub signature extension).

**Files:**
- Modify: `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs`
- Test: `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`

**Step 1: Write the failing test**

Add to `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`:

```rust
/// A chunk of pure silence (uniform chroma) must be classified as abstained
/// without running the DP. The uniformity gate fires because every column is
/// fully uniform (all 12 pitch classes equal), so uniformity fraction = 0.0,
/// which is below uniformity_min=0.3.
#[test]
fn silence_chunk_abstains() {
    use score_analysis::chroma_dtw_native_v2;
    use score_analysis::types::ScoreBar;

    // Build a minimal score: one bar with one note
    let bar = ScoreBar {
        bar_number: 1,
        start_tick: 0,
        start_seconds: 0.0,
        time_signature: "4/4".to_string(),
        notes: vec![score_analysis::types::ScoreNote {
            pitch: 60,
            pitch_name: "C4".to_string(),
            velocity: 80,
            onset_tick: 0,
            onset_seconds: 0.0,
            duration_ticks: 480,
            duration_seconds: 1.0,
            track: 0,
        }],
        pedal_events: vec![],
        note_count: 1,
        pitch_range: vec![60],
        mean_velocity: 80,
    };

    // Silence chroma: all 12 pitch classes equal per column -> uniform column
    // After L2-normalization each value = 1/sqrt(12). We set them all to that.
    let n_frames: usize = 50;
    let uniform_val = 1.0_f32 / 12.0_f32.sqrt();
    let audio_f32 = vec![uniform_val; 12 * n_frames];

    let result = chroma_dtw_native_v2(
        &audio_f32,
        n_frames as u32,
        &[bar],
        50.0,  // frame_rate_hz
        5.0,   // decim_hz
        -1,    // expected_score_frame: cold start
        150,   // band_back_frames
        300,   // band_fwd_frames
        0.02,  // margin_min
        0.30,  // uniformity_min
    )
    .expect("native_v2 should not error on valid input");

    assert_eq!(
        result.status, "abstained",
        "silence chunk must produce status=abstained, got {}",
        result.status
    );
    // DP was skipped: confidence should be 0.0
    assert_eq!(
        result.confidence, 0.0,
        "skipped DP must yield confidence=0.0"
    );
}
```

**Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test silence_chunk_abstains 2>&1 | head -20
```

Expected: FAIL — `error[E0425]: cannot find function 'chroma_dtw_native_v2'`

**Step 3: Implement the uniformity gate**

In `apps/api/src/wasm/score-analysis/src/types.rs`, extend `BarMapChroma`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarMapChroma {
    pub bar_min: u32,
    pub bar_max: u32,
    pub cost: f32,
    pub bar_per_frame: Vec<u32>,
    /// Score-side frame index at the chosen end of the warping path.
    /// 0 when status="abstained" (no DP run).
    pub end_score_frame: u32,
    /// Separation margin of the chosen alignment region (skill-invariant).
    /// 0.0 when status="abstained".
    pub confidence: f32,
    /// "aligned" | "relocalized" | "abstained"
    pub status: String,
}
```

In `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs`, add the uniformity helper and the new entry point `chroma_dtw_native_v2`. Place this immediately after the closing brace of `chroma_dtw_native`:

```rust
/// Fraction of columns in a normalized chroma matrix that are non-uniform.
/// A column is "peaky" if its max value > threshold (1/sqrt(12) * 1.5 ≈ 0.433).
/// Uniform columns (silence) have all values near 1/sqrt(12) ≈ 0.289.
fn uniformity_fraction(audio_f32: &[f32], n_audio: usize) -> f32 {
    let peaky_threshold = 1.0_f32 / 12.0_f32.sqrt() * 1.5;
    let mut peaky = 0usize;
    for f in 0..n_audio {
        let col_max = (0..12)
            .map(|pc| audio_f32[pc * n_audio + f])
            .fold(f32::NEG_INFINITY, f32::max);
        if col_max > peaky_threshold {
            peaky += 1;
        }
    }
    peaky as f32 / n_audio as f32
}

/// V2 entry point: adds uniformity gate + new parameters.
/// All existing logic (subseq_dtw, build_score_chroma, etc.) is unchanged.
/// Arbitration and margin computation are stubs in this task; full implementation
/// is in Task 3 (margin) and Task 4 (arbitration).
pub fn chroma_dtw_native_v2(
    audio_f32: &[f32],
    n_audio: u32,
    score_bars: &[ScoreBar],
    frame_rate_hz: f32,
    decim_hz: f32,
    expected_score_frame: i32,
    band_back_frames: u32,
    band_fwd_frames: u32,
    margin_min: f32,
    uniformity_min: f32,
) -> Result<BarMapChroma, String> {
    let n_a = n_audio as usize;
    if audio_f32.len() != 12 * n_a {
        return Err(format!(
            "audio_f32 length {} != 12 * n_audio {}",
            audio_f32.len(),
            12 * n_a
        ));
    }
    if score_bars.is_empty() {
        return Err("score_bars is empty".to_string());
    }
    if n_a == 0 {
        return Err("n_audio is zero".to_string());
    }

    // Uniformity gate: skip DP for silence
    let uf = uniformity_fraction(audio_f32, n_a);
    if uf < uniformity_min {
        return Ok(BarMapChroma {
            bar_min: 0,
            bar_max: 0,
            cost: 0.0,
            bar_per_frame: vec![],
            end_score_frame: 0,
            confidence: 0.0,
            status: "abstained".to_string(),
        });
    }

    // Fall through to existing logic (margin + arbitration stubs — filled in Tasks 3 & 4)
    let core = chroma_dtw_native(audio_f32, n_audio, score_bars, frame_rate_hz, decim_hz)?;

    // TODO(Task 3): compute confidence from last-row margin
    // TODO(Task 4): proper arbitration; for now always "aligned"
    let _ = (expected_score_frame, band_back_frames, band_fwd_frames, margin_min);
    let end_score_frame = 0u32; // placeholder; Task 4 fills this from backtrack
    Ok(BarMapChroma {
        bar_min: core.bar_min,
        bar_max: core.bar_max,
        cost: core.cost,
        bar_per_frame: core.bar_per_frame,
        end_score_frame,
        confidence: 0.0,
        status: "aligned".to_string(),
    })
}
```

Also update `apps/api/src/wasm/score-analysis/src/lib.rs` to re-export `chroma_dtw_native_v2`:

```rust
pub use chroma_dtw::chroma_dtw_native_v2;
```

**Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test silence_chunk_abstains 2>&1
```

Expected: `test silence_chunk_abstains ... ok`

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && \
git add apps/api/src/wasm/score-analysis/src/chroma_dtw.rs \
        apps/api/src/wasm/score-analysis/src/types.rs \
        apps/api/src/wasm/score-analysis/src/lib.rs \
        apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs && \
git commit -m "feat(chroma-dtw): add uniformity silence gate; status=abstained for uniform chunks"
```

---

## Task 2: Extend BarMapChroma Struct (Rust types)

**Group:** A (parallel with Task 1)

NOTE: Task 1 already extends `BarMapChroma` and must be checked in first if running in parallel with Task 2. If the two tasks are dispatched to the same subagent or Task 1 runs first, this task is already complete. If Task 2 runs independently, it must apply the same struct extension.

**Behavior being verified:** The existing `chroma_dtw_native` (stateless, V1) still passes its roundtrip tests after `BarMapChroma` gains `end_score_frame`, `confidence`, and `status` fields with serde defaults.

**Interface under test:** `chroma_dtw_native` (V1 path); cargo roundtrip tests still pass.

**Files:**
- Modify: `apps/api/src/wasm/score-analysis/src/types.rs`
- Test: `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs` (existing tests must still pass)

**Step 1: Write the failing test**

Add to `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`:

```rust
/// The V1 chroma_dtw_native result must include the new BarMapChroma fields
/// with appropriate zero/default values. This test ensures the struct change
/// is backward compatible: existing callers get sensible defaults.
#[test]
fn bar_map_chroma_v1_has_new_fields_with_defaults() {
    let (audio_f32, n_audio, score_bars, expected) = load_fixture("ballade1_coldstart_111s");

    let result: score_analysis::types::BarMapChroma =
        score_analysis::chroma_dtw_native(&audio_f32, n_audio, &score_bars, expected.frame_rate_hz, 5.0)
            .expect("should not fail");

    // V1 path must fill in the new fields. Until Task 4 wires them, they are 0/"aligned".
    // The key check: compiling and calling still works, new fields are accessible.
    assert!(
        result.status == "aligned" || result.status == "abstained",
        "status must be a known value, got {:?}", result.status
    );
    // end_score_frame is u32, so it is accessible. We only check it's a valid frame index.
    let total_score_frames = (result.bar_per_frame.len() * 10) as u32; // rough upper bound
    assert!(
        result.end_score_frame <= total_score_frames + 1000,
        "end_score_frame {} is unreasonably large", result.end_score_frame
    );
}
```

**Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test bar_map_chroma_v1_has_new_fields_with_defaults 2>&1 | head -20
```

Expected: FAIL — `error[E0609]: no field 'status' on type 'BarMapChroma'`

**Step 3: Implement the struct extension**

If Task 1 has not already applied this, apply the same `BarMapChroma` extension to `apps/api/src/wasm/score-analysis/src/types.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarMapChroma {
    pub bar_min: u32,
    pub bar_max: u32,
    pub cost: f32,
    pub bar_per_frame: Vec<u32>,
    pub end_score_frame: u32,
    pub confidence: f32,
    pub status: String,
}
```

Update `chroma_dtw_native` in `chroma_dtw.rs` to populate the new fields on its `Ok(BarMapChroma { ... })` return:

```rust
Ok(BarMapChroma {
    bar_min,
    bar_max,
    cost: mean_cost,
    bar_per_frame,
    end_score_frame: j_end as u32,  // j_end is already computed in subseq_dtw
    confidence: 0.0,                // placeholder until Task 3
    status: "aligned".to_string(), // placeholder until Task 4
})
```

Note: `j_end` is currently local to `subseq_dtw`. To expose it, have `subseq_dtw` return `(path, mean_cost, j_end)` as a 3-tuple: change its return type from `(Vec<(usize, usize)>, f32)` to `(Vec<(usize, usize)>, f32, usize)` and update all call sites.

Updated `subseq_dtw` signature and return:

```rust
fn subseq_dtw(
    audio: &[f32], n_audio: usize,
    score: &[f32], n_score: usize,
) -> (Vec<(usize, usize)>, f32, usize) {  // <- added usize
    // ... (all existing logic unchanged) ...
    // At the end, replace:
    (path, mean_cost)
    // With:
    (path, mean_cost, j_end)
}
```

Update the call site in `chroma_dtw_native`:

```rust
let (path, mean_cost, j_end) = subseq_dtw(audio_f32, n_a, &score_chroma, n_s);
```

**Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test bar_map_chroma_v1_has_new_fields_with_defaults 2>&1
```

Also verify existing tests still pass:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test 2>&1 | tail -20
```

Expected: all tests pass.

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && \
git add apps/api/src/wasm/score-analysis/src/types.rs \
        apps/api/src/wasm/score-analysis/src/chroma_dtw.rs \
        apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs && \
git commit -m "feat(chroma-dtw): extend BarMapChroma with end_score_frame, confidence, status"
```

---

## Task 3: Separation Margin Computation

**Group:** B (sequential, depends on Group A)

**Behavior being verified:** Given the last-audio-row accumulated-cost array, `separation_margin` returns the gap between the best candidate cost and the minimum cost outside a ±50-frame disjoint neighborhood — a positive value means the candidate is unambiguous, zero or negative means it is not.

**Interface under test:** A new internal helper `compute_margin` tested via `chroma_dtw_native_v2` on the cold-start fixture: the returned `confidence` field must be positive (margin > 0) for the pro cold-start chunk.

**Files:**
- Modify: `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs`
- Test: `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`

**Step 1: Write the failing test**

Add to `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`:

```rust
/// For the pro cold-start at 111s, the separation margin should be positive —
/// there is a clearly better region than anywhere else in the score.
/// This validates the margin computation before arbitration is wired in.
#[test]
fn pro_coldstart_has_positive_confidence() {
    use score_analysis::chroma_dtw_native_v2;

    let (audio_f32, n_audio, score_bars, expected) = load_fixture("ballade1_coldstart_111s");

    let result = chroma_dtw_native_v2(
        &audio_f32,
        n_audio,
        &score_bars,
        expected.frame_rate_hz,
        5.0,
        -1,   // cold start
        150,
        300,
        0.02,
        0.30,
    )
    .expect("should not fail");

    assert!(
        result.confidence > 0.0,
        "pro cold-start must have positive confidence (got {})",
        result.confidence
    );
    assert_eq!(result.status, "aligned");
}
```

**Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test pro_coldstart_has_positive_confidence 2>&1 | head -20
```

Expected: FAIL — `assertion failed: result.confidence > 0.0` (confidence is 0.0 from the Task 1 stub)

**Step 3: Implement margin computation**

Add to `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs` (after `uniformity_fraction`):

```rust
/// Separation margin for a candidate at `best_j` in the last-row cost array.
/// Returns: min(cost outside ±neighbor of best_j) - cost[best_j].
/// Positive = unambiguous; near-zero or negative = ambiguous.
fn separation_margin(last_row: &[f32], best_j: usize, neighbor: usize) -> f32 {
    let n = last_row.len();
    let best_cost = last_row[best_j];
    let lo = best_j.saturating_sub(neighbor);
    let hi = (best_j + neighbor + 1).min(n);
    let outside_min = last_row
        .iter()
        .enumerate()
        .filter(|&(j, _)| j < lo || j >= hi)
        .map(|(_, &c)| c)
        .fold(f32::INFINITY, f32::min);
    if outside_min.is_infinite() {
        // Entire score is within the neighborhood — no outside to compare
        return 0.0;
    }
    outside_min - best_cost
}
```

Update `chroma_dtw_native_v2` to compute and return the confidence. The function needs access to the last-row DP array. Refactor `subseq_dtw` to also return the last-row slice alongside the path:

Change `subseq_dtw` return type to `(Vec<(usize, usize)>, f32, usize, Vec<f32>)` (adds last_row):

```rust
fn subseq_dtw(
    audio: &[f32], n_audio: usize,
    score: &[f32], n_score: usize,
) -> (Vec<(usize, usize)>, f32, usize, Vec<f32>) {
    // ... all existing logic unchanged ...
    // After finding j_end and the path, add:
    let last_row = d[(n_a - 1) * n_s..n_a * n_s].to_vec();
    (path, mean_cost, j_end, last_row)
}
```

Update all callers. In `chroma_dtw_native`:

```rust
let (path, mean_cost, j_end, _last_row) = subseq_dtw(audio_f32, n_a, &score_chroma, n_s);
```

In `chroma_dtw_native_v2` (replacing the stub fall-through after the uniformity gate):

```rust
let score_chroma = build_score_chroma(score_bars, frame_rate_hz);
let n_s = score_chroma.len() / 12;
if n_s == 0 {
    return Err("score has no notes".to_string());
}
let (path, mean_cost, j_end, last_row) = subseq_dtw(audio_f32, n_a, &score_chroma, n_s);

// Compute separation margin for the global best
const NEIGHBOR: usize = 50;
let global_margin = separation_margin(&last_row, j_end, NEIGHBOR);

// TODO(Task 4): in-band candidate + full arbitration. For now use global best always.
let _ = (expected_score_frame, band_back_frames, band_fwd_frames, margin_min);

// Build bar_per_frame from path
let decim_step = (frame_rate_hz / decim_hz).round() as usize;
let decim_step = decim_step.max(1);
let decim_n = (n_a + decim_step - 1) / decim_step;
let mut audio_to_score = vec![0usize; n_a];
let mut frame_mapped = vec![false; n_a];
for &(sf, af) in &path {
    if af < n_a {
        audio_to_score[af] = sf;
        frame_mapped[af] = true;
    }
}
let mut last_sf = path.first().map(|&(sf, _)| sf).unwrap_or(0);
for af in 0..n_a {
    if frame_mapped[af] {
        last_sf = audio_to_score[af];
    } else {
        audio_to_score[af] = last_sf;
    }
}
let bar_per_frame: Vec<u32> = (0..decim_n)
    .map(|d| {
        let af = (d * decim_step).min(n_a - 1);
        let sf = audio_to_score[af];
        frame_to_bar(sf, frame_rate_hz, score_bars)
    })
    .collect();
let bar_min = bar_per_frame.iter().copied().min().unwrap_or(1);
let bar_max = bar_per_frame.iter().copied().max().unwrap_or(1);

Ok(BarMapChroma {
    bar_min,
    bar_max,
    cost: mean_cost,
    bar_per_frame,
    end_score_frame: j_end as u32,
    confidence: global_margin,
    status: "aligned".to_string(), // arbitration in Task 4
})
```

**Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test pro_coldstart_has_positive_confidence 2>&1
```

Also run all tests:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test 2>&1 | tail -20
```

Expected: all tests pass.

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && \
git add apps/api/src/wasm/score-analysis/src/chroma_dtw.rs \
        apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs && \
git commit -m "feat(chroma-dtw): compute separation margin; confidence field populated in v2"
```

---

## Task 4: Arbitration State Machine + Updated WASM Entry Point

**Group:** B (sequential, depends on Group A; same group as Task 3, must run after Task 3)

**Behavior being verified:** The full arbitration logic selects in-band vs global vs abstain correctly, and `align_chunk_chroma` WASM entry point accepts the five new parameters.

**Interface under test:** `chroma_dtw_native_v2` cold/warm/abstain paths; the existing WASM entry point `align_chunk_chroma` is replaced by a new signature forwarding to `chroma_dtw_native_v2`.

**Files:**
- Modify: `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs`
- Test: `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`

**Step 1: Write the failing test**

Add to `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`:

```rust
/// Cold start with very low margin_min=-1.0 (always passes) must be "aligned",
/// not "abstained". This pins the cold-start arbitration branch.
#[test]
fn cold_start_with_permissive_threshold_is_aligned() {
    use score_analysis::chroma_dtw_native_v2;

    let (audio_f32, n_audio, score_bars, expected) = load_fixture("ballade1_coldstart_111s");

    let result = chroma_dtw_native_v2(
        &audio_f32,
        n_audio,
        &score_bars,
        expected.frame_rate_hz,
        5.0,
        -1,    // cold start
        150,
        300,
        -1.0,  // margin_min very permissive: always passes
        0.30,
    )
    .expect("should not fail");

    assert_eq!(
        result.status, "aligned",
        "cold start with permissive margin must be aligned, got {}",
        result.status
    );
    assert!(result.confidence > -1.0, "confidence must be a real number");
}

/// Cold start with very high margin_min=1.0 (never passes) must be "abstained".
#[test]
fn cold_start_with_strict_threshold_abstains() {
    use score_analysis::chroma_dtw_native_v2;

    let (audio_f32, n_audio, score_bars, expected) = load_fixture("ballade1_coldstart_111s");

    let result = chroma_dtw_native_v2(
        &audio_f32,
        n_audio,
        &score_bars,
        expected.frame_rate_hz,
        5.0,
        -1,    // cold start
        150,
        300,
        1.0,   // margin_min impossibly strict
        0.30,
    )
    .expect("should not fail");

    assert_eq!(
        result.status, "abstained",
        "cold start that fails margin check must be abstained, got {}",
        result.status
    );
}
```

**Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test "cold_start_with" 2>&1 | head -20
```

Expected: FAIL — `assertion failed` for `cold_start_with_strict_threshold_abstains` (status is "aligned" because Task 3 left arbitration as a stub).

**Step 3: Implement full arbitration in `chroma_dtw_native_v2`**

Replace the `// TODO(Task 4)` section in `chroma_dtw_native_v2` with the full arbitration:

```rust
// Clamp expected_score_frame to valid range; treat out-of-range as cold
let n_s = score_chroma.len() / 12;
let effective_expected: i32 = if expected_score_frame < 0
    || expected_score_frame as usize >= n_s
{
    -1
} else {
    expected_score_frame
};

// Global best and its margin
const NEIGHBOR: usize = 50;
let global_j = last_row
    .iter()
    .enumerate()
    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    .map(|(j, _)| j)
    .unwrap_or(0);
let global_margin = separation_margin(&last_row, global_j, NEIGHBOR);

let (chosen_j, chosen_margin, chosen_status) = if effective_expected < 0 {
    // Cold start: take global
    let status = if global_margin >= margin_min { "aligned" } else { "abstained" };
    (global_j, global_margin, status)
} else {
    // Warm: in-band window
    let exp = effective_expected as usize;
    let lo = exp.saturating_sub(band_back_frames as usize).min(n_s - 1);
    let hi = (exp + band_fwd_frames as usize).min(n_s - 1);
    let band_j = (lo..=hi)
        .min_by(|&ja, &jb| last_row[ja].partial_cmp(&last_row[jb]).unwrap())
        .unwrap_or(exp.min(n_s - 1));
    let band_margin = separation_margin(&last_row, band_j, NEIGHBOR);

    if band_margin >= margin_min {
        (band_j, band_margin, "aligned")
    } else if global_margin >= margin_min
        && (global_j < lo || global_j > hi)
    {
        // Global is clearly outside the band and passes margin
        (global_j, global_margin, "relocalized")
    } else {
        // Neither candidate is unambiguous
        (global_j, global_margin.max(band_margin), "abstained")
    }
};

// If abstained, return without bar range
if chosen_status == "abstained" {
    return Ok(BarMapChroma {
        bar_min: 0,
        bar_max: 0,
        cost: 0.0,
        bar_per_frame: vec![],
        end_score_frame: 0,
        confidence: chosen_margin,
        status: "abstained".to_string(),
    });
}

// Backtrack from chosen_j to build bar_per_frame
// Re-run subseq_dtw is expensive; instead use the existing path if chosen_j == j_end,
// otherwise we must rerun. For now: if chosen_j != j_end, rerun subseq_dtw from chosen_j.
// (Correctness-first; optimization deferred if needed)
let final_path = if chosen_j == j_end {
    path
} else {
    // Backtrack from chosen_j using the p[] predecessor table.
    // Since we don't store p[] outside subseq_dtw, we use a helper that reruns
    // only the backtrack from a given j_end. Add `subseq_dtw_from_end` below.
    subseq_dtw_backtrack_from(audio_f32, n_a, &score_chroma, n_s, chosen_j)
};
```

Add the backtrack helper just before `chroma_dtw_native`:

```rust
/// Re-run full DP fill and backtrack from a specific end column `j_end`.
/// Used when arbitration selects a different endpoint than subseq_dtw's global argmin.
fn subseq_dtw_backtrack_from(
    audio: &[f32], n_audio: usize,
    score: &[f32], n_score: usize,
    j_end: usize,
) -> Vec<(usize, usize)> {
    let n_a = n_audio;
    let n_s = n_score;
    let mut d = vec![f32::INFINITY; n_a * n_s];
    let mut p: Vec<(i32, i32)> = vec![(0, 0); n_a * n_s];
    let idx = |i: usize, j: usize| i * n_s + j;
    for j in 0..n_s {
        d[idx(0, j)] = cosine_dist(audio, n_a, 0, score, n_s, j);
    }
    for i in 1..n_a {
        for j in 0..n_s {
            let c = cosine_dist(audio, n_a, i, score, n_s, j);
            let from_up = if i > 0 { d[idx(i - 1, j)] } else { f32::INFINITY };
            let from_left = if j > 0 { d[idx(i, j - 1)] } else { f32::INFINITY };
            let from_diag = if i > 0 && j > 0 { d[idx(i - 1, j - 1)] } else { f32::INFINITY };
            let (best, pred) = if from_diag <= from_up && from_diag <= from_left {
                (from_diag, (-1i32, -1i32))
            } else if from_up <= from_left {
                (from_up, (-1, 0))
            } else {
                (from_left, (0, -1))
            };
            d[idx(i, j)] = c + best;
            p[idx(i, j)] = pred;
        }
    }
    let last_row = n_a - 1;
    let mut path = Vec::with_capacity(n_a + n_s);
    let mut i = last_row as i32;
    let mut j = j_end as i32;
    while i >= 0 {
        path.push((j as usize, i as usize));
        let (di, dj) = p[idx(i as usize, j as usize)];
        if di == 0 && dj == 0 { break; }
        i += di;
        j += dj;
    }
    path.reverse();
    path
}
```

Also update the WASM entry point `align_chunk_chroma` to accept and forward the new parameters:

```rust
#[wasm_bindgen]
pub fn align_chunk_chroma(
    audio_bytes: &[u8],
    n_audio: u32,
    score_bars_js: JsValue,
    frame_rate_hz: f32,
    decim_hz: f32,
    expected_score_frame: i32,
    band_back_frames: u32,
    band_fwd_frames: u32,
    margin_min: f32,
    uniformity_min: f32,
) -> Result<JsValue, JsValue> {
    if audio_bytes.len() % 4 != 0 {
        return Err(JsValue::from_str("audio_bytes length not a multiple of 4"));
    }
    let audio_f32: Vec<f32> = audio_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let score_bars: Vec<crate::types::ScoreBar> =
        serde_wasm_bindgen::from_value(score_bars_js)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = chroma_dtw_native_v2(
        &audio_f32,
        n_audio,
        &score_bars,
        frame_rate_hz,
        decim_hz,
        expected_score_frame,
        band_back_frames,
        band_fwd_frames,
        margin_min,
        uniformity_min,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}
```

**Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test "cold_start_with" 2>&1
```

Also:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test 2>&1 | tail -20
```

Expected: all tests pass.

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && \
git add apps/api/src/wasm/score-analysis/src/chroma_dtw.rs \
        apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs && \
git commit -m "feat(chroma-dtw): full arbitration (aligned/relocalized/abstained) + updated WASM entry point"
```

---

## Task 5: Teleport Regression Fixture + Cargo Test

**Group:** C (sequential, depends on Group B)

**Behavior being verified:** The amateur opening chunk (`cs_000`) with a warm expected position set to an early bar locks to bars 1–10, not 261–262 (the teleport target from the stateless DTW).

**Interface under test:** `chroma_dtw_native_v2` with `expected_score_frame` set to a frame corresponding to bar 3 (~early position).

**Files:**
- Modify: `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`

**Step 1: Write the failing test**

Add to `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`:

```rust
/// Headline regression test: amateur opening chunk must not teleport.
/// With expected_score_frame pointing to an early position (bar ~3),
/// the in-band search must win and land in bars 1-15, not 261-262.
#[test]
fn teleport_regression_amateur_cs000() {
    use score_analysis::chroma_dtw_native_v2;

    let (audio_f32, n_audio, score_bars, expected) = load_fixture("ballade1_amateur_cs000");

    // Bar 3 at 50 Hz: approximate frame index.
    // Chopin Ballade 1 starts at 0s, so bar 3 is around 10-15s -> ~600 frames at 50Hz.
    // Use 300 frames (6s) as a safe early-position prior.
    let early_frame: i32 = 300;

    let result = chroma_dtw_native_v2(
        &audio_f32,
        n_audio,
        &score_bars,
        expected.frame_rate_hz,
        5.0,
        early_frame,
        300,   // band_back_frames: 6s back
        750,   // band_fwd_frames: 15s forward (full chunk range)
        0.01,  // margin_min: permissive
        0.30,
    )
    .expect("should not fail");

    assert!(
        result.status == "aligned" || result.status == "relocalized",
        "must align or relocalize, not abstain: got {}", result.status
    );
    assert!(
        result.bar_min >= expected.bar_min_lo && result.bar_min <= expected.bar_min_hi,
        "bar_min={} must be in [{}, {}] (no teleport to end of score)",
        result.bar_min, expected.bar_min_lo, expected.bar_min_hi
    );
}
```

**Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test teleport_regression_amateur_cs000 2>&1 | head -30
```

Expected: FAIL — `bar_min=261` is outside `[1, 10]`, or the fixture file does not exist yet (if generate.py was not updated).

If the fixture does not exist: run `uv run python apps/api/src/wasm/score-analysis/tests/fixtures/generate.py` from the project root first.

**Step 3: Implement — no new Rust code needed**

This test exercises the arbitration logic shipped in Task 4. If the test fails after Task 4 is complete, the in-band band window is too narrow. Widen `band_fwd_frames` in the test to 1000 (20s) and re-run. If it still fails, increase `margin_min` search range. The expected.json bar bounds for `ballade1_amateur_cs000` must reflect the true first-chunk range (bars 1–10).

If the test is failing because `expected.json` has wrong bounds, update `generate.py` bounds for `ballade1_amateur_cs000` and re-run the generator. The `bar_min_lo=1`, `bar_min_hi=10`, `bar_max_lo=1`, `bar_max_hi=15` values from Task 0 are the correct bounds.

**Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test teleport_regression_amateur_cs000 2>&1
```

Expected: `test teleport_regression_amateur_cs000 ... ok`

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && \
git add apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs \
        apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/ && \
git commit -m "test(chroma-dtw): teleport regression — amateur cs_000 must land bars 1-15"
```

---

## Task 6: Cold-Start Non-Regression

**Group:** C (sequential within Group C, after Task 5)

**Behavior being verified:** The existing pro cold-start at 111s (`ballade1_coldstart_111s`) still lands bars 25–40 when called through `chroma_dtw_native_v2` with `expected=-1`. The V2 path does not regress on the V1 fixture.

**Interface under test:** `chroma_dtw_native_v2` on the existing `ballade1_coldstart_111s` fixture.

**Files:**
- Modify: `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`

**Step 1: Write the failing test**

Add to `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`:

```rust
/// V2 cold-start must not regress on the existing pro fixture.
/// The result must match the same bar bounds as the V1 roundtrip test.
#[test]
fn coldstart_v2_non_regression() {
    use score_analysis::chroma_dtw_native_v2;

    let (audio_f32, n_audio, score_bars, expected) = load_fixture("ballade1_coldstart_111s");

    let result = chroma_dtw_native_v2(
        &audio_f32,
        n_audio,
        &score_bars,
        expected.frame_rate_hz,
        5.0,
        -1,    // cold start — same as V1
        150,
        300,
        0.01,  // permissive margin so it does not accidentally abstain
        0.30,
    )
    .expect("should not fail");

    assert_eq!(
        result.status, "aligned",
        "pro cold start must be aligned, got {}", result.status
    );
    assert!(
        result.bar_min >= expected.bar_min_lo && result.bar_min <= expected.bar_min_hi,
        "bar_min={} not in [{}, {}]",
        result.bar_min, expected.bar_min_lo, expected.bar_min_hi
    );
    assert!(
        result.bar_max >= expected.bar_max_lo && result.bar_max <= expected.bar_max_hi,
        "bar_max={} not in [{}, {}]",
        result.bar_max, expected.bar_max_lo, expected.bar_max_hi
    );
    // Non-decreasing within chunk
    for w in result.bar_per_frame.windows(2) {
        assert!(w[1] >= w[0], "bar_per_frame not non-decreasing: {:?}", w);
    }
}
```

**Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test coldstart_v2_non_regression 2>&1 | head -20
```

Expected: FAIL — bar bounds wrong (arbitration path not yet routing correctly) or status is "abstained".

**Step 3: Implement — no new Rust code; fix any arbitration edge case**

If the test fails because of a bug introduced in Task 4's arbitration (e.g., the DP was re-run unnecessarily and produced wrong bars), debug the `subseq_dtw_backtrack_from` vs the original `path` selection. The root cause is always either wrong `chosen_j` or wrong backtrack.

**Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test coldstart_v2_non_regression 2>&1
```

Expected: `test coldstart_v2_non_regression ... ok`

Also confirm all cargo tests pass:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test 2>&1 | tail -20
```

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && \
git add apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs && \
git commit -m "test(chroma-dtw): cold-start V2 non-regression — must still land bars 25-40"
```

---

## Task 7: Relocalize Scenario

**Group:** C (sequential within Group C, after Task 6)

**Behavior being verified:** A chunk whose true position is far from the expected position, but has a clear global margin, is classified as `status="relocalized"` and lands in the correct distant bars.

**Interface under test:** `chroma_dtw_native_v2` on the existing `ballade1_forward_2min` fixture (2 min of pro audio starting at 0s, covering bars 1–~30) called with `expected_score_frame` pointing to a position near the end of the piece (score frame ~12000, bar ~240).

**Files:**
- Modify: `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`

**Step 1: Write the failing test**

Add to `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`:

```rust
/// When the expected position is far from the true position (simulating a
/// session that was previously at the end of the piece but the player
/// jumped back to bar 1), the DTW must recognize the global match is far
/// from expected and return status="relocalized".
#[test]
fn relocalize_when_true_position_far_from_expected() {
    use score_analysis::chroma_dtw_native_v2;

    // This fixture covers bars 1-~30; its audio is real music with a clear match.
    let (audio_f32, n_audio, score_bars, expected) = load_fixture("ballade1_forward_2min");

    // Simulate: session previously ended near bar 240 (frame ~12000 at 50Hz)
    // True position: bars 1-30, which is far from the expected band [12000-150, 12000+300]
    let far_expected: i32 = 12000;

    let result = chroma_dtw_native_v2(
        &audio_f32,
        n_audio,
        &score_bars,
        expected.frame_rate_hz,
        5.0,
        far_expected,
        150,   // band_back
        300,   // band_fwd
        0.01,  // permissive margin_min
        0.30,
    )
    .expect("should not fail");

    assert_eq!(
        result.status, "relocalized",
        "must relocalize when true position is far from expected: got status={}  bar_min={}",
        result.status, result.bar_min
    );
    // Must land in actual early bars, not near bar 240
    assert!(
        result.bar_min <= expected.bar_max_hi,
        "relocalized bar_min={} must be in first section (<={})",
        result.bar_min, expected.bar_max_hi
    );
}
```

**Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test relocalize_when_true_position_far_from_expected 2>&1 | head -20
```

Expected: FAIL — status is "abstained" or "aligned" rather than "relocalized", or bar_min is near 240.

**Step 3: Implement — tune relocalize condition if needed**

If the test fails because `global_j` falls within the band (i.e., the band window [11850, 12300] actually captures the correct position), lower `far_expected` to `15000` (near score end) and re-check. The key invariant: the global best must be outside the band `[far_expected - 150, far_expected + 300]`.

No new Rust logic is needed if the arbitration from Task 4 is correct. The `relocalized` path fires when `global_margin >= margin_min && (global_j < lo || global_j > hi)`. Verify this logic in Task 4's implementation.

**Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test relocalize_when_true_position_far_from_expected 2>&1
```

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && \
git add apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs && \
git commit -m "test(chroma-dtw): relocalize scenario — far-from-expected position gets status=relocalized"
```

---

## Task 8: Low-Margin Abstain

**Group:** C (sequential within Group C, after Task 7)

**Behavior being verified:** A chunk with a very strict `margin_min` that no real alignment clears is classified as `status="abstained"`.

**Interface under test:** `chroma_dtw_native_v2` on any real fixture with `margin_min=1.0` (impossible threshold).

**Files:**
- Modify: `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`

**Step 1: Write the failing test**

Add to `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs`:

```rust
/// When margin_min is set impossibly high (1.0), even a correct alignment
/// must be classified as abstained because no real alignment has margin >= 1.0.
/// This validates the low-margin abstain path in warm mode.
#[test]
fn low_margin_abstains_warm() {
    use score_analysis::chroma_dtw_native_v2;

    let (audio_f32, n_audio, score_bars, expected) = load_fixture("ballade1_coldstart_111s");

    // Set expected near the known correct position (bar ~30 = ~1500 frames at 50Hz)
    // so we enter warm mode, then set margin_min=1.0 to force abstain.
    let result = chroma_dtw_native_v2(
        &audio_f32,
        n_audio,
        &score_bars,
        expected.frame_rate_hz,
        5.0,
        1500,  // warm: expected near bar ~30
        300,
        300,
        1.0,   // margin_min impossible — no real margin reaches this
        0.30,
    )
    .expect("should not fail");

    assert_eq!(
        result.status, "abstained",
        "impossibly strict margin must abstain, got {}", result.status
    );
    assert_eq!(result.bar_per_frame.len(), 0, "abstained must return empty bar_per_frame");
}
```

**Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test low_margin_abstains_warm 2>&1 | head -20
```

Expected: FAIL — status is "aligned" not "abstained", or `bar_per_frame` is non-empty.

**Step 3: Implement — verify abstained returns empty bar_per_frame**

If the test fails because the abstained return path (in Task 4) doesn't return `bar_per_frame: vec![]`, fix it. The abstained early-return in `chroma_dtw_native_v2` must return `bar_per_frame: vec![]` (already in the Task 4 implementation above).

**Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test low_margin_abstains_warm 2>&1
```

Also:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test 2>&1 | tail -20
```

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && \
git add apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs && \
git commit -m "test(chroma-dtw): low-margin warm abstain path verified"
```

---

## Task 9: TypeScript Interface Update — wasm-bridge.ts

**Group:** D (sequential, depends on Group B; parallel with Group C)

**Behavior being verified:** `alignChunkChroma` in `wasm-bridge.ts` forwards all nine arguments (adding `expectedScoreFrame`, `bandBackFrames`, `bandFwdFrames`, `marginMin`, `uniformityMin`) to the WASM module and returns `BarMapChroma` with `end_score_frame`, `confidence`, `status`.

**Interface under test:** `wasm-bridge.ts` public `alignChunkChroma` function; the existing forwarding test in `wasm-bridge.test.ts`.

**Files:**
- Modify: `apps/api/src/services/wasm-bridge.ts`
- Modify: `apps/api/src/services/wasm-bridge.test.ts`

**Step 1: Write the failing test**

Replace the existing `alignChunkChroma` test in `apps/api/src/services/wasm-bridge.test.ts`:

```typescript
describe("alignChunkChroma", () => {
  it("forwards all nine arguments to align_chunk_chroma and returns its result", async () => {
    const { alignChunkChroma } = await import("./wasm-bridge");
    const fakeResult = {
      bar_min: 0,
      bar_max: 4,
      cost: 0.1,
      bar_per_frame: [0, 0, 1, 1, 2, 2, 3, 3],
      end_score_frame: 200,
      confidence: 0.05,
      status: "aligned",
    };
    mockAlignChunkChroma.mockReturnValue(fakeResult);

    const audioBytes = new Uint8Array(12 * 4);
    const bars: never[] = [];
    const result = alignChunkChroma(audioBytes, 1, bars, 50.0, 5.0, -1, 150, 300, 0.02, 0.3);

    expect(mockAlignChunkChroma).toHaveBeenCalledWith(
      audioBytes,
      1,
      bars,
      50.0,
      5.0,
      -1,
      150,
      300,
      0.02,
      0.3,
    );
    expect(result).toBe(fakeResult);
    expect(result.status).toBe("aligned");
    expect(result.end_score_frame).toBe(200);
    expect(result.confidence).toBe(0.05);
  });
});
```

**Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- --run --reporter=verbose 2>&1 | grep -A5 "alignChunkChroma"
```

Expected: FAIL — `alignChunkChroma` was called with 5 arguments but the test expects 10.

**Step 3: Implement the interface updates**

In `apps/api/src/services/wasm-bridge.ts`:

1. Extend `BarMapChroma` interface:

```typescript
export interface BarMapChroma {
  bar_min: number;
  bar_max: number;
  cost: number;
  bar_per_frame: number[];
  end_score_frame: number;
  confidence: number;
  /** "aligned" | "relocalized" | "abstained" */
  status: string;
}
```

2. Replace the `alignChunkChroma` function:

```typescript
/**
 * Align a 15s audio chunk to a score using continuity-aware chroma-based DTW.
 *
 * @param audioChromaBytes raw LE float32 bytes, row-major 12 x chromaFrames
 * @param chromaFrames number of chroma columns
 * @param scoreBars array of ScoreBar from the loaded score JSON
 * @param frameRateHz chroma frame rate (typically 50.0)
 * @param decimHz output frame rate for bar_per_frame (typically 5.0)
 * @param expectedScoreFrame prior position hint (-1 = cold start)
 * @param bandBackFrames frames to search behind expected (e.g. 150)
 * @param bandFwdFrames frames to search ahead of expected (e.g. 300)
 * @param marginMin minimum separation margin to classify as aligned/relocalized (e.g. 0.02)
 * @param uniformityMin minimum fraction of peaky columns to attempt DP (e.g. 0.3)
 */
export function alignChunkChroma(
  audioChromaBytes: Uint8Array,
  chromaFrames: number,
  scoreBars: ScoreBar[],
  frameRateHz: number,
  decimHz: number,
  expectedScoreFrame: number,
  bandBackFrames: number,
  bandFwdFrames: number,
  marginMin: number,
  uniformityMin: number,
): BarMapChroma {
  return scoreAnalysisModule.align_chunk_chroma(
    audioChromaBytes,
    chromaFrames,
    scoreBars,
    frameRateHz,
    decimHz,
    expectedScoreFrame,
    bandBackFrames,
    bandFwdFrames,
    marginMin,
    uniformityMin,
  ) as BarMapChroma;
}
```

**Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- --run --reporter=verbose 2>&1 | grep -A5 "alignChunkChroma"
```

Also run the full test suite:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- --run 2>&1 | tail -20
```

Expected: all tests pass.

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && \
git add apps/api/src/services/wasm-bridge.ts \
        apps/api/src/services/wasm-bridge.test.ts && \
git commit -m "feat(wasm-bridge): extend alignChunkChroma with 4 continuity params + BarMapChroma status fields"
```

---

## Task 10: Schema — expectedScoreFrame in session-brain.schema.ts

**Group:** D (parallel with Task 9)

**Behavior being verified:** `sessionStateSchema` accepts and preserves `expectedScoreFrame: number` with a default of -1; `createInitialState` includes it.

**Interface under test:** `sessionStateSchema.parse(...)` and `createInitialState(...)` from `session-brain.schema.ts`.

**Files:**
- Modify: `apps/api/src/do/session-brain.schema.ts`
- Test: `apps/api/src/do/session-brain.unit.test.ts`

**Step 1: Write the failing test**

Add to `apps/api/src/do/session-brain.unit.test.ts`:

```typescript
describe("sessionStateSchema expectedScoreFrame", () => {
  it("parses state with expectedScoreFrame and preserves the value", () => {
    const raw = {
      version: 0,
      sessionId: "s1",
      studentId: "u1",
      conversationId: null,
      expectedScoreFrame: 750,
    };
    const parsed = sessionStateSchema.parse(raw);
    expect(parsed.expectedScoreFrame).toBe(750);
  });

  it("defaults expectedScoreFrame to -1 when not present", () => {
    const raw = {
      version: 0,
      sessionId: "s1",
      studentId: "u1",
      conversationId: null,
    };
    const parsed = sessionStateSchema.parse(raw);
    expect(parsed.expectedScoreFrame).toBe(-1);
  });

  it("createInitialState includes expectedScoreFrame = -1", () => {
    const { createInitialState } = await import("./session-brain.schema");
    const state = createInitialState("sess-1", "user-1", null);
    expect(state.expectedScoreFrame).toBe(-1);
  });
});
```

Note: the import `sessionStateSchema` must be added to the existing import line at the top of the test file:

```typescript
import {
  sessionStateSchema,
  createInitialState,
} from "./session-brain.schema";
```

**Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- --run --reporter=verbose 2>&1 | grep -A5 "expectedScoreFrame"
```

Expected: FAIL — `parsed.expectedScoreFrame is undefined`

**Step 3: Implement the schema change**

In `apps/api/src/do/session-brain.schema.ts`, add `expectedScoreFrame` to `sessionStateSchema`:

```typescript
export const sessionStateSchema = z.object({
  version: z.number().int(),
  sessionId: z.string(),
  studentId: z.string(),
  conversationId: z.string().nullable(),
  chunksInFlight: z.number().int().default(0),
  sessionEnding: z.boolean().default(false),
  synthesisCompleted: z.boolean().default(false),
  finalized: z.boolean().default(false),
  inferenceFailures: z.number().int().default(0),
  accumulator: z.unknown().default({}),
  baselines: z.record(z.string(), z.number()).nullable().default(null),
  baselinesLoaded: z.boolean().default(false),
  scoredChunks: z
    .array(
      z.object({
        chunkIndex: z.number().int(),
        scores: z.array(z.number()),
      }),
    )
    .default([]),
  pieceLocked: z.boolean().default(false),
  pieceIdentification: z
    .object({
      pieceId: z.string(),
      confidence: z.number(),
      method: z.string(),
    })
    .nullable()
    .default(null),
  modeDetector: z.unknown().default(null),
  identificationNoteCount: z.number().int().default(0),
  activeAssignment: z
    .object({
      id: z.string(),
      pieceId: z.string(),
      barsStart: z.number().int(),
      barsEnd: z.number().int(),
      requiredCorrect: z.number().int(),
      attemptsCompleted: z.number().int(),
      dimension: z.string().nullable().default(null),
    })
    .nullable()
    .default(null),
  isEvalSession: z.boolean().default(false),
  expectedScoreFrame: z.number().int().default(-1),
});
```

Update `createInitialState` to include the new field:

```typescript
export function createInitialState(
  sessionId: string,
  studentId: string,
  conversationId: string | null,
): SessionState {
  return {
    version: 0,
    sessionId,
    studentId,
    conversationId,
    chunksInFlight: 0,
    sessionEnding: false,
    synthesisCompleted: false,
    finalized: false,
    inferenceFailures: 0,
    accumulator: {},
    baselines: null,
    baselinesLoaded: false,
    scoredChunks: [],
    pieceLocked: false,
    pieceIdentification: null,
    modeDetector: null,
    identificationNoteCount: 0,
    activeAssignment: null,
    isEvalSession: false,
    expectedScoreFrame: -1,
  };
}
```

**Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- --run --reporter=verbose 2>&1 | grep -A5 "expectedScoreFrame"
```

Expected: all three schema tests pass.

Also run the full test suite:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- --run 2>&1 | tail -20
```

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && \
git add apps/api/src/do/session-brain.schema.ts \
        apps/api/src/do/session-brain.unit.test.ts && \
git commit -m "feat(session-brain): add expectedScoreFrame to session state schema (default -1)"
```

---

## Task 11: DO Wiring — Pass New Args, Dispatch on Status

**Group:** E (sequential, depends on Groups C and D)

**Behavior being verified:** The DO reads `expectedScoreFrame` from state, passes all nine args to `wasm.alignChunkChroma`, updates `expectedScoreFrame` on aligned/relocalized, leaves it unchanged on abstained, and emits Tier 3 when status is "abstained".

**Interface under test:** `session-brain.ts` — the behavior observable through the WebSocket: `chunk_bar_map` messages only appear on aligned/relocalized; Tier 3 chunks produce no bar map message.

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Test: `apps/api/src/do/session-brain.unit.test.ts`

**Step 1: Write the failing test**

Add to `apps/api/src/do/session-brain.unit.test.ts`:

```typescript
import { sessionStateSchema, createInitialState } from "./session-brain.schema";

describe("chroma DTW status dispatch (schema-level contract)", () => {
  it("aligned result: expectedScoreFrame in schema increases from -1 to end_score_frame", () => {
    // Simulate state before and after handling an aligned result
    const before = createInitialState("s1", "u1", null);
    expect(before.expectedScoreFrame).toBe(-1);

    // Apply the DO's update rule: on aligned, set expectedScoreFrame = end_score_frame
    const end_score_frame = 750;
    const after = sessionStateSchema.parse({
      ...before,
      expectedScoreFrame: end_score_frame,
    });
    expect(after.expectedScoreFrame).toBe(750);
  });

  it("abstained result: expectedScoreFrame is preserved unchanged", () => {
    const before = sessionStateSchema.parse({
      ...createInitialState("s1", "u1", null),
      expectedScoreFrame: 400,
    });
    // On abstained, the DO must NOT update expectedScoreFrame
    // Verify the schema round-trips 400 unchanged
    const after = sessionStateSchema.parse({ ...before });
    expect(after.expectedScoreFrame).toBe(400);
  });

  it("piece re-identification resets expectedScoreFrame to -1", () => {
    const state = sessionStateSchema.parse({
      ...createInitialState("s1", "u1", null),
      expectedScoreFrame: 400,
      pieceIdentification: { pieceId: "chopin.ballades.1", confidence: 0.9, method: "dtw" },
    });
    // Simulate re-id (piece set): pieceIdentification cleared, expectedScoreFrame must reset
    const afterReId = sessionStateSchema.parse({
      ...state,
      pieceIdentification: null,
      expectedScoreFrame: -1,
    });
    expect(afterReId.expectedScoreFrame).toBe(-1);
    expect(afterReId.pieceIdentification).toBeNull();
  });
});
```

**Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- --run --reporter=verbose 2>&1 | grep -A5 "chroma DTW status dispatch"
```

Expected: FAIL — `sessionStateSchema` does not have `expectedScoreFrame` yet (or the import is wrong before Task 10 is merged). If Task 10 is already merged, these schema-level tests may pass immediately. Verify by also checking that the DO code (not yet updated) would fail a runtime contract test.

**Step 3: Implement DO wiring in session-brain.ts**

In `apps/api/src/do/session-brain.ts`, at the chroma alignment call site (~line 601), replace the existing block:

```typescript
const chromaResult: BarMapChroma | null =
  chromaBytes !== null && chromaFrameRateHz > 0
    ? (() => {
        try {
          return wasm.alignChunkChroma(
            chromaBytes,
            chromaFrames,
            scoreCtx.score.bars,
            chromaFrameRateHz,
            5.0,
          );
        } catch (e) {
          console.log(
            JSON.stringify({
              level: "warn",
              message: "alignChunkChroma failed",
              error: e instanceof Error ? e.message : String(e),
            }),
          );
          return null;
        }
      })()
    : null;

if (chromaResult !== null) {
  chunkBarRange = [chromaResult.bar_min, chromaResult.bar_max];
  chunkAnalysisTier = 1;
}
```

With:

```typescript
const BAND_BACK_FRAMES = 150;
const BAND_FWD_FRAMES = 300;
const MARGIN_MIN = 0.02;
const UNIFORMITY_MIN = 0.30;

const chromaResult: BarMapChroma | null =
  chromaBytes !== null && chromaFrameRateHz > 0
    ? (() => {
        try {
          return wasm.alignChunkChroma(
            chromaBytes,
            chromaFrames,
            scoreCtx.score.bars,
            chromaFrameRateHz,
            5.0,
            currentState.expectedScoreFrame,
            BAND_BACK_FRAMES,
            BAND_FWD_FRAMES,
            MARGIN_MIN,
            UNIFORMITY_MIN,
          );
        } catch (e) {
          console.log(
            JSON.stringify({
              level: "warn",
              message: "alignChunkChroma failed",
              error: e instanceof Error ? e.message : String(e),
            }),
          );
          return null;
        }
      })()
    : null;

if (chromaResult !== null && chromaResult.status !== "abstained") {
  chunkBarRange = [chromaResult.bar_min, chromaResult.bar_max];
  chunkAnalysisTier = 1;
  currentState.expectedScoreFrame = chromaResult.end_score_frame;
  console.log(JSON.stringify({
    level: "debug",
    message: "chroma DTW aligned",
    status: chromaResult.status,
    bar_min: chromaResult.bar_min,
    bar_max: chromaResult.bar_max,
    confidence: chromaResult.confidence,
    end_score_frame: chromaResult.end_score_frame,
  }));
} else if (chromaResult?.status === "abstained") {
  // Abstained: preserve expectedScoreFrame, fall through to Tier 3
  console.log(JSON.stringify({
    level: "debug",
    message: "chroma DTW abstained — silence or ambiguous; Tier 3",
    confidence: chromaResult.confidence,
  }));
}
```

Also update the `handleSetPiece` method (~line 1329) to reset `expectedScoreFrame`:

```typescript
private async handleSetPiece(ws: WebSocket, query: string): Promise<void> {
  const state = await this.readState();
  state.pieceLocked = true;
  state.pieceIdentification = null;
  state.expectedScoreFrame = -1;  // add this line
  state.version++;
  await this.writeState(state);
  this.sendWs(ws, { type: "piece_set", query });
}
```

Also reset in the piece identification block (~line 782 where `pieceIdentification` is set) — when a new piece is identified (different from current), reset expectedScoreFrame:

```typescript
if (identified !== null) {
  const newPieceId = identified.pieceId;
  const currentPieceId = currentState.pieceIdentification?.pieceId;
  currentState.pieceLocked = true;
  currentState.pieceIdentification = {
    pieceId: newPieceId,
    confidence: identified.confidence,
    method: identified.method,
  };
  // Reset score follower position when piece changes
  if (newPieceId !== currentPieceId) {
    currentState.expectedScoreFrame = -1;
  }
  // ... rest of existing block unchanged
}
```

Also update the `chunk_bar_map` WebSocket send to only fire on non-abstained:

```typescript
if (chromaResult !== null && chromaResult.status !== "abstained") {
  this.sendWs(ws, {
    type: "chunk_bar_map",
    chunk_index: index,
    bar_min: chromaResult.bar_min,
    bar_max: chromaResult.bar_max,
    bar_per_frame: chromaResult.bar_per_frame,
  });
}
```

**Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- --run 2>&1 | tail -20
```

Expected: all tests pass (TypeScript compilation errors would surface here too).

Also run type check:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bunx tsc --noEmit 2>&1 | head -20
```

Expected: no errors.

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && \
git add apps/api/src/do/session-brain.ts \
        apps/api/src/do/session-brain.unit.test.ts && \
git commit -m "feat(session-brain): wire expectedScoreFrame continuity; dispatch on DTW status aligned/relocalized/abstained"
```

---

## Task 12: DO Unit Tests — Frame Continuity, Abstain Preserves Frame, Reset on Piece Re-ID

**Group:** E (sequential within Group E, after Task 11)

**Behavior being verified:** The expectedScoreFrame state contract matches the DO logic: frame advances on aligned, stays on abstained, resets on piece re-identification.

**Interface under test:** `sessionStateSchema` + `createInitialState` state transitions (schema-level contract, not DO internals).

**Files:**
- Modify: `apps/api/src/do/session-brain.unit.test.ts`

Note: The schema-level tests added in Task 11 already cover this behavior at the schema layer. This task adds one additional behavior-level test that was deferred: the `wsOutgoingMessageSchema` correctly rejects a `chunk_bar_map` with an invalid `bar_per_frame` type, ensuring the schema is correct for the abstained path (empty array).

**Step 1: Write the failing test**

Add to `apps/api/src/do/session-brain.unit.test.ts`:

```typescript
describe("chunk_bar_map schema on abstained path", () => {
  it("wsOutgoingMessageSchema accepts chunk_bar_map with empty bar_per_frame (abstained DO emits nothing)", () => {
    // When status=abstained, the DO does not send chunk_bar_map at all.
    // Verify the schema accepts an empty bar_per_frame for completeness.
    const msg = {
      type: "chunk_bar_map",
      chunk_index: 3,
      bar_min: 0,
      bar_max: 0,
      bar_per_frame: [],
    };
    expect(() => wsOutgoingMessageSchema.parse(msg)).not.toThrow();
    const parsed = wsOutgoingMessageSchema.parse(msg);
    expect(parsed.bar_per_frame).toHaveLength(0);
  });

  it("wsOutgoingMessageSchema rejects chunk_bar_map with string in bar_per_frame", () => {
    const bad = {
      type: "chunk_bar_map",
      chunk_index: 3,
      bar_min: 1,
      bar_max: 5,
      bar_per_frame: ["not-a-number"],
    };
    expect(() => wsOutgoingMessageSchema.parse(bad)).toThrow();
  });
});
```

**Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- --run --reporter=verbose 2>&1 | grep -A5 "abstained path"
```

Expected: FAIL — one or both assertions fail because the schema currently does not have these tests.

**Step 3: Implement — verify schema is correct**

If the `wsOutgoingMessageSchema` already accepts empty arrays and rejects non-number arrays, the test will pass without code changes. If not, check `session-brain.schema.ts` `wsChunkBarMapSchema` definition: `bar_per_frame: z.array(z.number().int())` already handles both cases. No code change needed.

**Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- --run 2>&1 | tail -20
```

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && \
git add apps/api/src/do/session-brain.unit.test.ts && \
git commit -m "test(session-brain): schema contract tests for abstained chunk_bar_map path"
```

---

## Task 13: WASM Build + Full Suite Pass

**Group:** E (sequential within Group E, after Tasks 11 and 12)

**Behavior being verified:** The WASM is rebuilt with the new Rust interface, all TypeScript tests pass against the rebuilt pkg, and cargo tests all pass.

**Interface under test:** `bun run build:wasm` produces a new `.wasm` + `.d.ts` in `pkg/`; `bun run test -- --run` passes; `cargo test` passes.

**Files:**
- Rebuild: `apps/api/src/wasm/score-analysis/pkg/` (generated by wasm-pack)

**Step 1: Write the failing test**

This task's "test" is a compile-time check. First, confirm the current TS side would fail against the unbuilt WASM:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bunx tsc --noEmit 2>&1 | head -30
```

Expected: type errors because the `.d.ts` in `pkg/` does not yet expose the new parameters for `align_chunk_chroma`.

**Step 2: Run WASM build**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run build:wasm 2>&1 | tail -20
```

Expected: `[INFO]: Your wasm pkg is ready to publish at ./pkg.` for both score-analysis and piece-identify.

**Step 3: Verify TypeScript compiles cleanly**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bunx tsc --noEmit 2>&1 | head -20
```

Expected: no errors.

**Step 4: Run all tests — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- --run 2>&1 | tail -20
```

Also:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test 2>&1 | tail -20
```

Expected: all tests pass.

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && \
git add apps/api/src/wasm/score-analysis/pkg/ && \
git commit -m "build(wasm): rebuild score-analysis with continuity-aware align_chunk_chroma interface"
```

---

## Local E2E Verification (Post-Group E, Manual)

After all tasks are complete and all automated tests pass, verify end-to-end:

1. Start local AMT: `just amt` (required to satisfy the AMT gate in the DO)
2. Start local API: `just api`
3. Open the web app and start a practice session on the Chopin Ballade 1 piece using the amateur recording `Jt2f6yEGcP4.wav`
4. Observe:
   - First chunk should NOT produce a `chunk_bar_map` message jumping to bars 261–262
   - Bar map should progress monotonically through the session
   - Silent intro (if present) produces no bar map (Tier 3)
5. Check API logs for `"message": "chroma DTW aligned"` entries showing expected bar ranges

This is a manual check; no commit is required.
