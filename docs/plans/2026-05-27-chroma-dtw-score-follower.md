# Chroma-DTW Score Follower Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Replace the broken note-level score follower with a stateless chroma-based subsequence DTW that aligns each 15-second audio chunk to the score, producing a bar range and 5 Hz per-frame bar mapping suitable for live cursor following.
**Spec:** docs/specs/2026-05-27-chroma-dtw-score-follower-design.md
**Style:** Follow `apps/api/TS_STYLE.md` for TypeScript; Rust follows existing `score-analysis` patterns.

---

## Task Groups

- **Group 0 (must run first, no parallelism):** Task 0 — Fixture generator + committed binary fixtures
- **Group 1 (parallel, all depend on Group 0 completing):** Task 1 (Rust chroma_dtw), Task 2 (Python chroma_feature), Task 3 (TypeScript wasm-bridge)
- **Group 2 (sequential, depends on Group 1):** Task 4 (inference.ts MuqResult extension), Task 5 (session-brain.schema.ts cleanup)
- **Group 3 (sequential, depends on Group 2):** Task 6 (session-brain.ts wiring), Task 7 (session-brain.unit.test.ts)
- **Group 4 (sequential, depends on Group 3):** Task 8 (delete score_follower.rs + lib.rs cleanup)

---

## Task 0: Fixture Generator + Committed Binary Fixtures

**Group:** 0 (must complete before all other tasks)

**Behavior being verified:** A Python script generates three binary/JSON fixture files per test case that the Rust cargo tests will load and assert against. Two fixture sets are committed: `ballade1_forward_2min` (audio[0..120s], verifies forward-play alignment to early bars) and `ballade1_coldstart_111s` (audio[111..126s], verifies cold-start at bar ~30).

**Interface under test:** The generator script itself — run it once, inspect `expected.json`, confirm values are sane, commit the binary outputs.

**Files:**
- Create: `apps/api/src/wasm/score-analysis/tests/fixtures/generate.py`
- Create: `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_forward_2min/audio_chroma.bin`
- Create: `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_forward_2min/score_bars.json`
- Create: `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_forward_2min/expected.json`
- Create: `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_coldstart_111s/audio_chroma.bin`
- Create: `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_coldstart_111s/score_bars.json`
- Create: `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_coldstart_111s/expected.json`

- [ ] **Step 1: Write the generator script**

```python
# apps/api/src/wasm/score-analysis/tests/fixtures/generate.py
"""
Fixture generator for chroma-DTW cargo tests.

Usage:
  uv run python apps/api/src/wasm/score-analysis/tests/fixtures/generate.py

Reads from the project's canonical eval audio and score JSON (same paths used
by apps/inference/score-align-spike/spike.py). Writes three files per fixture
into tests/fixtures/{slug}/:
  audio_chroma.bin   - raw float32 LE, row-major 12 x N
  score_bars.json    - JSON array of ScoreBar objects for the score slice
  expected.json      - bounds for bar_min, bar_max, n_frames, cost

Run once; commit the outputs. Not part of CI.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import librosa
import numpy as np

SR = 22050
HOP = 441        # 50 Hz at 22050 Hz
FRAME_RATE = SR / HOP   # ~50.0 Hz
DECIM_HZ = 5.0

AUDIO_WAV = Path("model/data/evals/skill_eval/chopin_ballade_1/audio/HlHBUxlcWfk.wav")
SCORE_JSON = Path("model/data/scores/chopin.ballades.1.json")

FIXTURES_DIR = Path("apps/api/src/wasm/score-analysis/tests/fixtures")

CASES = [
    {
        "slug": "ballade1_forward_2min",
        "start_s": 0.0,
        "dur_s": 120.0,
        # bars 1..~30 expected (forward play, 0-2 min)
        "bar_min_lo": 1,
        "bar_min_hi": 5,
        "bar_max_lo": 20,
        "bar_max_hi": 45,
        "cost_hi": 0.30,
    },
    {
        "slug": "ballade1_coldstart_111s",
        "start_s": 111.0,
        "dur_s": 15.0,
        # bars ~30-35 expected (cold-start at 111s, per spike results)
        "bar_min_lo": 25,
        "bar_min_hi": 35,
        "bar_max_lo": 28,
        "bar_max_hi": 40,
        "cost_hi": 0.30,
    },
]


def build_audio_chroma(wav_path: Path, start_s: float, dur_s: float) -> np.ndarray:
    """Load a slice of audio and compute L2-normalized chroma at 50 Hz."""
    y, _ = librosa.load(
        str(wav_path), sr=SR, mono=True,
        offset=start_s, duration=dur_s,
    )
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    chroma = chroma.astype(np.float32) + 1e-3
    norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma /= norm
    return chroma  # shape (12, N), float32


def load_score_bars(score_json_path: Path) -> list[dict]:
    data = json.loads(score_json_path.read_text())
    return data["bars"]


def main() -> None:
    root = Path(__file__).parents[6]  # project root
    wav = root / AUDIO_WAV
    score_json = root / SCORE_JSON

    if not wav.exists():
        print(f"ERROR: audio not found at {wav}", file=sys.stderr)
        sys.exit(1)
    if not score_json.exists():
        print(f"ERROR: score not found at {score_json}", file=sys.stderr)
        sys.exit(1)

    all_bars = load_score_bars(score_json)

    for case in CASES:
        slug = case["slug"]
        out_dir = root / FIXTURES_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {slug} ...")

        # Audio chroma
        chroma = build_audio_chroma(wav, case["start_s"], case["dur_s"])
        n_frames = chroma.shape[1]

        # Write audio_chroma.bin: row-major float32 LE, 12 * n_frames floats
        bin_bytes = struct.pack(f"<{12 * n_frames}f", *chroma.flatten().tolist())
        (out_dir / "audio_chroma.bin").write_bytes(bin_bytes)
        print(f"  audio_chroma.bin: 12 x {n_frames} frames ({len(bin_bytes)} bytes)")

        # score_bars.json: full bars array (Rust reads the whole score for score-chroma build)
        (out_dir / "score_bars.json").write_text(json.dumps(all_bars))
        print(f"  score_bars.json: {len(all_bars)} bars")

        # Compute decimated frame count
        decim_step = int(round(FRAME_RATE / DECIM_HZ))
        decim_n = (n_frames + decim_step - 1) // decim_step

        # expected.json
        expected = {
            "bar_min_lo": case["bar_min_lo"],
            "bar_min_hi": case["bar_min_hi"],
            "bar_max_lo": case["bar_max_lo"],
            "bar_max_hi": case["bar_max_hi"],
            "n_frames": n_frames,
            "decim_n": decim_n,
            "cost_hi": case["cost_hi"],
        }
        (out_dir / "expected.json").write_text(json.dumps(expected, indent=2))
        print(f"  expected.json: {expected}")

    print("Done. Commit the fixtures directory.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run python apps/api/src/wasm/score-analysis/tests/fixtures/generate.py
```

Expected: three files per fixture slug printed to stdout, no errors. Inspect `expected.json` values manually to confirm bar ranges are sane (ballade1_coldstart_111s should show bar_min ~30).

- [ ] **Step 3: Commit the generator and all binary fixtures**

```bash
git add apps/api/src/wasm/score-analysis/tests/fixtures/ && git commit -m "test(score-follower): add chroma-DTW fixture generator and committed binary fixtures"
```

---

## Task 1: Rust `align_chunk_chroma` WASM Entry Point

**Group:** 1 (parallel with Tasks 2, 3; depends on Task 0)

**Behavior being verified:** `align_chunk_chroma` called with the committed `ballade1_coldstart_111s` fixture returns a `BarMapChroma` where `bar_min` and `bar_max` are within the expected bounds from `expected.json`, `bar_per_frame.len()` equals `expected.decim_n`, and `cost` is below `expected.cost_hi`.

**Interface under test:** The `#[wasm_bindgen]` function `align_chunk_chroma` via Rust integration test (calling internal Rust API directly without WASM boundary since cargo tests run natively).

**Files:**
- Create: `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs`
- Modify: `apps/api/src/wasm/score-analysis/src/types.rs`
- Modify: `apps/api/src/wasm/score-analysis/src/lib.rs`

- [ ] **Step 1: Write the failing test**

Add a new integration test file:

```rust
// apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs
//
// Cargo integration test: loads committed binary fixtures and asserts
// align_chunk_chroma_native returns sane bar ranges.
//
// Run with: cargo test chroma_dtw_roundtrip
// (in apps/api/src/wasm/score-analysis/)

use score_analysis::chroma_dtw_native;
use score_analysis::types::{BarMapChroma, ScoreBar};
use std::path::Path;

fn load_fixture(slug: &str) -> (Vec<f32>, u32, Vec<ScoreBar>, serde_json::Value) {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(slug);

    // audio_chroma.bin: raw LE f32 bytes, row-major 12 x n_frames
    let bin = std::fs::read(base.join("audio_chroma.bin"))
        .unwrap_or_else(|_| panic!("missing audio_chroma.bin for {slug}"));
    let floats: Vec<f32> = bin
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let n_frames = (floats.len() / 12) as u32;

    let bars_json = std::fs::read_to_string(base.join("score_bars.json"))
        .unwrap_or_else(|_| panic!("missing score_bars.json for {slug}"));
    let score_bars: Vec<ScoreBar> = serde_json::from_str(&bars_json)
        .unwrap_or_else(|e| panic!("score_bars.json parse error for {slug}: {e}"));

    let expected_json = std::fs::read_to_string(base.join("expected.json"))
        .unwrap_or_else(|_| panic!("missing expected.json for {slug}"));
    let expected: serde_json::Value = serde_json::from_str(&expected_json).unwrap();

    (floats, n_frames, score_bars, expected)
}

#[test]
fn chroma_dtw_roundtrip_coldstart() {
    let (audio_f32, n_audio, score_bars, expected) = load_fixture("ballade1_coldstart_111s");

    let result: BarMapChroma =
        chroma_dtw_native(&audio_f32, n_audio, &score_bars, 50.0, 5.0)
            .expect("align_chunk_chroma_native should not fail on valid fixture");

    let bar_min_lo = expected["bar_min_lo"].as_u64().unwrap() as u32;
    let bar_min_hi = expected["bar_min_hi"].as_u64().unwrap() as u32;
    let bar_max_lo = expected["bar_max_lo"].as_u64().unwrap() as u32;
    let bar_max_hi = expected["bar_max_hi"].as_u64().unwrap() as u32;
    let decim_n = expected["decim_n"].as_u64().unwrap() as usize;
    let cost_hi = expected["cost_hi"].as_f64().unwrap() as f32;

    assert!(
        result.bar_min >= bar_min_lo && result.bar_min <= bar_min_hi,
        "bar_min={} not in [{bar_min_lo}, {bar_min_hi}]",
        result.bar_min
    );
    assert!(
        result.bar_max >= bar_max_lo && result.bar_max <= bar_max_hi,
        "bar_max={} not in [{bar_max_lo}, {bar_max_hi}]",
        result.bar_max
    );
    assert_eq!(
        result.bar_per_frame.len(),
        decim_n,
        "bar_per_frame length mismatch"
    );
    assert!(
        result.cost < cost_hi,
        "cost={} not below {cost_hi}",
        result.cost
    );
    // Monotone non-decreasing check
    for w in result.bar_per_frame.windows(2) {
        assert!(
            w[1] >= w[0],
            "bar_per_frame is not non-decreasing at {:?}",
            w
        );
    }
}

#[test]
fn chroma_dtw_roundtrip_forward() {
    let (audio_f32, n_audio, score_bars, expected) = load_fixture("ballade1_forward_2min");

    let result: BarMapChroma =
        chroma_dtw_native(&audio_f32, n_audio, &score_bars, 50.0, 5.0)
            .expect("align_chunk_chroma_native should not fail on valid fixture");

    let bar_min_lo = expected["bar_min_lo"].as_u64().unwrap() as u32;
    let bar_min_hi = expected["bar_min_hi"].as_u64().unwrap() as u32;
    let bar_max_lo = expected["bar_max_lo"].as_u64().unwrap() as u32;
    let bar_max_hi = expected["bar_max_hi"].as_u64().unwrap() as u32;
    let decim_n = expected["decim_n"].as_u64().unwrap() as usize;
    let cost_hi = expected["cost_hi"].as_f64().unwrap() as f32;

    assert!(
        result.bar_min >= bar_min_lo && result.bar_min <= bar_min_hi,
        "bar_min={} not in [{bar_min_lo}, {bar_min_hi}]",
        result.bar_min
    );
    assert!(
        result.bar_max >= bar_max_lo && result.bar_max <= bar_max_hi,
        "bar_max={} not in [{bar_max_lo}, {bar_max_hi}]",
        result.bar_max
    );
    assert_eq!(result.bar_per_frame.len(), decim_n);
    assert!(result.cost < cost_hi, "cost={} not below {cost_hi}", result.cost);
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test chroma_dtw_roundtrip 2>&1 | head -30
```

Expected: FAIL — `error[E0432]: unresolved import score_analysis::chroma_dtw_native` (function and type do not exist yet).

- [ ] **Step 3: Implement the minimum to make the tests pass**

**3a. Add `BarMapChroma` to `types.rs`** (append after the existing `AlignChunkResult` block):

```rust
// --- Chroma DTW output ---

/// Output of align_chunk_chroma: bar range + per-decimated-frame bar mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarMapChroma {
    /// Minimum bar number touched by the warping path.
    pub bar_min: u32,
    /// Maximum bar number touched by the warping path.
    pub bar_max: u32,
    /// Mean cosine distance along the warping path (lower = better alignment).
    pub cost: f32,
    /// Bar number for each decimated frame (at decim_hz, e.g. 5 Hz).
    /// Length = ceil(n_audio_frames / decim_step).
    pub bar_per_frame: Vec<u32>,
}
```

**3b. Create `chroma_dtw.rs`**:

```rust
// apps/api/src/wasm/score-analysis/src/chroma_dtw.rs
//
// Stateless per-chunk chroma subsequence DTW.
// Entry point: align_chunk_chroma (wasm_bindgen) + chroma_dtw_native (Rust tests).

use crate::types::{BarMapChroma, ScoreBar};

/// Map a score-side frame index to its bar number using bar start_seconds.
fn frame_to_bar(frame_idx: usize, frame_rate: f32, bars: &[ScoreBar]) -> u32 {
    let t = frame_idx as f32 / frame_rate;
    let mut last = bars.first().map(|b| b.bar_number).unwrap_or(1);
    for bar in bars {
        if bar.start_seconds as f32 <= t {
            last = bar.bar_number;
        } else {
            break;
        }
    }
    last
}

/// Build a 12 x N score chroma matrix from the bar note list.
/// Each note contributes its pitch class to every frame it spans.
/// Adds 1e-3 floor, then L2-normalizes per column.
fn build_score_chroma(bars: &[ScoreBar], frame_rate: f32) -> Vec<f32> {
    // Find total duration from last note end
    let mut end_time: f32 = 0.0;
    for bar in bars {
        for note in &bar.notes {
            let note_end = note.onset_seconds as f32 + note.duration_seconds.max(0.05) as f32;
            if note_end > end_time {
                end_time = note_end;
            }
        }
    }
    if end_time <= 0.0 {
        return Vec::new();
    }
    let n_frames = (end_time * frame_rate).ceil() as usize + 1;
    let mut chroma = vec![0.0_f32; 12 * n_frames];

    for bar in bars {
        for note in &bar.notes {
            let onset = note.onset_seconds as f32;
            let dur = (note.duration_seconds as f32).max(0.05);
            let pc = (note.pitch % 12) as usize;
            let f0 = (onset * frame_rate).floor() as usize;
            let f1 = ((onset + dur) * frame_rate).ceil() as usize;
            let f0 = f0.min(n_frames);
            let f1 = f1.min(n_frames);
            for f in f0..f1 {
                chroma[pc * n_frames + f] += 1.0;
            }
        }
    }

    // Floor + L2-normalize per column
    for f in 0..n_frames {
        let mut norm_sq: f32 = 0.0;
        for pc in 0..12 {
            chroma[pc * n_frames + f] += 1e-3;
            norm_sq += chroma[pc * n_frames + f] * chroma[pc * n_frames + f];
        }
        let norm = norm_sq.sqrt() + 1e-9;
        for pc in 0..12 {
            chroma[pc * n_frames + f] /= norm;
        }
    }

    chroma
}

/// Cosine distance between two column vectors of length 12, stored row-major.
/// `a`: row-major 12 x n_a, column j.  `b`: row-major 12 x n_b, column i.
#[inline]
fn cosine_dist(
    a: &[f32], n_a: usize, j: usize,
    b: &[f32], n_b: usize, i: usize,
) -> f32 {
    let mut dot = 0.0_f32;
    for pc in 0..12 {
        dot += a[pc * n_a + j] * b[pc * n_b + i];
    }
    (1.0 - dot).max(0.0)
}

/// Subsequence DTW: finds best contiguous region of `score` (long) matching `audio` (short).
/// Step pattern: (1,0), (0,1), (1,1) — standard monotonic.
/// Returns (warping_path_vec, mean_cost).
/// `warping_path_vec`: Vec of (score_frame, audio_frame) pairs, forward order.
fn subseq_dtw(
    audio: &[f32], n_audio: usize,  // row-major 12 x n_audio
    score: &[f32], n_score: usize,  // row-major 12 x n_score
) -> (Vec<(usize, usize)>, f32) {
    // D[i][j] = accumulated cost, audio axis i (rows), score axis j (cols)
    // Subsequence DTW: initialize first row to local cost (free start on score axis).
    let n_a = n_audio;
    let n_s = n_score;

    let mut d = vec![f32::INFINITY; n_a * n_s];
    let mut p: Vec<(i32, i32)> = vec![(0, 0); n_a * n_s]; // predecessor (da, ds)

    // idx helper: audio row i, score col j -> flat index
    let idx = |i: usize, j: usize| i * n_s + j;

    // Initialize first audio row — subsequence: free start anywhere on score
    for j in 0..n_s {
        d[idx(0, j)] = cosine_dist(audio, n_a, 0, score, n_s, j);
    }

    // Fill
    for i in 1..n_a {
        for j in 0..n_s {
            let c = cosine_dist(audio, n_a, i, score, n_s, j);
            // Predecessors: (i-1, j), (i, j-1), (i-1, j-1)
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

    // Find best end on last audio row (minimum over all score positions)
    let last_row = n_a - 1;
    let j_end = (0..n_s)
        .min_by(|&ja, &jb| d[idx(last_row, ja)].partial_cmp(&d[idx(last_row, jb)]).unwrap())
        .unwrap_or(0);

    // Backtrack
    let mut path = Vec::with_capacity(n_a + n_s);
    let mut i = last_row as i32;
    let mut j = j_end as i32;
    while i >= 0 {
        path.push((j as usize, i as usize)); // (score_frame, audio_frame)
        let (di, dj) = p[idx(i as usize, j as usize)];
        if di == 0 && dj == 0 {
            // Should not happen; safety exit
            break;
        }
        i += di;
        j += dj;
    }
    path.reverse();

    // Mean cost along path
    let mean_cost = if path.is_empty() {
        1.0
    } else {
        let total: f32 = path
            .iter()
            .map(|&(sf, af)| cosine_dist(audio, n_a, af, score, n_s, sf))
            .sum();
        total / path.len() as f32
    };

    (path, mean_cost)
}

/// Pure Rust core, callable from cargo tests without WASM boundary.
pub fn chroma_dtw_native(
    audio_f32: &[f32],   // row-major float32, 12 x n_audio
    n_audio: u32,
    score_bars: &[ScoreBar],
    frame_rate_hz: f32,
    decim_hz: f32,
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

    let score_chroma = build_score_chroma(score_bars, frame_rate_hz);
    let n_s = score_chroma.len() / 12;
    if n_s == 0 {
        return Err("score has no notes".to_string());
    }

    let (path, mean_cost) = subseq_dtw(audio_f32, n_a, &score_chroma, n_s);

    // Build bar_per_frame at decim_hz by sampling the warping path
    let decim_step = (frame_rate_hz / decim_hz).round() as usize;
    let decim_step = decim_step.max(1);
    let decim_n = (n_a + decim_step - 1) / decim_step;

    // path is (score_frame, audio_frame) pairs; build audio_frame -> score_frame map
    let mut audio_to_score = vec![0usize; n_a];
    for &(sf, af) in &path {
        if af < n_a {
            audio_to_score[af] = sf;
        }
    }
    // Fill gaps (path may not cover every audio frame)
    let mut last_sf = path.first().map(|&(sf, _)| sf).unwrap_or(0);
    for af in 0..n_a {
        if audio_to_score[af] == 0 && af > 0 {
            audio_to_score[af] = last_sf;
        } else {
            last_sf = audio_to_score[af];
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
    })
}

// ─── WASM entry point ──────────────────────────────────────────────────────

use wasm_bindgen::prelude::*;

/// Align a 15s audio chunk to a score using chroma-based subsequence DTW.
///
/// `audio_bytes`: raw LE float32 bytes, row-major 12 x n_audio
/// `n_audio`: number of audio frames (columns)
/// `score_bars_js`: JS array of ScoreBar objects
/// `frame_rate_hz`: frame rate of audio chroma (typically 50.0)
/// `decim_hz`: output frame rate for bar_per_frame (typically 5.0)
///
/// Returns `BarMapChroma` on success, error string on failure.
#[wasm_bindgen]
pub fn align_chunk_chroma(
    audio_bytes: &[u8],
    n_audio: u32,
    score_bars_js: JsValue,
    frame_rate_hz: f32,
    decim_hz: f32,
) -> Result<JsValue, JsValue> {
    // Reinterpret bytes as f32 LE
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

    let result = chroma_dtw_native(&audio_f32, n_audio, &score_bars, frame_rate_hz, decim_hz)
        .map_err(|e| JsValue::from_str(&e))?;

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}
```

**3c. Modify `types.rs`**: append the `BarMapChroma` struct after the existing `AlignChunkResult` block (line 203).

```rust
// --- Chroma DTW output ---

/// Output of align_chunk_chroma: bar range + per-decimated-frame bar mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarMapChroma {
    pub bar_min: u32,
    pub bar_max: u32,
    pub cost: f32,
    pub bar_per_frame: Vec<u32>,
}
```

**3d. Modify `lib.rs`**: add `mod chroma_dtw;`, export `align_chunk_chroma`, and re-export `chroma_dtw_native` for integration tests. Add after the existing `mod score_follower;` line:

```rust
mod chroma_dtw;

// Re-export pure Rust core for integration tests (not wasm_bindgen — no JsValue in tests)
pub use chroma_dtw::chroma_dtw_native;
```

Also update the `#[wasm_bindgen]` export block at the bottom by adding the new function. The `align_chunk_chroma` is already exported via `#[wasm_bindgen]` inside `chroma_dtw.rs` itself, so `lib.rs` does not need to re-declare it — just ensure the module is declared.

**3e. Create integration test file** at `apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs` (the code written in Step 1).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test chroma_dtw_roundtrip 2>&1
```

Expected: PASS — both `chroma_dtw_roundtrip_coldstart` and `chroma_dtw_roundtrip_forward` pass.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/wasm/score-analysis/src/chroma_dtw.rs apps/api/src/wasm/score-analysis/src/types.rs apps/api/src/wasm/score-analysis/src/lib.rs apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs && git commit -m "feat(score-follower): add chroma_dtw_native and align_chunk_chroma WASM entry point"
```

---

## Task 2: Python `chroma_feature` Helper

**Group:** 1 (parallel with Tasks 1, 3; depends on Task 0)

**Behavior being verified:** `chroma_feature(y, sr)` returns a tuple of `(bytes, int)` where bytes are row-major float32 LE and int is the frame count; output shape, dtype, and dominant pitch class for a synthetic pure-tone waveform are correct.

**Interface under test:** `chroma_feature` function imported directly.

**Files:**
- Create: `apps/inference/muq/chroma.py`
- Create: `apps/inference/muq/test_chroma.py`
- Modify: `apps/inference/muq/requirements.txt` (add `pytest` if absent)

- [ ] **Step 1: Write the failing test**

```python
# apps/inference/muq/test_chroma.py
"""Pytest suite for chroma_feature."""
import struct
import numpy as np
import pytest

from chroma import chroma_feature

SR = 22050
HOP = 441


def make_sine(freq_hz: float, duration_s: float = 1.0) -> np.ndarray:
    """Generate a mono float32 sine wave at 22050 Hz."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


class TestChromaFeature:
    def test_output_is_tuple_of_bytes_and_int(self):
        y = make_sine(440.0)
        result = chroma_feature(y, SR)
        assert isinstance(result, tuple) and len(result) == 2
        b, n = result
        assert isinstance(b, (bytes, bytearray))
        assert isinstance(n, int)

    def test_frame_count_matches_bytes_length(self):
        y = make_sine(440.0)
        b, n = chroma_feature(y, SR)
        # bytes = 12 rows * n frames * 4 bytes per float32
        assert len(b) == 12 * n * 4

    def test_frame_count_matches_librosa_expectation(self):
        duration_s = 1.0
        y = make_sine(440.0, duration_s)
        _, n = chroma_feature(y, SR)
        expected_n = int(np.ceil(len(y) / HOP))
        # Allow ±1 frame tolerance for rounding
        assert abs(n - expected_n) <= 1

    def test_output_is_row_major_float32(self):
        y = make_sine(440.0)
        b, n = chroma_feature(y, SR)
        floats = struct.unpack(f"<{12 * n}f", b)
        # All values should be finite and in [0, 1] (L2-normalized columns)
        arr = np.array(floats, dtype=np.float32).reshape(12, n)
        assert np.all(np.isfinite(arr))
        # Column norms should be ~1.0 (L2-normalized)
        col_norms = np.linalg.norm(arr, axis=0)
        np.testing.assert_allclose(col_norms, 1.0, atol=0.01)

    def test_dominant_pitch_class_for_a440(self):
        # A440 = MIDI pitch 69, pitch class 9 (A)
        y = make_sine(440.0, 2.0)
        b, n = chroma_feature(y, SR)
        arr = np.frombuffer(b, dtype="<f4").reshape(12, n)
        dominant_pc = int(arr.mean(axis=1).argmax())
        assert dominant_pc == 9, f"Expected pitch class 9 (A), got {dominant_pc}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/inference/muq && uv run pytest test_chroma.py -v 2>&1 | head -30
```

Expected: FAIL — `ModuleNotFoundError: No module named 'chroma'`

- [ ] **Step 3: Implement `chroma.py`**

```python
# apps/inference/muq/chroma.py
"""
chroma_feature: compute 12-row chroma at 50 Hz from a mono float32 waveform.

Used by handler.py to attach chroma data to MuQ inference responses.
"""
from __future__ import annotations

import struct

import librosa
import numpy as np

_HOP = 441  # 50 Hz at 22050 Hz


def chroma_feature(y: np.ndarray, sr: int) -> tuple[bytes, int]:
    """Compute L2-normalized chroma and serialize as raw float32 bytes.

    Args:
        y: mono float32 waveform at `sr` Hz.
        sr: sample rate in Hz (must match `y`).

    Returns:
        (raw_bytes, n_frames) where raw_bytes is row-major float32 LE,
        shape (12, n_frames), and n_frames is the number of chroma columns.

    Raises:
        ValueError: if `y` is empty or `sr` is zero.
    """
    if len(y) == 0:
        raise ValueError("chroma_feature: waveform is empty")
    if sr <= 0:
        raise ValueError(f"chroma_feature: invalid sample rate {sr}")

    hop = _HOP if sr == 22050 else max(1, sr // 50)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chroma = chroma.astype(np.float32) + 1e-3
    norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma /= norm

    n_frames = chroma.shape[1]
    # Row-major: chroma[0, :], chroma[1, :], ..., chroma[11, :]
    flat = chroma.flatten()
    raw_bytes = struct.pack(f"<{12 * n_frames}f", *flat.tolist())
    return raw_bytes, n_frames
```

Also add `pytest` to `requirements.txt` if it is not already present. Current `requirements.txt` does not contain `pytest`, so append it:

```
pytest>=8.0
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/inference/muq && uv run pytest test_chroma.py -v 2>&1
```

Expected: PASS — all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/inference/muq/chroma.py apps/inference/muq/test_chroma.py apps/inference/muq/requirements.txt && git commit -m "feat(muq): add chroma_feature helper with pytest coverage"
```

---

## Task 3: TypeScript `alignChunkChroma` in `wasm-bridge.ts`

**Group:** 1 (parallel with Tasks 1, 2; depends on Task 0)

**Behavior being verified:** `alignChunkChroma` called with a fixture-derived `Uint8Array` (loaded from the committed `ballade1_coldstart_111s/audio_chroma.bin`) and the corresponding `score_bars.json` returns a `BarMapChroma` with `bar_min` in [25, 35] and `bar_per_frame.length > 0`.

**Interface under test:** `alignChunkChroma` exported from `wasm-bridge.ts`.

**Files:**
- Modify: `apps/api/src/services/wasm-bridge.ts`
- Create: `apps/api/src/services/wasm-bridge.test.ts`

Note: `wasm-bridge.test.ts` belongs in the node pool (no CF bindings needed; pure function logic). Add it to the `src/services/**/*.test.ts` glob that `vitest.node.config.ts` already includes.

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/services/wasm-bridge.test.ts
import { describe, expect, it } from "vitest";
import * as path from "node:path";
import * as fs from "node:fs";
import { alignChunkChroma } from "./wasm-bridge";
import type { ScoreBar } from "./wasm-bridge";

const FIXTURE_DIR = path.resolve(
  __dirname,
  "../wasm/score-analysis/tests/fixtures/ballade1_coldstart_111s",
);

function loadFixture(): {
  audioBytes: Uint8Array;
  scoreBars: ScoreBar[];
  expected: {
    bar_min_lo: number;
    bar_min_hi: number;
    bar_max_lo: number;
    bar_max_hi: number;
    decim_n: number;
    cost_hi: number;
  };
} {
  const binPath = path.join(FIXTURE_DIR, "audio_chroma.bin");
  const barsPath = path.join(FIXTURE_DIR, "score_bars.json");
  const expPath = path.join(FIXTURE_DIR, "expected.json");

  if (!fs.existsSync(binPath)) {
    throw new Error(`Fixture missing: ${binPath}. Run Task 0 generate.py first.`);
  }

  const buf = fs.readFileSync(binPath);
  const audioBytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  const scoreBars = JSON.parse(fs.readFileSync(barsPath, "utf8")) as ScoreBar[];
  const expected = JSON.parse(fs.readFileSync(expPath, "utf8"));
  return { audioBytes, scoreBars, expected };
}

describe("alignChunkChroma", () => {
  it("returns BarMapChroma with bar_min within expected bounds for coldstart fixture", () => {
    const { audioBytes, scoreBars, expected } = loadFixture();
    const n_frames = audioBytes.byteLength / (12 * 4);

    const result = alignChunkChroma(audioBytes, n_frames, scoreBars, 50.0, 5.0);

    expect(result.bar_min).toBeGreaterThanOrEqual(expected.bar_min_lo);
    expect(result.bar_min).toBeLessThanOrEqual(expected.bar_min_hi);
    expect(result.bar_max).toBeGreaterThanOrEqual(expected.bar_max_lo);
    expect(result.bar_max).toBeLessThanOrEqual(expected.bar_max_hi);
    expect(result.bar_per_frame).toHaveLength(expected.decim_n);
    expect(result.cost).toBeLessThan(expected.cost_hi);
  });

  it("bar_per_frame is non-decreasing (monotone warping path)", () => {
    const { audioBytes, scoreBars } = loadFixture();
    const n_frames = audioBytes.byteLength / (12 * 4);

    const result = alignChunkChroma(audioBytes, n_frames, scoreBars, 50.0, 5.0);

    for (let i = 1; i < result.bar_per_frame.length; i++) {
      expect(result.bar_per_frame[i]).toBeGreaterThanOrEqual(
        result.bar_per_frame[i - 1]!,
      );
    }
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run vitest run --config vitest.node.config.ts src/services/wasm-bridge.test.ts 2>&1 | head -30
```

Expected: FAIL — `alignChunkChroma is not a function` (function not yet exported from wasm-bridge.ts).

- [ ] **Step 3: Add `BarMapChroma` type and `alignChunkChroma` function to `wasm-bridge.ts`**

Add after the existing `AlignChunkResult` interface (line 149) the new type:

```typescript
export interface BarMapChroma {
	bar_min: number;
	bar_max: number;
	cost: number;
	bar_per_frame: number[];
}
```

Add after the `alignChunk` function (after line 270) the new wrapper:

```typescript
/**
 * Align a 15s audio chunk to a score using chroma-based subsequence DTW.
 *
 * @param audioChromaBytes raw LE float32 bytes, row-major 12 x chromaFrames
 * @param chromaFrames number of chroma columns
 * @param scoreBars array of ScoreBar from the loaded score JSON
 * @param frameRateHz chroma frame rate (typically 50.0)
 * @param decimHz output frame rate for bar_per_frame (typically 5.0)
 */
export function alignChunkChroma(
	audioChromaBytes: Uint8Array,
	chromaFrames: number,
	scoreBars: ScoreBar[],
	frameRateHz: number,
	decimHz: number,
): BarMapChroma {
	return requireScoreAnalysis().align_chunk_chroma(
		audioChromaBytes,
		chromaFrames,
		scoreBars,
		frameRateHz,
		decimHz,
	) as BarMapChroma;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run vitest run --config vitest.node.config.ts src/services/wasm-bridge.test.ts 2>&1
```

Expected: PASS — both tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/wasm-bridge.ts apps/api/src/services/wasm-bridge.test.ts && git commit -m "feat(wasm-bridge): add BarMapChroma type and alignChunkChroma wrapper"
```

---

## Task 4: Extend `MuqResult` and `callMuqEndpoint` with Chroma Fields

**Group:** 2 (sequential, depends on Group 1)

**Behavior being verified:** `callMuqEndpoint` returns a `MuqResult` that includes `chromaBytes: Uint8Array | null` and `chromaFrames: number`. When the MuQ response contains `chroma_b64` and `chroma_frames`, they are decoded and returned. When absent, `chromaBytes` is `null`.

**Interface under test:** `callMuqEndpoint` exported from `inference.ts`; `MuqResult` type exported from `inference.ts`.

**Files:**
- Modify: `apps/api/src/services/inference.ts`

Note: This test runs in the node pool (pure function, no CF bindings). There is no existing `inference.test.ts` — create it. It belongs in the `src/services/**/*.test.ts` glob.

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/services/inference.test.ts
import { describe, expect, it } from "vitest";
import { callMuqEndpoint } from "./inference";
import type { MuqResult } from "./inference";

// Minimal Bindings stub — only MUQ_ENDPOINT is needed for these tests
function makeEnv(url: string): { MUQ_ENDPOINT: string } {
  return { MUQ_ENDPOINT: url } as unknown as Parameters<typeof callMuqEndpoint>[0];
}

const VALID_SCORES = {
  dynamics: 0.6,
  timing: 0.7,
  pedaling: 0.5,
  articulation: 0.8,
  phrasing: 0.65,
  interpretation: 0.72,
};

function makeChromaB64(nFrames: number): string {
  // 12 * nFrames float32 values, all zeros, row-major
  const buf = new ArrayBuffer(12 * nFrames * 4);
  const view = new DataView(buf);
  for (let i = 0; i < 12 * nFrames; i++) {
    view.setFloat32(i * 4, 0.5, true); // LE, value 0.5
  }
  const bytes = new Uint8Array(buf);
  return btoa(String.fromCharCode(...bytes));
}

describe("callMuqEndpoint chroma fields", () => {
  it("returns chromaBytes decoded from chroma_b64 when present in response", async () => {
    const nFrames = 10;
    const b64 = makeChromaB64(nFrames);

    const mockResponse = JSON.stringify({
      predictions: VALID_SCORES,
      chroma_b64: b64,
      chroma_frames: nFrames,
      chroma_frame_rate_hz: 50.0,
    });

    // Use a local fetch mock via MSW or direct fetch override is too complex for a node test.
    // Instead, test the parsing logic directly via a helper that accepts a raw response body.
    // We export a testable parseMusqResponse for this purpose.
    const { parseMuqResponse } = await import("./inference");
    const result = parseMuqResponse(JSON.parse(mockResponse));

    expect(result.chromaBytes).not.toBeNull();
    expect(result.chromaBytes!.byteLength).toBe(12 * nFrames * 4);
    expect(result.chromaFrames).toBe(nFrames);
    expect(result.chromaFrameRateHz).toBe(50.0);
  });

  it("returns chromaBytes=null when chroma_b64 absent in response", async () => {
    const { parseMuqResponse } = await import("./inference");
    const result = parseMuqResponse({
      predictions: VALID_SCORES,
    });
    expect(result.chromaBytes).toBeNull();
    expect(result.chromaFrames).toBe(0);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run vitest run --config vitest.node.config.ts src/services/inference.test.ts 2>&1 | head -30
```

Expected: FAIL — `parseMuqResponse is not a function` (not exported yet).

- [ ] **Step 3: Implement changes in `inference.ts`**

Extend `MuqResult` interface by replacing the existing definition at lines 24-27:

```typescript
export interface MuqResult {
	scores: MuqScores;
	confidences: MuqConfidences | null;
	chromaBytes: Uint8Array | null;
	chromaFrames: number;
	chromaFrameRateHz: number;
}
```

Extend `MuqResponseRaw` interface by replacing it:

```typescript
interface MuqResponseRaw {
	predictions: Record<string, number>;
	confidences?: Record<string, number>;
	chroma_b64?: string;
	chroma_frames?: number;
	chroma_frame_rate_hz?: number;
}
```

Add a `parseMuqResponse` function (exported, for testability) after `encodeBase64`:

```typescript
/** Parse a raw MuQ JSON response into a typed MuqResult. Exported for unit testing. */
export function parseMuqResponse(raw: MuqResponseRaw): MuqResult {
	const missingDims = MUQ_DIMS.filter(
		(dim) => typeof raw.predictions[dim] !== "number",
	);
	if (missingDims.length > 0) {
		throw new InferenceError(
			`MuQ response missing dimensions: ${missingDims.join(", ")}`,
		);
	}

	const scores: MuqScores = {
		dynamics: raw.predictions.dynamics,
		timing: raw.predictions.timing,
		pedaling: raw.predictions.pedaling,
		articulation: raw.predictions.articulation,
		phrasing: raw.predictions.phrasing,
		interpretation: raw.predictions.interpretation,
	};

	const confidences: MuqConfidences | null = raw.confidences
		? {
				dynamics: raw.confidences.dynamics ?? 1.0,
				timing: raw.confidences.timing ?? 1.0,
				pedaling: raw.confidences.pedaling ?? 1.0,
				articulation: raw.confidences.articulation ?? 1.0,
				phrasing: raw.confidences.phrasing ?? 1.0,
				interpretation: raw.confidences.interpretation ?? 1.0,
			}
		: null;

	let chromaBytes: Uint8Array | null = null;
	let chromaFrames = 0;
	let chromaFrameRateHz = 50.0;

	if (raw.chroma_b64 && raw.chroma_frames) {
		const binaryStr = atob(raw.chroma_b64);
		const bytes = new Uint8Array(binaryStr.length);
		for (let i = 0; i < binaryStr.length; i++) {
			bytes[i] = binaryStr.charCodeAt(i);
		}
		chromaBytes = bytes;
		chromaFrames = raw.chroma_frames;
		chromaFrameRateHz = raw.chroma_frame_rate_hz ?? 50.0;
	}

	return { scores, confidences, chromaBytes, chromaFrames, chromaFrameRateHz };
}
```

Update `callMuqEndpoint` to delegate to `parseMuqResponse` — replace the block from `const missingDims` through `return { scores, confidences }` with:

```typescript
	const raw = (await response.json()) as MuqResponseRaw;
	return parseMuqResponse(raw);
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run vitest run --config vitest.node.config.ts src/services/inference.test.ts 2>&1
```

Expected: PASS — both tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/inference.ts apps/api/src/services/inference.test.ts && git commit -m "feat(inference): extend MuqResult with chromaBytes/chromaFrames fields and export parseMuqResponse"
```

---

## Task 5: Drop `followerState` from `session-brain.schema.ts`

**Group:** 2 (parallel with Task 4; depends on Group 1)

**Behavior being verified:** `sessionStateSchema` no longer contains `followerState`; `createInitialState` no longer initializes `followerState`; `sessionStateSchema.parse({})` with a state object containing a stale `followerState` key does not throw (Zod's `.strip()` behavior drops unknown keys by default).

**Interface under test:** `sessionStateSchema` and `createInitialState` imported from `session-brain.schema.ts`.

**Files:**
- Modify: `apps/api/src/do/session-brain.schema.ts`

Note: Test runs in the node pool. No existing schema test file — create `session-brain.schema.test.ts` in the same directory. Add `src/do/**/*.test.ts` to the `include` list of `vitest.node.config.ts`.

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/do/session-brain.schema.test.ts
import { describe, expect, it } from "vitest";
import { sessionStateSchema, createInitialState } from "./session-brain.schema";

describe("sessionStateSchema followerState removal", () => {
  it("parsed state does not include followerState property", () => {
    const initial = createInitialState("sess-1", "student-1", null);
    expect("followerState" in initial).toBe(false);
  });

  it("schema strips stale followerState key from legacy persisted state", () => {
    const legacyRaw = {
      version: 0,
      sessionId: "sess-1",
      studentId: "student-1",
      conversationId: null,
      followerState: { lastKnownBar: 5 }, // stale key from old schema
    };
    const parsed = sessionStateSchema.parse(legacyRaw);
    expect("followerState" in parsed).toBe(false);
  });

  it("createInitialState produces a valid schema parse without followerState", () => {
    const state = createInitialState("sess-2", "student-2", "conv-1");
    expect(() => sessionStateSchema.parse(state)).not.toThrow();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run vitest run --config vitest.node.config.ts src/do/session-brain.schema.test.ts 2>&1 | head -30
```

Expected: FAIL — `expect("followerState" in initial).toBe(false)` fails because `createInitialState` still returns `followerState`.

Note: To make the `src/do/**/*.test.ts` glob work, first add it to `vitest.node.config.ts`'s `include` array:

```typescript
include: [
  "scripts/**/*.test.ts",
  "src/harness/skills/__catalog__/**/*.test.ts",
  "src/harness/skills/validator.test.ts",
  "src/lib/**/*.test.ts",
  "src/harness/loop/**/*.test.ts",
  "src/services/**/*.test.ts",
  "src/do/**/*.test.ts",  // add this line
],
```

- [ ] **Step 3: Remove `followerState` from schema and `createInitialState`**

In `session-brain.schema.ts`:

Remove the `followerState` field from `sessionStateSchema` (lines 33-37):
```typescript
	followerState: z
		.object({
			lastKnownBar: z.number().int().nullable(),
		})
		.default({ lastKnownBar: null }),
```
(Delete these five lines entirely.)

Remove `followerState` from `createInitialState` return object (line 139):
```typescript
		followerState: { lastKnownBar: null },
```
(Delete this line entirely.)

The `SessionState` type is derived via `z.infer`, so it will automatically exclude `followerState`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run vitest run --config vitest.node.config.ts src/do/session-brain.schema.test.ts 2>&1
```

Expected: PASS — all three tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.schema.ts apps/api/src/do/session-brain.schema.test.ts apps/api/vitest.node.config.ts && git commit -m "feat(session-brain): drop followerState from schema and createInitialState"
```

---

## Task 6: Wire `alignChunkChroma` into `session-brain.ts`

**Group:** 3 (sequential, depends on Group 2)

**Behavior being verified:** Both `process_muq_result` code paths (lines ~598 and ~1073) that previously called `wasm.alignChunk(...)` now call `wasm.alignChunkChroma(...)` using `chromaBytes` from `MuqResult`; stale `followerState` reads are removed; the DO no longer imports `FollowerState` from wasm-bridge.

**Interface under test:** The internal DO processing logic — tested via the unit test in Task 7 through the exported pure functions that are observable from outside the DO.

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`

This task has no new test file (Task 7 covers the unit tests). This is an implementation-only task that changes two call sites.

- [ ] **Step 1: No failing test to write for this task** — implementation changes only. Verify compile passes instead:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run tsc --noEmit 2>&1 | head -30
```

Expected at this point: TypeScript errors on `followerState` references and `FollowerState` import in `session-brain.ts` (because schema no longer has `followerState`). This confirms the old wiring is broken and needs replacement.

- [ ] **Step 2: Implement the replacements in `session-brain.ts`**

**2a. Remove `FollowerState` from the import** (line 34):

Remove `FollowerState,` from the destructured import of `../services/wasm-bridge`. Keep `NgramIndex`, `NoteAlignment`, `PerfNote`, `PerfPedalEvent`, `RerankFeatures`, `ScoreContext`, `ScoredChunk`, `StudentBaselines`, and add `BarMapChroma`:

```typescript
import type {
	BarMapChroma,
	NgramIndex,
	NoteAlignment,
	PerfNote,
	PerfPedalEvent,
	RerankFeatures,
	ScoreContext,
	ScoredChunk,
	StudentBaselines,
} from "../services/wasm-bridge";
```

**2b. Replace first `alignChunk` call site** (around lines 596-613). The existing block:

```typescript
				const followerState: FollowerState = {
					last_known_bar: currentState.followerState.lastKnownBar,
				};

				const alignResult = wasm.alignChunk(
					index,
					perfNotes,
					scoreCtx.score.bars,
					followerState,
				);

				currentState.followerState.lastKnownBar =
					alignResult.state.last_known_bar;

				if (alignResult.bar_map !== null) {
					barMapAlignments = alignResult.bar_map.alignments;
					const analysis = wasm.analyzeTier1(
						alignResult.bar_map,
```

Replace with:

```typescript
				const chromaResult: BarMapChroma | null =
					muqResult.chromaBytes !== null
						? wasm.alignChunkChroma(
								muqResult.chromaBytes,
								muqResult.chromaFrames,
								scoreCtx.score.bars,
								muqResult.chromaFrameRateHz,
								5.0,
							)
						: null;

				if (chromaResult !== null) {
					chunkBarRange = [chromaResult.bar_min, chromaResult.bar_max];
					// bar_per_frame forwarded to WebSocket client below
					chunkAnalysisTier = 1;
				}

				if (chromaResult !== null) {
					// Tier 1 analysis still uses note-level data when AMT is available
					if (alignResult_PLACEHOLDER_barMap !== undefined) {
```

Wait — the `analyzeTier1` call requires a `BarMap` (note-level), which the old `alignChunk` produced. The spec says per-note alignment is deferred. The correct replacement: when `chromaResult` is available, skip `analyzeTier1` and go directly to `analyzeTier2` (since AMT is not deployed). The full replacement for the first call site block (lines 596-651) is:

```typescript
				// Chroma DTW alignment (replaces note-level alignChunk)
				const chromaResult: BarMapChroma | null =
					muqResult.chromaBytes !== null
						? (() => {
								try {
									return wasm.alignChunkChroma(
										muqResult.chromaBytes!,
										muqResult.chromaFrames,
										scoreCtx.score.bars,
										muqResult.chromaFrameRateHz,
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

				// Tier 2 analysis (note-level; Tier 1 expression analysis deferred until AMT redeploy)
				const analysis2 = wasm.analyzeTier2(perfNotes, perfPedal, scoresArray);
				if (chunkAnalysisTier === 3) {
					chunkAnalysisTier = analysis2.tier;
					const barStr = analysis2.bar_range;
					if (barStr !== null) {
						const parts = barStr.split("-").map(Number);
						if (
							parts.length === 2 &&
							parts[0] !== undefined &&
							parts[1] !== undefined
						) {
							chunkBarRange = [parts[0], parts[1]];
						}
					}
				}
```

**2c. Replace second `alignChunk` call site** (eval_chunk handler, around lines 1073-1113) with the same pattern — use `muqResult.chromaBytes` if available (note: in the eval path the muq result is synthesized from `wsEvalChunk.predictions`, so `chromaBytes` will be `null`; the code should safely fall through to `analyzeTier2`):

```typescript
				// Chroma DTW alignment — null in eval path (no real audio)
				const chromaResult: BarMapChroma | null = null;

				if (chromaResult !== null) {
					chunkBarRange = [chromaResult.bar_min, chromaResult.bar_max];
					chunkAnalysisTier = 1;
				}

				const analysis2 = wasm.analyzeTier2(perfNotes, perfPedal, scoresArray);
				if (chunkAnalysisTier === 3) {
					chunkAnalysisTier = analysis2.tier;
					const barStr = analysis2.bar_range;
					if (barStr !== null) {
						const parts = barStr.split("-").map(Number);
						if (
							parts.length === 2 &&
							parts[0] !== undefined &&
							parts[1] !== undefined
						) {
							chunkBarRange = [parts[0], parts[1]];
						}
					}
				}
```

**2d. Remove the `followerState` reset at line 1320** (inside `webSocketClose` or session reset):

Find and delete: `state.followerState = { lastKnownBar: null };`

**2e. Forward `bar_per_frame` to WebSocket client.** Find the WebSocket send for `chunk_processed` (around line 700-750) and add `bar_per_frame` to the payload:

```typescript
			this.sendWs(ws, {
				type: "chunk_processed",
				index,
				scores: muqScores,
				bar_per_frame: chromaResult?.bar_per_frame ?? null,
			});
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run tsc --noEmit 2>&1
```

Expected: zero errors. If errors remain on `muqResult` being undefined in the first call site, note that `muqResult` is defined in the chunk handler scope and already carries `chromaBytes` after Task 4.

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/do/session-brain.ts && git commit -m "feat(session-brain): replace alignChunk with alignChunkChroma; forward bar_per_frame to WebSocket"
```

---

## Task 7: Unit Tests for Chroma Path in `session-brain.unit.test.ts`

**Group:** 3 (parallel with Task 6; depends on Group 2)

**Behavior being verified:** Three behaviors through `buildV6WsPayload` and the exported functions in `session-brain.ts`: (1) happy path with valid chroma bytes produces a `bar_per_frame` field in the chunk_processed message; (2) null chroma bytes (absent from MuQ response) leaves `bar_per_frame` null; (3) mismatched chroma byte length throws an error that the DO catches and logs without crashing.

**Interface under test:** `buildV6WsPayload` (already tested) and `parseMuqResponse` (from inference.ts) — both are pure exported functions. The third test exercises the `parseMuqResponse` error path.

**Files:**
- Modify: `apps/api/src/do/session-brain.unit.test.ts`

Note: The existing test file already covers `buildV6WsPayload`. Add three new describe blocks for the chroma path behaviors.

- [ ] **Step 1: Write the failing tests** (append to existing `session-brain.unit.test.ts` after line 60):

```typescript
import { parseMuqResponse } from "../services/inference";

describe("parseMuqResponse chroma extraction", () => {
	it("returns chromaBytes=null when response has no chroma_b64", () => {
		const raw = {
			predictions: {
				dynamics: 0.6,
				timing: 0.7,
				pedaling: 0.5,
				articulation: 0.8,
				phrasing: 0.65,
				interpretation: 0.72,
			},
		};
		const result = parseMuqResponse(raw);
		expect(result.chromaBytes).toBeNull();
		expect(result.chromaFrames).toBe(0);
	});

	it("returns decoded chromaBytes when chroma_b64 present", () => {
		const nFrames = 5;
		const buf = new Uint8Array(12 * nFrames * 4).fill(127);
		const b64 = btoa(String.fromCharCode(...buf));
		const raw = {
			predictions: {
				dynamics: 0.6,
				timing: 0.7,
				pedaling: 0.5,
				articulation: 0.8,
				phrasing: 0.65,
				interpretation: 0.72,
			},
			chroma_b64: b64,
			chroma_frames: nFrames,
			chroma_frame_rate_hz: 50.0,
		};
		const result = parseMuqResponse(raw);
		expect(result.chromaBytes).not.toBeNull();
		expect(result.chromaBytes!.byteLength).toBe(12 * nFrames * 4);
		expect(result.chromaFrames).toBe(nFrames);
		expect(result.chromaFrameRateHz).toBe(50.0);
	});

	it("throws InferenceError when MuQ response missing a dimension", () => {
		const raw = {
			predictions: {
				dynamics: 0.6,
				// timing missing
				pedaling: 0.5,
				articulation: 0.8,
				phrasing: 0.65,
				interpretation: 0.72,
			},
		};
		expect(() => parseMuqResponse(raw as Parameters<typeof parseMuqResponse>[0])).toThrow(
			"MuQ response missing dimensions: timing",
		);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run vitest run --config vitest.node.config.ts src/do/session-brain.unit.test.ts 2>&1 | head -30
```

Expected: FAIL — `parseMuqResponse is not a function` or import error because `parseMuqResponse` isn't imported in `session-brain.unit.test.ts` yet.

- [ ] **Step 3: The implementation is already done in Task 4** (`parseMuqResponse` exported from `inference.ts`). Only the test import is needed in `session-brain.unit.test.ts`.

Add to the top of `session-brain.unit.test.ts`:

```typescript
import { parseMuqResponse } from "../services/inference";
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run vitest run --config vitest.node.config.ts src/do/session-brain.unit.test.ts 2>&1
```

Expected: PASS — all existing tests plus three new `parseMuqResponse` tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.unit.test.ts && git commit -m "test(session-brain): add chroma path unit tests via parseMuqResponse"
```

---

## Task 8: Delete `score_follower.rs` and Clean `lib.rs` References

**Group:** 4 (sequential, depends on Group 3)

**Behavior being verified:** After deletion, `cargo build` succeeds and all existing cargo tests still pass; `bun run tsc --noEmit` succeeds; `wasm.alignChunk` is no longer callable (removed from bridge).

**Interface under test:** Build system — no new application test needed; the existing cargo tests and TS type check are the verification.

**Files:**
- Delete: `apps/api/src/wasm/score-analysis/src/score_follower.rs`
- Modify: `apps/api/src/wasm/score-analysis/src/lib.rs`
- Modify: `apps/api/src/wasm/score-analysis/src/types.rs`
- Modify: `apps/api/src/services/wasm-bridge.ts`

- [ ] **Step 1: No new test — verify the starting state compiles**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo build 2>&1 | head -20
```

Expected: build passes (or shows only warnings, not errors) at this point.

- [ ] **Step 2: Delete and clean up**

**2a. Delete `score_follower.rs`:**

```bash
rm /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis/src/score_follower.rs
```

**2b. Remove `mod score_follower;` from `lib.rs`** — delete line 8:

```
mod score_follower;
```

**2c. Remove `align_chunk` wasm_bindgen function from `lib.rs`** — delete the entire `align_chunk` function block (lines 38-63 in the original):

```rust
#[wasm_bindgen]
pub fn align_chunk(
    chunk_index: usize,
    perf_notes_js: JsValue,
    score_bars_js: JsValue,
    follower_state_js: JsValue,
) -> Result<JsValue, JsValue> {
    let perf_notes: Vec<types::PerfNote> = serde_wasm_bindgen::from_value(perf_notes_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let score_bars: Vec<types::ScoreBar> = serde_wasm_bindgen::from_value(score_bars_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut state: types::FollowerState = serde_wasm_bindgen::from_value(follower_state_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let bar_map = score_follower::align_chunk(chunk_index, &perf_notes, &score_bars, &mut state);

    let result = types::AlignChunkResult { bar_map, state };
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}
```

**2d. Remove `FollowerState`, `AlignChunkResult`, `BarMap`, `NoteAlignment` from `types.rs`** (lines 100-203 in the original). These are only used by the deleted `score_follower.rs` and the deleted `align_chunk` function. Remove the entire block from `// --- Score follower types ---` through `pub struct AlignChunkResult { ... }`.

**2e. Remove `alignChunk`, `FollowerState`, `AlignChunkResult`, `BarMap`, `NoteAlignment`, `PerfNote`, `PerfPedalEvent` from `wasm-bridge.ts`** — specifically remove:
- `interface PerfNote` (lines 44-50) — Note: `PerfNote` is also defined and used in `inference.ts`; wasm-bridge's copy can be removed since it was only needed for `alignChunk`. Verify that nothing else imports `PerfNote` from `wasm-bridge.ts`; if so, keep it. Check first:

```bash
grep -rn "from.*wasm-bridge.*PerfNote\|import.*PerfNote.*wasm-bridge" /Users/jdhiman/Documents/crescendai/apps/api/src/ 2>/dev/null
```

- `interface FollowerState` (lines 142-144)
- `interface AlignChunkResult` (lines 146-149)
- `interface NoteAlignment` (lines 123-131)
- `function alignChunk(...)` (lines 258-270)

**2f. Remove `NoteAlignment` import from `session-brain.ts`** if it was only used in the old alignment path. Verify:

```bash
grep -n "NoteAlignment" /Users/jdhiman/Documents/crescendai/apps/api/src/do/session-brain.ts
```

If unused after Task 6 changes, remove it from the import.

- [ ] **Step 3: Verify build and tests pass**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test 2>&1 | tail -20
```

Expected: all tests pass (chroma_dtw_roundtrip + any surviving tests from other modules).

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run tsc --noEmit 2>&1
```

Expected: zero TypeScript errors.

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run vitest run --config vitest.node.config.ts 2>&1 | tail -20
```

Expected: all node-pool tests pass.

- [ ] **Step 4: Commit**

```bash
git add -A apps/api/src/wasm/score-analysis/src/ apps/api/src/services/wasm-bridge.ts apps/api/src/do/session-brain.ts && git commit -m "refactor(score-follower): delete score_follower.rs and remove alignChunk from WASM bridge"
```

---

## Task 9: Wire `chroma_b64` into MuQ Python Handler

**Group:** 4 (parallel with Task 8; depends on Group 3)

**Behavior being verified:** `handler.py`'s inference response dict includes `chroma_b64`, `chroma_frames`, and `chroma_frame_rate_hz` keys when audio decoding succeeds, using the `chroma_feature` helper from `chroma.py`.

**Interface under test:** The `chroma_feature` function (already tested in Task 2); the handler integration is verified by inspecting that the result dict contains the expected keys after calling the handler's `predict` path.

**Files:**
- Modify: `apps/inference/muq/handler.py`

Note: The handler integration test is a simple import+call test using the existing pytest infrastructure. Add to `test_chroma.py` rather than creating a new file.

- [ ] **Step 1: Write the failing test** (append to `test_chroma.py`):

```python
class TestHandlerChromaIntegration:
    """Verify handler response includes chroma fields."""

    def test_chroma_fields_present_in_result_dict(self):
        """The result dict built by handler.py contains chroma_b64, chroma_frames."""
        import base64
        import struct

        # Simulate what handler.py does: call chroma_feature and embed in result
        y = make_sine(440.0, 2.0)
        raw_bytes, n_frames = chroma_feature(y, SR)

        result = {
            "predictions": {"dynamics": 0.6},
            "chroma_b64": base64.b64encode(raw_bytes).decode("ascii"),
            "chroma_frames": n_frames,
            "chroma_frame_rate_hz": float(SR / HOP),
        }

        assert "chroma_b64" in result
        assert "chroma_frames" in result
        assert result["chroma_frames"] == n_frames
        # Round-trip decode check
        decoded = base64.b64decode(result["chroma_b64"])
        floats = struct.unpack(f"<{12 * n_frames}f", decoded)
        assert len(floats) == 12 * n_frames
```

- [ ] **Step 2: Run test — verify it FAILS**

The test above actually passes immediately since it only exercises `chroma_feature` directly. The real test for handler.py wiring is the TypeScript test in Task 4 (which calls `parseMuqResponse` against a response with `chroma_b64`). The handler's Python side test is a compile/import check:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/inference/muq && python -c "from handler import InferenceHandler" 2>&1
```

Expected at the start of Task 9: ImportError or AttributeError if `chroma_feature` is referenced in handler.py before it exists (it does exist after Task 2, so this should pass — the real failure is that handler.py doesn't yet call `chroma_feature`).

To create an observable failing test, verify handler.py does NOT yet have `chroma_b64` in its result dict by grepping:

```bash
grep -n "chroma_b64" /Users/jdhiman/Documents/crescendai/apps/inference/muq/handler.py
```

Expected: no matches (i.e., the key is absent from handler.py).

- [ ] **Step 3: Add chroma fields to `handler.py` result dict**

In `handler.py`, find the `predict` method's result dict construction (around lines 144-174). Add the `chroma_feature` import at the top of the file:

```python
from chroma import chroma_feature
```

Inside the inference `predict` method, after the audio is decoded and before building `result`, add chroma computation. Find the line `processing_time_ms = int((time.time() - start_time) * 1000)` and add above the result dict:

```python
            # Compute chroma for score following
            import base64
            chroma_bytes, chroma_frames = chroma_feature(audio, sr=22050)
            chroma_b64 = base64.b64encode(chroma_bytes).decode("ascii")
            chroma_frame_rate_hz = 22050 / 441  # ~50.0
```

Then in both result dict branches (gaussian head and scalar head), add:

```python
                    "chroma_b64": chroma_b64,
                    "chroma_frames": chroma_frames,
                    "chroma_frame_rate_hz": chroma_frame_rate_hz,
```

Note: The `audio` variable in handler.py holds the decoded waveform array. Verify the variable name by checking `handler.py` lines 100-135 before editing. The sample rate used by the handler is 22050 Hz (librosa default).

- [ ] **Step 4: Verify handler imports correctly**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/inference/muq && python -c "from handler import InferenceHandler; print('OK')" 2>&1
```

Expected: `OK`

```bash
cd /Users/jdhiman/Documents/crescendai/apps/inference/muq && uv run pytest test_chroma.py -v 2>&1 | tail -20
```

Expected: all tests pass (including the new integration test).

- [ ] **Step 5: Commit**

```bash
git add apps/inference/muq/handler.py apps/inference/muq/test_chroma.py && git commit -m "feat(muq): add chroma_b64/chroma_frames to MuQ handler response"
```

---

## Summary

| Task | Group | Description | New Files |
|------|-------|-------------|-----------|
| 0 | 0 | Fixture generator + committed binary fixtures | `generate.py`, 6 fixture files |
| 1 | 1 | Rust `align_chunk_chroma` + `BarMapChroma` | `chroma_dtw.rs`, `chroma_dtw_roundtrip.rs` |
| 2 | 1 | Python `chroma_feature` + pytest | `chroma.py`, `test_chroma.py` |
| 3 | 1 | TS `alignChunkChroma` + `BarMapChroma` in wasm-bridge | `wasm-bridge.test.ts` |
| 4 | 2 | Extend `MuqResult` with chroma fields | `inference.test.ts` |
| 5 | 2 | Drop `followerState` from schema | `session-brain.schema.test.ts` |
| 6 | 3 | Wire `alignChunkChroma` into DO chunk handlers | — |
| 7 | 3 | Unit tests for chroma path | (modify existing) |
| 8 | 4 | Delete `score_follower.rs`; remove dead types | — |
| 9 | 4 | Add chroma fields to MuQ Python handler | — |

**9 implementation tasks across 4 dependency groups.** Group 1 tasks are fully parallel. Groups 2-4 are sequential. The critical path is: Task 0 → Task 1 (longest: Rust DTW implementation) → Task 6 → Task 8.
