# Score Following + Bar-Aligned Analysis Engine Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the subagent's input from raw dimension scores ("pedaling: 0.35") to bar-aligned musical facts ("pedal held 3.2 beats at bars 20-24, harmony changes every 2 beats") by adding score following (Phase 1c), a musical analysis engine (Phase 1d), and a reference performance cache (Phase 1e) to the PracticeSession Durable Object.

**Architecture:** The HF inference endpoint already returns both MuQ scores and AMT MIDI notes. This plan adds three layers inside the DO: (1) fuzzy piece matching + score loading, (2) onset+pitch DTW to map performance MIDI to score bar numbers, (3) per-dimension musical analysis comparing performance against score and reference profiles. Three tiers of feedback quality: Tier 1 (piece known, full bar-aligned facts), Tier 2 (piece unknown, absolute MIDI analysis), Tier 3 (AMT failed, scores only -- current behavior). Also extracts pedal CC64 events from the AMT transcription handler.

**Tech Stack:** Rust (Cloudflare Workers / Durable Objects), Python (HF inference handler, offline reference cache generation), D1 (SQLite), R2 (score JSON + reference profiles)

---

## File Structure

### New Files (Rust -- API Worker)

| File | Responsibility |
|------|---------------|
| `apps/api/src/practice/score_context.rs` | Session-level score context: fuzzy matching, R2 fetch, ScoreData deserialization, caching for session lifetime |
| `apps/api/src/practice/score_follower.rs` | Subsequence DTW on onset+pitch pairs, cross-chunk bar map continuity, re-anchoring on skip/restart |
| `apps/api/src/practice/analysis.rs` | Per-dimension bar-aligned musical fact generation (all 6 dims), reference comparison, Tier 1/2/3 output |
| `apps/api/src/practice/piece_match.rs` | Fuzzy piece matching against D1 catalog (composer filter + trigram title similarity) |

### New Files (Python -- Offline Tooling)

| File | Responsibility |
|------|---------------|
| `model/src/score_library/reference_cache.py` | Offline job: run A1-Max + AMT on MAESTRO recordings, align to score, compute per-bar stats, upload to R2 |

### Modified Files

| File | Changes |
|------|---------|
| `apps/inference/models/transcription.py` | Extract pedal CC64 events from ByteDance MIDI output |
| `apps/inference/handler.py` | Include `pedal_events` in response alongside `midi_notes` |
| `apps/api/src/practice/session.rs` | Parse full HF response (midi_notes + pedal_events), call score context/follower/analysis, pass enriched piece_context to handle_ask_inner |
| `apps/api/src/practice/mod.rs` | Register new modules |
| `apps/api/src/services/prompts.rs` | Expand `<piece_context>` block with `<musical_analysis>` per-dimension facts |
| `apps/api/src/services/teaching_moments.rs` | Accept optional bar_range in TeachingMoment struct |
| `apps/api/migrations/0005_piece_requests.sql` | New table for tracking unmatched piece queries |

### Reference Data

| Location | Content |
|----------|---------|
| R2: `scores/v1/{piece_id}.json` | Already deployed (242 ASAP pieces) |
| R2: `references/v1/{piece_id}.json` | New: per-bar stats from MAESTRO professional recordings |

---

## Chunk 1: AMT Pedal Extraction + HF Response Enrichment

### Task 1: Extract Pedal Events from ByteDance AMT Output

The ByteDance `PianoTranscription` model jointly transcribes note onsets and pedal events. The current handler only extracts `instrument.notes` but ignores `instrument.control_changes` where CC64 (sustain pedal) events live. `pretty_midi` stores these as objects with `number`, `value` (0-127, >64 = pedal on), and `time` (seconds).

**Files:**
- Modify: `apps/inference/models/transcription.py:54-102`
- Modify: `apps/inference/handler.py:141-183`

- [ ] **Step 1: Write test for pedal extraction**

Create `apps/inference/tests/test_transcription.py`:

```python
"""Tests for AMT transcription including pedal event extraction."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from models.transcription import TranscriptionModel, TranscriptionError


class FakeNote:
    def __init__(self, pitch, start, end, velocity):
        self.pitch = pitch
        self.start = start
        self.end = end
        self.velocity = velocity


class FakeCC:
    def __init__(self, number, value, time):
        self.number = number
        self.value = value
        self.time = time


class FakeInstrument:
    def __init__(self, notes, control_changes=None):
        self.notes = notes
        self.control_changes = control_changes or []


class FakeMIDI:
    def __init__(self, instruments):
        self.instruments = instruments


def test_transcribe_returns_pedal_events():
    """Pedal CC64 events should be extracted alongside notes."""
    instrument = FakeInstrument(
        notes=[FakeNote(60, 0.1, 0.5, 80)],
        control_changes=[
            FakeCC(64, 127, 0.05),   # pedal on
            FakeCC(64, 0, 0.45),     # pedal off
            FakeCC(67, 100, 0.2),    # NOT pedal (CC67 = soft pedal)
        ],
    )
    midi = FakeMIDI(instruments=[instrument])

    with patch("models.transcription.PianoTranscription") as mock_pt:
        mock_pt.return_value.transcribe = MagicMock()
        model = TranscriptionModel.__new__(TranscriptionModel)
        model._transcriber = mock_pt.return_value

        with patch("models.transcription.pretty_midi.PrettyMIDI", return_value=midi):
            with patch("pathlib.Path.exists", return_value=True):
                notes, pedal_events = model.transcribe(np.zeros(24000), 24000)

    assert len(notes) == 1
    assert len(pedal_events) == 2  # Only CC64, not CC67
    assert pedal_events[0] == {"time": 0.05, "value": 127}
    assert pedal_events[1] == {"time": 0.45, "value": 0}


def test_transcribe_empty_pedal():
    """No pedal events should return empty list."""
    instrument = FakeInstrument(
        notes=[FakeNote(60, 0.1, 0.5, 80)],
        control_changes=[],
    )
    midi = FakeMIDI(instruments=[instrument])

    with patch("models.transcription.PianoTranscription") as mock_pt:
        mock_pt.return_value.transcribe = MagicMock()
        model = TranscriptionModel.__new__(TranscriptionModel)
        model._transcriber = mock_pt.return_value

        with patch("models.transcription.pretty_midi.PrettyMIDI", return_value=midi):
            with patch("pathlib.Path.exists", return_value=True):
                notes, pedal_events = model.transcribe(np.zeros(24000), 24000)

    assert len(pedal_events) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/inference && python -m pytest tests/test_transcription.py -v`
Expected: FAIL -- `transcribe()` returns a single list, not a tuple.

- [ ] **Step 3: Update transcription model to extract pedal events**

Modify `apps/inference/models/transcription.py`. Change `transcribe()` return type from `list[dict]` to `tuple[list[dict], list[dict]]` (notes, pedal_events):

```python
def transcribe(self, audio: np.ndarray, sample_rate: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Transcribe audio to notes and pedal events.

    Returns:
        Tuple of (notes, pedal_events):
        - notes: [{"pitch": 60, "onset": 0.12, "offset": 0.45, "velocity": 78}, ...]
        - pedal_events: [{"time": 0.05, "value": 127}, ...]  (CC64 only)
    """
    transcribe_start = time.time()
    temp_dir = tempfile.mkdtemp(prefix="amt_")

    try:
        midi_path = Path(temp_dir) / "transcription.mid"

        print("Running ByteDance AMT transcription...")
        self._transcriber.transcribe(audio, str(midi_path))

        if not midi_path.exists():
            raise TranscriptionError("Transcriber did not produce a MIDI file")

        midi = pretty_midi.PrettyMIDI(str(midi_path))

        notes = []
        pedal_events = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                notes.append({
                    "pitch": int(note.pitch),
                    "onset": round(float(note.start), 4),
                    "offset": round(float(note.end), 4),
                    "velocity": int(note.velocity),
                })
            for cc in instrument.control_changes:
                if cc.number == 64:  # Sustain pedal only
                    pedal_events.append({
                        "time": round(float(cc.time), 4),
                        "value": int(cc.value),
                    })

        notes.sort(key=lambda n: (n["onset"], n["pitch"]))
        pedal_events.sort(key=lambda p: p["time"])

        elapsed_ms = int((time.time() - transcribe_start) * 1000)
        print(f"AMT complete: {len(notes)} notes, {len(pedal_events)} pedal events in {elapsed_ms}ms")

        return notes, pedal_events

    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/inference && python -m pytest tests/test_transcription.py -v`
Expected: PASS

- [ ] **Step 5: Update handler.py to include pedal_events in response**

Modify `apps/inference/handler.py:141-183`. Update the AMT section:

```python
# Run AMT transcription (after MuQ scoring, sequential)
midi_notes = None
pedal_events = None
transcription_info = None
amt_error = None

try:
    print("Running AMT transcription...")
    amt_start = time.time()
    midi_notes, pedal_events = self._transcription.transcribe(audio, 24000)
    amt_elapsed_ms = int((time.time() - amt_start) * 1000)

    pitches = [n["pitch"] for n in midi_notes]
    transcription_info = {
        "note_count": len(midi_notes),
        "pedal_event_count": len(pedal_events),
        "pitch_range": [min(pitches), max(pitches)] if pitches else [0, 0],
        "transcription_time_ms": amt_elapsed_ms,
    }
except TranscriptionError as e:
    print(f"AMT failed (graceful degradation): {e}")
    amt_error = str(e)
```

And in the result dict (around line 165), add:

```python
result = {
    "predictions": self._predictions_to_dict(predictions),
    "midi_notes": midi_notes,
    "pedal_events": pedal_events,           # NEW
    "transcription_info": transcription_info,
    # ... rest unchanged
}
```

- [ ] **Step 6: Run full handler test suite**

Run: `cd apps/inference && python -m pytest tests/ -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add apps/inference/models/transcription.py apps/inference/handler.py apps/inference/tests/test_transcription.py
git commit -m "feat: extract pedal CC64 events from AMT transcription output"
```

---

## Chunk 2: Fuzzy Piece Matching + Demand Tracking

### Task 2: Piece Demand Tracking Table

**Files:**
- Create: `apps/api/migrations/0005_piece_requests.sql`

- [ ] **Step 1: Write migration**

```sql
-- Track piece identification requests for catalog prioritization.
-- Unmatched queries (matched_piece_id IS NULL) form a scoreboard
-- of student demand that guides which scores to source next.

CREATE TABLE IF NOT EXISTS piece_requests (
  id TEXT PRIMARY KEY,
  query TEXT NOT NULL,
  student_id TEXT NOT NULL,
  matched_piece_id TEXT,
  match_confidence REAL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_piece_requests_unmatched
  ON piece_requests(matched_piece_id) WHERE matched_piece_id IS NULL;
```

- [ ] **Step 2: Commit**

```bash
git add apps/api/migrations/0005_piece_requests.sql
git commit -m "feat: add piece_requests table for catalog demand tracking"
```

### Task 3: Fuzzy Piece Matching Module

Matches free-text student input ("chopin ballade 1") against the 242-piece D1 catalog. Two-stage: composer filter (substring) then title similarity (bigram Dice coefficient). Dice is simpler than Levenshtein for partial title matches and works well for short strings.

**Files:**
- Create: `apps/api/src/practice/piece_match.rs`
- Modify: `apps/api/src/practice/mod.rs`

- [ ] **Step 1: Write failing tests for piece matching**

Create `apps/api/src/practice/piece_match.rs` with tests at the bottom:

```rust
/// Fuzzy piece matching against the score catalog.
///
/// Two-stage matching:
/// 1. Composer filter: case-insensitive substring match
/// 2. Title similarity: bigram Dice coefficient on normalized strings
///
/// Returns the best match above a confidence threshold, or None.

/// Represents a piece from the D1 catalog for matching purposes.
#[derive(Debug, Clone)]
pub struct CatalogPiece {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
}

/// Result of fuzzy matching a query against the catalog.
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
    pub confidence: f64,
}

const MATCH_THRESHOLD: f64 = 0.3;

/// Normalize a string for matching: lowercase, remove punctuation, collapse whitespace.
fn normalize(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Extract bigrams from a string for Dice coefficient.
fn bigrams(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 2 {
        return vec![s.to_string()];
    }
    chars.windows(2).map(|w| w.iter().collect()).collect()
}

/// Dice coefficient between two strings (2 * |intersection| / (|A| + |B|)).
fn dice_similarity(a: &str, b: &str) -> f64 {
    let a_norm = normalize(a);
    let b_norm = normalize(b);
    let a_bi = bigrams(&a_norm);
    let b_bi = bigrams(&b_norm);

    if a_bi.is_empty() && b_bi.is_empty() {
        return 1.0;
    }
    if a_bi.is_empty() || b_bi.is_empty() {
        return 0.0;
    }

    let mut matches = 0;
    let mut b_used = vec![false; b_bi.len()];
    for a_bg in &a_bi {
        for (j, b_bg) in b_bi.iter().enumerate() {
            if !b_used[j] && a_bg == b_bg {
                matches += 1;
                b_used[j] = true;
                break;
            }
        }
    }

    (2.0 * matches as f64) / (a_bi.len() + b_bi.len()) as f64
}

/// Check if query contains a composer name (case-insensitive substring).
fn extract_composer<'a>(query: &str, catalog: &'a [CatalogPiece]) -> Option<&'a str> {
    let q = query.to_lowercase();
    // Collect unique composers
    let mut composers: Vec<&str> = catalog.iter().map(|p| p.composer.as_str()).collect();
    composers.sort();
    composers.dedup();

    for composer in composers {
        if q.contains(&composer.to_lowercase()) {
            return Some(composer);
        }
    }
    None
}

/// Remove the composer name from the query to get the title part.
fn strip_composer(query: &str, composer: &str) -> String {
    let q = query.to_lowercase();
    let c = composer.to_lowercase();
    normalize(&q.replace(&c, ""))
}

/// Match a free-text query against the catalog. Returns best match or None.
pub fn match_piece(query: &str, catalog: &[CatalogPiece]) -> Option<MatchResult> {
    if query.trim().is_empty() || catalog.is_empty() {
        return None;
    }

    let composer = extract_composer(query, catalog);

    // Filter by composer if detected, otherwise search all
    let candidates: Vec<&CatalogPiece> = match composer {
        Some(c) => catalog.iter().filter(|p| p.composer == c).collect(),
        None => catalog.iter().collect(),
    };

    if candidates.is_empty() {
        return None;
    }

    // Title part: remove composer from query
    let title_query = match composer {
        Some(c) => strip_composer(query, c),
        None => normalize(query),
    };

    // Score each candidate
    let mut best: Option<(f64, &CatalogPiece)> = None;
    for piece in &candidates {
        let score = dice_similarity(&title_query, &piece.title);
        if let Some((best_score, _)) = &best {
            if score > *best_score {
                best = Some((score, piece));
            }
        } else {
            best = Some((score, piece));
        }
    }

    match best {
        Some((score, piece)) if score >= MATCH_THRESHOLD => Some(MatchResult {
            piece_id: piece.piece_id.clone(),
            composer: piece.composer.clone(),
            title: piece.title.clone(),
            confidence: score,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uses real piece_ids and titles from the deployed ASAP catalog.
    fn sample_catalog() -> Vec<CatalogPiece> {
        vec![
            CatalogPiece {
                piece_id: "chopin.ballades.1".into(),
                composer: "Chopin".into(),
                title: "Ballade No. 1".into(),
            },
            CatalogPiece {
                piece_id: "chopin.ballades.4".into(),
                composer: "Chopin".into(),
                title: "Ballade No. 4".into(),
            },
            CatalogPiece {
                piece_id: "bach.prelude.bwv_846".into(),
                composer: "Bach".into(),
                title: "Prelude - BWV 846".into(),
            },
            CatalogPiece {
                piece_id: "chopin.etudes_op_10.3".into(),
                composer: "Chopin".into(),
                title: "Etudes Op. 10 No. 3".into(),
            },
            CatalogPiece {
                piece_id: "bach.fugue.bwv_846".into(),
                composer: "Bach".into(),
                title: "Fugue - BWV 846".into(),
            },
        ]
    }

    #[test]
    fn matches_chopin_ballade_casual() {
        let catalog = sample_catalog();
        let result = match_piece("chopin ballade 1", &catalog);
        let m = result.expect("should match");
        assert_eq!(m.piece_id, "chopin.ballades.1");
        assert!(m.confidence > 0.3);
    }

    #[test]
    fn matches_chopin_ballade_number_only() {
        let catalog = sample_catalog();
        // Should prefer "Ballade No. 1" over "Ballade No. 4"
        let result = match_piece("chopin ballade no 1", &catalog);
        let m = result.expect("should match");
        assert_eq!(m.piece_id, "chopin.ballades.1");
    }

    #[test]
    fn matches_bach_prelude() {
        let catalog = sample_catalog();
        let result = match_piece("bach prelude bwv 846", &catalog);
        let m = result.expect("should match");
        assert_eq!(m.piece_id, "bach.prelude.bwv_846");
    }

    #[test]
    fn no_match_for_unknown_piece() {
        let catalog = sample_catalog();
        let result = match_piece("debussy clair de lune", &catalog);
        assert!(result.is_none());
    }

    #[test]
    fn empty_query_returns_none() {
        let catalog = sample_catalog();
        assert!(match_piece("", &catalog).is_none());
        assert!(match_piece("  ", &catalog).is_none());
    }

    #[test]
    fn empty_catalog_returns_none() {
        assert!(match_piece("chopin ballade", &[]).is_none());
    }

    #[test]
    fn case_insensitive() {
        let catalog = sample_catalog();
        let result = match_piece("CHOPIN BALLADE", &catalog);
        assert!(result.is_some());
    }

    #[test]
    fn filters_by_composer() {
        let catalog = sample_catalog();
        // "chopin etude" should match the Chopin etude, not Bach
        let result = match_piece("chopin etude", &catalog);
        let m = result.expect("should match");
        assert_eq!(m.composer, "Chopin");
        assert_eq!(m.piece_id, "chopin.etudes_op_10.3");
    }

    #[test]
    fn normalize_removes_punctuation() {
        assert_eq!(normalize("No. 14 in C-sharp"), "no 14 in c sharp");
    }

    #[test]
    fn dice_identical_strings() {
        assert!((dice_similarity("hello", "hello") - 1.0).abs() < 0.01);
    }

    #[test]
    fn dice_completely_different() {
        assert!(dice_similarity("abc", "xyz") < 0.1);
    }
}
```

- [ ] **Step 2: Register module**

Add to `apps/api/src/practice/mod.rs`:

```rust
pub mod piece_match;
```

- [ ] **Step 3: Run tests to verify they fail then pass**

Run: `cd apps/api && cargo test piece_match -- --nocapture`
Expected: All 11 tests PASS (since implementation is inline with tests).

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/practice/piece_match.rs apps/api/src/practice/mod.rs
git commit -m "feat: add fuzzy piece matching against score catalog"
```

---

## Chunk 3: Score Context Loading

### Task 4: Score Context Module

Loads the full score JSON from R2 for the matched piece, deserializes it, and caches it in the DO's session state. Also handles logging unmatched queries to D1 for demand tracking.

**Files:**
- Create: `apps/api/src/practice/score_context.rs`
- Modify: `apps/api/src/practice/mod.rs`

- [ ] **Step 1: Write score context module**

Create `apps/api/src/practice/score_context.rs`:

```rust
/// Session-level score context: piece matching, R2 score loading, and demand tracking.
///
/// ScoreContext is loaded once per session (when the student identifies their piece)
/// and cached in SessionState for the session lifetime.

use serde::{Deserialize, Serialize};
use wasm_bindgen::JsValue;
use worker::{console_error, Env};

use crate::practice::piece_match::{CatalogPiece, MatchResult};

/// A single note from the score MIDI, deserialized from R2 JSON.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScoreNote {
    pub pitch: u8,
    pub pitch_name: String,
    pub velocity: u8,
    pub onset_tick: u32,
    pub onset_seconds: f64,
    pub duration_ticks: u32,
    pub duration_seconds: f64,
    pub track: u8,
}

/// A pedal event from the score MIDI.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScorePedalEvent {
    #[serde(rename = "type")]
    pub event_type: String, // "on" or "off"
    pub tick: u32,
    pub seconds: f64,
}

/// A single bar from the score, with all its notes and pedal events.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScoreBar {
    pub bar_number: u32,
    pub start_tick: u32,
    pub start_seconds: f64,
    pub time_signature: String,
    pub notes: Vec<ScoreNote>,
    pub pedal_events: Vec<ScorePedalEvent>,
    pub note_count: u32,
    pub pitch_range: Vec<u8>,
    pub mean_velocity: u8,
}

/// Full score data loaded from R2.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScoreData {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
    pub key_signature: Option<String>,
    pub time_signatures: Vec<serde_json::Value>,
    pub tempo_markings: Vec<serde_json::Value>,
    pub total_bars: u32,
    pub bars: Vec<ScoreBar>,
}

/// Reference performance statistics per bar (from MAESTRO professional recordings).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReferenceBar {
    pub bar_number: u32,
    pub velocity_mean: f64,
    pub velocity_std: f64,
    pub onset_deviation_mean_ms: f64,
    pub onset_deviation_std_ms: f64,
    pub pedal_duration_mean_beats: Option<f64>,
    pub pedal_changes: Option<u32>,
    pub note_duration_ratio_mean: f64,   // perf_duration / score_duration
    pub performer_count: u32,
}

/// Full reference profile for a piece (multiple MAESTRO performers aggregated).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReferenceProfile {
    pub piece_id: String,
    pub performer_count: u32,
    pub bars: Vec<ReferenceBar>,
}

/// Cached score context for a session. Loaded once, used for every chunk.
#[derive(Debug, Clone)]
pub struct ScoreContext {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
    pub score: ScoreData,
    pub reference: Option<ReferenceProfile>,
    pub match_confidence: f64,
}

/// Load the piece catalog from D1 for fuzzy matching.
pub async fn load_catalog(env: &Env) -> Vec<CatalogPiece> {
    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed: {:?}", e);
            return vec![];
        }
    };

    let stmt = db.prepare("SELECT piece_id, composer, title FROM pieces");
    let result = match stmt.all().await {
        Ok(r) => r,
        Err(e) => {
            console_error!("Catalog query failed: {:?}", e);
            return vec![];
        }
    };

    let rows = match result.results::<serde_json::Value>() {
        Ok(rows) => rows,
        Err(e) => {
            console_error!("Catalog parse failed: {:?}", e);
            return vec![];
        }
    };

    rows.iter()
        .filter_map(|row| {
            Some(CatalogPiece {
                piece_id: row.get("piece_id")?.as_str()?.to_string(),
                composer: row.get("composer")?.as_str()?.to_string(),
                title: row.get("title")?.as_str()?.to_string(),
            })
        })
        .collect()
}

/// Fetch score JSON from R2 and deserialize.
pub async fn load_score(env: &Env, piece_id: &str) -> Result<ScoreData, String> {
    let bucket = env.bucket("SCORES")
        .map_err(|e| format!("SCORES R2 binding failed: {:?}", e))?;
    let key = format!("scores/v1/{}.json", piece_id);
    let object = bucket.get(&key).execute().await
        .map_err(|e| format!("R2 get failed for {}: {:?}", key, e))?;
    let object = object.ok_or_else(|| format!("Score not found in R2: {}", key))?;
    let bytes = object.body()
        .ok_or("Score object has no body")?
        .bytes().await
        .map_err(|e| format!("R2 read failed: {:?}", e))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| format!("Score JSON parse failed for {}: {:?}", piece_id, e))
}

/// Fetch reference profile from R2. Returns None if not available (not all pieces have references).
pub async fn load_reference(env: &Env, piece_id: &str) -> Option<ReferenceProfile> {
    let bucket = match env.bucket("SCORES") {
        Ok(b) => b,
        Err(_) => return None,
    };
    let key = format!("references/v1/{}.json", piece_id);
    let object = match bucket.get(&key).execute().await {
        Ok(Some(obj)) => obj,
        _ => return None,
    };
    let bytes = match object.body()?.bytes().await {
        Ok(b) => b,
        Err(_) => return None,
    };
    serde_json::from_slice(&bytes).ok()
}

/// Log a piece identification request to D1 for demand tracking.
pub async fn log_piece_request(
    env: &Env,
    query: &str,
    student_id: &str,
    match_result: Option<&MatchResult>,
) {
    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(_) => return,
    };

    let id = crate::services::ask::generate_uuid();
    let (piece_id, confidence): (Option<&str>, Option<f64>) = match match_result {
        Some(m) => (Some(&m.piece_id), Some(m.confidence)),
        None => (None, None),
    };

    let stmt = db.prepare(
        "INSERT INTO piece_requests (id, query, student_id, matched_piece_id, match_confidence) VALUES (?1, ?2, ?3, ?4, ?5)"
    );
    let bound = match stmt.bind(&[
        id.into(),
        query.into(),
        student_id.into(),
        piece_id.map(|s| s.into()).unwrap_or(JsValue::NULL),
        confidence.map(|c| c.into()).unwrap_or(JsValue::NULL),
    ]) {
        Ok(s) => s,
        Err(_) => return,
    };
    let _ = bound.run().await;
}

/// Full flow: match query -> load score + reference -> return ScoreContext.
pub async fn resolve_piece(
    env: &Env,
    query: &str,
    student_id: &str,
) -> Option<ScoreContext> {
    let catalog = load_catalog(env).await;
    let match_result = crate::practice::piece_match::match_piece(query, &catalog);

    // Log for demand tracking (both matched and unmatched)
    log_piece_request(env, query, student_id, match_result.as_ref()).await;

    let m = match_result?;

    let score = match load_score(env, &m.piece_id).await {
        Ok(s) => s,
        Err(e) => {
            console_error!("Failed to load score for {}: {}", m.piece_id, e);
            return None;
        }
    };

    let reference = load_reference(env, &m.piece_id).await;

    Some(ScoreContext {
        piece_id: m.piece_id,
        composer: m.composer,
        title: m.title,
        score,
        reference,
        match_confidence: m.confidence,
    })
}
```

- [ ] **Step 2: Register module**

Add to `apps/api/src/practice/mod.rs`:

```rust
pub mod score_context;
```

- [ ] **Step 3: Verify compilation**

Run: `cd apps/api && cargo check`
Expected: Compiles. (No unit tests for this module -- it's I/O-heavy; tested via integration.)

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/practice/score_context.rs apps/api/src/practice/mod.rs apps/api/migrations/0005_piece_requests.sql
git commit -m "feat: add score context loading from R2 with demand tracking"
```

---

## Chunk 4: Score Following (DTW)

### Task 5: Subsequence DTW Score Follower

Maps AMT performance MIDI (note onsets from a 15s chunk) to bar numbers in the score using onset+pitch subsequence DTW. Maintains cross-chunk continuity via `last_known_bar` in session state.

**Files:**
- Create: `apps/api/src/practice/score_follower.rs`
- Modify: `apps/api/src/practice/mod.rs`

- [ ] **Step 1: Write score follower with inline tests**

Create `apps/api/src/practice/score_follower.rs`:

```rust
/// Score following via onset+pitch subsequence DTW.
///
/// Maps AMT performance notes (from a 15s chunk) to bar numbers in the score.
/// Cross-chunk continuity: uses `last_known_bar` to restrict the search window,
/// with full re-scan fallback when the student skips ahead or restarts.
///
/// Algorithm:
/// 1. Flatten score notes in the search window into (onset_seconds, pitch) pairs
/// 2. Build cost matrix: |onset_perf - onset_score| + pitch_mismatch_penalty
/// 3. Subsequence DTW: find the best-matching region of the score
/// 4. Map aligned notes back to bar numbers
/// 5. Return bar range and per-note alignments

use crate::practice::score_context::{ScoreBar, ScoreData};

/// A performance note from AMT (subset of HF response).
#[derive(Debug, Clone)]
pub struct PerfNote {
    pub pitch: u8,
    pub onset: f64,     // seconds relative to chunk start
    pub offset: f64,
    pub velocity: u8,
}

/// A pedal event from AMT.
#[derive(Debug, Clone)]
pub struct PerfPedalEvent {
    pub time: f64,      // seconds relative to chunk start
    pub value: u8,      // 0-127, >64 = on
}

/// Alignment of a single performance note to a score note.
#[derive(Debug, Clone)]
pub struct NoteAlignment {
    pub perf_onset: f64,
    pub perf_pitch: u8,
    pub perf_velocity: u8,
    pub score_bar: u32,
    pub score_beat: f64,
    pub score_pitch: u8,
    pub onset_deviation_ms: f64,  // positive = late, corrected for alignment offset
}

/// Bar map for a single chunk: which bars the student was playing.
#[derive(Debug, Clone)]
pub struct BarMap {
    pub chunk_index: usize,
    pub bar_start: u32,
    pub bar_end: u32,
    pub alignments: Vec<NoteAlignment>,
    pub confidence: f64,          // 0-1, lower = better alignment
    pub is_reanchored: bool,      // true if search window was widened
}

/// Persistent state across chunks for continuity.
#[derive(Debug, Clone, Default)]
pub struct FollowerState {
    pub last_known_bar: Option<u32>,
}

const PITCH_MISMATCH_PENALTY: f64 = 0.5;  // seconds equivalent
const SEARCH_WINDOW_BARS: u32 = 30;
const REANCHOR_COST_THRESHOLD: f64 = 0.3; // normalized DTW cost above this = re-anchor
const MIN_PERF_NOTES: usize = 3;

/// Flatten score bars into a list of (onset_seconds, pitch, bar_number, beat_in_bar).
fn flatten_score_notes(bars: &[ScoreBar]) -> Vec<(f64, u8, u32, f64)> {
    let mut notes = Vec::new();
    for (i, bar) in bars.iter().enumerate() {
        let bar_start = bar.start_seconds;
        let beats_per_bar: f64 = match bar.time_signature.split('/').next() {
            Some(n) => n.parse().unwrap_or(4.0),
            None => 4.0,
        };
        // O(1) bar duration from next bar's start_seconds
        let bar_duration = if i + 1 < bars.len() {
            bars[i + 1].start_seconds - bar.start_seconds
        } else {
            2.0 // fallback for last bar
        };

        for note in &bar.notes {
            let beat = if bar_duration > 0.0 {
                ((note.onset_seconds - bar_start) / bar_duration) * beats_per_bar
            } else {
                0.0
            };
            notes.push((note.onset_seconds, note.pitch, bar.bar_number, beat));
        }
    }
    notes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    notes
}

/// Compute subsequence DTW between performance notes and a score window.
/// Returns (aligned_pairs, normalized_cost).
fn subsequence_dtw(
    perf: &[(f64, u8)],          // (onset, pitch)
    score: &[(f64, u8, u32, f64)], // (onset, pitch, bar, beat)
) -> (Vec<(usize, usize)>, f64) {
    let n = perf.len();
    let m = score.len();
    if n == 0 || m == 0 {
        return (vec![], f64::MAX);
    }

    // Cost matrix
    let cost = |i: usize, j: usize| -> f64 {
        let onset_diff = (perf[i].0 - score[j].0).abs();
        let pitch_penalty = if perf[i].1 == score[j].1 {
            0.0
        } else if (perf[i].1 as i16 - score[j].1 as i16).abs() <= 1 {
            PITCH_MISMATCH_PENALTY * 0.25 // semitone tolerance
        } else if (perf[i].1 as i16 - score[j].1 as i16).abs() == 12 {
            PITCH_MISMATCH_PENALTY * 0.5  // octave error (common AMT mistake)
        } else {
            PITCH_MISMATCH_PENALTY
        };
        onset_diff + pitch_penalty
    };

    // DTW matrix: subsequence DTW initializes first row to 0
    // (performance can start matching anywhere in the score)
    let mut dtw = vec![vec![f64::MAX; m + 1]; n + 1];
    for j in 0..=m {
        dtw[0][j] = 0.0; // subsequence: free start in score
    }

    for i in 1..=n {
        for j in 1..=m {
            let c = cost(i - 1, j - 1);
            dtw[i][j] = c + dtw[i - 1][j - 1]
                .min(dtw[i - 1][j])    // insertion (skip score note)
                .min(dtw[i][j - 1]);   // deletion (skip perf note)
        }
    }

    // Find best endpoint in last row (where performance ends)
    let mut best_j = 0;
    let mut best_cost = f64::MAX;
    for j in 1..=m {
        if dtw[n][j] < best_cost {
            best_cost = dtw[n][j];
            best_j = j;
        }
    }

    // Backtrace
    let mut path = Vec::new();
    let mut i = n;
    let mut j = best_j;
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let diag = dtw[i - 1][j - 1];
        let up = dtw[i - 1][j];
        let left = dtw[i][j - 1];
        if diag <= up && diag <= left {
            i -= 1;
            j -= 1;
        } else if up <= left {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    path.reverse();

    let normalized = if n > 0 { best_cost / n as f64 } else { f64::MAX };
    (path, normalized)
}

/// Main entry point: align a chunk's performance notes to the score.
pub fn align_chunk(
    chunk_index: usize,
    chunk_offset_seconds: f64, // chunk_index * 15.0
    perf_notes: &[PerfNote],
    score: &ScoreData,
    state: &mut FollowerState,
) -> Option<BarMap> {
    if perf_notes.len() < MIN_PERF_NOTES {
        return None;
    }

    // Determine search window
    let (search_bars, is_wide) = match state.last_known_bar {
        Some(last) => {
            let start = last.saturating_sub(5); // small backward tolerance
            let end = (last + SEARCH_WINDOW_BARS).min(score.total_bars);
            let bars: Vec<&ScoreBar> = score.bars.iter()
                .filter(|b| b.bar_number >= start && b.bar_number <= end)
                .collect();
            (bars, false)
        }
        None => {
            // First chunk or re-anchor: search full score
            (score.bars.iter().collect::<Vec<_>>(), true)
        }
    };

    if search_bars.is_empty() {
        return None;
    }

    // Flatten score notes in window
    let owned_bars: Vec<ScoreBar> = search_bars.into_iter().cloned().collect();
    let score_notes = flatten_score_notes(&owned_bars);
    if score_notes.is_empty() {
        return None;
    }

    // Build performance onset+pitch pairs
    let perf_pairs: Vec<(f64, u8)> = perf_notes.iter()
        .map(|n| (n.onset, n.pitch))
        .collect();

    // Normalize score onsets relative to the search window start
    let window_start = score_notes.first().map(|n| n.0).unwrap_or(0.0);
    let score_pairs_rel: Vec<(f64, u8, u32, f64)> = score_notes.iter()
        .map(|&(onset, pitch, bar, beat)| (onset - window_start, pitch, bar, beat))
        .collect();

    let (path, cost) = subsequence_dtw(&perf_pairs, &score_pairs_rel);

    // Check if we need to re-anchor (cost too high in narrow window)
    if !is_wide && cost > REANCHOR_COST_THRESHOLD {
        // Re-anchor: search full score
        let all_bars: Vec<ScoreBar> = score.bars.clone();
        let all_notes = flatten_score_notes(&all_bars);
        if all_notes.is_empty() {
            return None;
        }
        let all_start = all_notes.first().map(|n| n.0).unwrap_or(0.0);
        let all_rel: Vec<(f64, u8, u32, f64)> = all_notes.iter()
            .map(|&(onset, pitch, bar, beat)| (onset - all_start, pitch, bar, beat))
            .collect();
        let (path2, cost2) = subsequence_dtw(&perf_pairs, &all_rel);
        return build_bar_map(chunk_index, &perf_pairs, perf_notes, &all_rel, &path2, cost2, true, state);
    }

    build_bar_map(chunk_index, &perf_pairs, perf_notes, &score_pairs_rel, &path, cost, is_wide, state)
}

fn build_bar_map(
    chunk_index: usize,
    perf: &[(f64, u8)],
    perf_notes: &[PerfNote],  // full notes with velocity
    score: &[(f64, u8, u32, f64)],
    path: &[(usize, usize)],
    cost: f64,
    is_reanchored: bool,
    state: &mut FollowerState,
) -> Option<BarMap> {
    if path.is_empty() {
        return None;
    }

    // Compute median onset offset to correct for alignment shift.
    // Raw (perf_onset - score_onset) conflates the alignment offset with
    // actual timing deviations. Subtracting the median isolates true deviations.
    let raw_offsets: Vec<f64> = path.iter()
        .map(|&(pi, si)| perf[pi].0 - score[si].0)
        .collect();
    let median_offset = {
        let mut sorted = raw_offsets.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    };

    let alignments: Vec<NoteAlignment> = path.iter()
        .map(|&(pi, si)| {
            let (p_onset, p_pitch) = perf[pi];
            let (s_onset, s_pitch, s_bar, s_beat) = score[si];
            let velocity = perf_notes.get(pi).map(|n| n.velocity).unwrap_or(80);
            NoteAlignment {
                perf_onset: p_onset,
                perf_pitch: p_pitch,
                perf_velocity: velocity,
                score_bar: s_bar,
                score_beat: s_beat,
                score_pitch: s_pitch,
                onset_deviation_ms: (p_onset - s_onset - median_offset) * 1000.0,
            }
        })
        .collect();

    let bars: Vec<u32> = alignments.iter().map(|a| a.score_bar).collect();
    let bar_start = *bars.iter().min()?;
    let bar_end = *bars.iter().max()?;

    // Update follower state for next chunk
    state.last_known_bar = Some(bar_end);

    Some(BarMap {
        chunk_index,
        bar_start,
        bar_end,
        alignments,
        confidence: 1.0 / (1.0 + cost), // smooth mapping: higher = better
        is_reanchored,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::practice::score_context::ScoreNote as SN;

    fn make_score_bar(bar_num: u32, start_sec: f64, notes: Vec<(u8, f64)>) -> ScoreBar {
        ScoreBar {
            bar_number: bar_num,
            start_tick: 0,
            start_seconds: start_sec,
            time_signature: "4/4".into(),
            notes: notes.iter().map(|&(pitch, onset)| SN {
                pitch,
                pitch_name: String::new(),
                velocity: 80,
                onset_tick: 0,
                onset_seconds: onset,
                duration_ticks: 100,
                duration_seconds: 0.25,
                track: 0,
            }).collect(),
            pedal_events: vec![],
            note_count: notes.len() as u32,
            pitch_range: vec![60, 72],
            mean_velocity: 80,
        }
    }

    fn make_score(bars: Vec<ScoreBar>) -> ScoreData {
        let total = bars.len() as u32;
        ScoreData {
            piece_id: "test".into(),
            composer: "Test".into(),
            title: "Test Piece".into(),
            key_signature: Some("C".into()),
            time_signatures: vec![],
            tempo_markings: vec![],
            total_bars: total,
            bars,
        }
    }

    #[test]
    fn aligns_simple_ascending_scale() {
        // Score: C4-D4-E4-F4 in bar 1 at 0.0, 0.5, 1.0, 1.5s
        let score = make_score(vec![
            make_score_bar(1, 0.0, vec![(60, 0.0), (62, 0.5), (64, 1.0), (65, 1.5)]),
        ]);

        // Performance: same scale, slightly shifted
        let perf = vec![
            PerfNote { pitch: 60, onset: 0.05, offset: 0.4, velocity: 75 },
            PerfNote { pitch: 62, onset: 0.52, offset: 0.9, velocity: 80 },
            PerfNote { pitch: 64, onset: 1.03, offset: 1.4, velocity: 70 },
            PerfNote { pitch: 65, onset: 1.55, offset: 1.9, velocity: 85 },
        ];

        let mut state = FollowerState::default();
        let result = align_chunk(0, 0.0, &perf, &score, &mut state);
        let bar_map = result.expect("should align");

        assert_eq!(bar_map.bar_start, 1);
        assert_eq!(bar_map.bar_end, 1);
        assert_eq!(bar_map.alignments.len(), 4);
        assert!(bar_map.confidence > 0.5);
        assert_eq!(state.last_known_bar, Some(1));
    }

    #[test]
    fn aligns_across_bars() {
        let score = make_score(vec![
            make_score_bar(1, 0.0, vec![(60, 0.0), (62, 0.5)]),
            make_score_bar(2, 1.0, vec![(64, 1.0), (65, 1.5)]),
            make_score_bar(3, 2.0, vec![(67, 2.0), (69, 2.5)]),
        ]);

        let perf = vec![
            PerfNote { pitch: 60, onset: 0.0, offset: 0.4, velocity: 80 },
            PerfNote { pitch: 62, onset: 0.5, offset: 0.9, velocity: 80 },
            PerfNote { pitch: 64, onset: 1.0, offset: 1.4, velocity: 80 },
            PerfNote { pitch: 65, onset: 1.5, offset: 1.9, velocity: 80 },
        ];

        let mut state = FollowerState::default();
        let result = align_chunk(0, 0.0, &perf, &score, &mut state);
        let bar_map = result.expect("should align");

        assert_eq!(bar_map.bar_start, 1);
        assert_eq!(bar_map.bar_end, 2);
    }

    #[test]
    fn too_few_notes_returns_none() {
        let score = make_score(vec![
            make_score_bar(1, 0.0, vec![(60, 0.0), (62, 0.5)]),
        ]);

        let perf = vec![
            PerfNote { pitch: 60, onset: 0.0, offset: 0.4, velocity: 80 },
        ];

        let mut state = FollowerState::default();
        let result = align_chunk(0, 0.0, &perf, &score, &mut state);
        assert!(result.is_none());
    }

    #[test]
    fn continuity_across_chunks() {
        let score = make_score(vec![
            make_score_bar(1, 0.0, vec![(60, 0.0), (62, 0.5)]),
            make_score_bar(2, 1.0, vec![(64, 1.0), (65, 1.5)]),
            make_score_bar(3, 2.0, vec![(67, 2.0), (69, 2.5)]),
            make_score_bar(4, 3.0, vec![(71, 3.0), (72, 3.5)]),
        ]);

        let perf_chunk_0 = vec![
            PerfNote { pitch: 60, onset: 0.0, offset: 0.4, velocity: 80 },
            PerfNote { pitch: 62, onset: 0.5, offset: 0.9, velocity: 80 },
            PerfNote { pitch: 64, onset: 1.0, offset: 1.4, velocity: 80 },
        ];

        let mut state = FollowerState::default();
        let map0 = align_chunk(0, 0.0, &perf_chunk_0, &score, &mut state).unwrap();
        assert!(state.last_known_bar.is_some());

        let perf_chunk_1 = vec![
            PerfNote { pitch: 67, onset: 0.0, offset: 0.4, velocity: 80 },
            PerfNote { pitch: 69, onset: 0.5, offset: 0.9, velocity: 80 },
            PerfNote { pitch: 71, onset: 1.0, offset: 1.4, velocity: 80 },
        ];

        let map1 = align_chunk(1, 15.0, &perf_chunk_1, &score, &mut state).unwrap();
        assert!(map1.bar_start >= map0.bar_end);
    }

    #[test]
    fn handles_octave_error() {
        // Score has C4, performance has C5 (octave error from AMT)
        let score = make_score(vec![
            make_score_bar(1, 0.0, vec![(60, 0.0), (62, 0.5), (64, 1.0)]),
        ]);

        let perf = vec![
            PerfNote { pitch: 72, onset: 0.0, offset: 0.4, velocity: 80 }, // C5 instead of C4
            PerfNote { pitch: 62, onset: 0.5, offset: 0.9, velocity: 80 },
            PerfNote { pitch: 64, onset: 1.0, offset: 1.4, velocity: 80 },
        ];

        let mut state = FollowerState::default();
        let result = align_chunk(0, 0.0, &perf, &score, &mut state);
        assert!(result.is_some(), "should tolerate octave errors");
    }
}
```

- [ ] **Step 2: Register module**

Add to `apps/api/src/practice/mod.rs`:

```rust
pub mod score_follower;
```

- [ ] **Step 3: Run tests**

Run: `cd apps/api && cargo test score_follower -- --nocapture`
Expected: All 5 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/practice/score_follower.rs apps/api/src/practice/mod.rs
git commit -m "feat: add onset+pitch DTW score follower with cross-chunk continuity"
```

---

## Chunk 5: Bar-Aligned Analysis Engine

### Task 6: Musical Analysis Engine (All 6 Dimensions)

Transforms model scores + AMT MIDI + score data + reference profiles into structured musical facts per dimension for the bar range identified by the score follower. Three tiers: Tier 1 (full bar-aligned with score+reference), Tier 2 (absolute MIDI analysis, no score), Tier 3 (scores only, no MIDI).

**Files:**
- Create: `apps/api/src/practice/analysis.rs`
- Modify: `apps/api/src/practice/mod.rs`

- [ ] **Step 1: Write analysis engine**

Create `apps/api/src/practice/analysis.rs`:

```rust
/// Bar-aligned musical analysis engine.
///
/// Produces structured musical facts per dimension by comparing:
/// - Performance MIDI (from AMT) against score MIDI
/// - Performance statistics against reference profiles (MAESTRO)
/// - Model dimension scores against student baselines
///
/// Three tiers of output:
/// - Tier 1: Full bar-aligned facts (piece known, AMT succeeded)
/// - Tier 2: Absolute MIDI facts (piece unknown, AMT succeeded)
/// - Tier 3: Scores only (AMT failed -- current behavior, no analysis)

use serde::Serialize;

use crate::practice::score_context::{ReferenceBar, ReferenceProfile, ScoreBar, ScoreContext};
use crate::practice::score_follower::{BarMap, NoteAlignment, PerfNote, PerfPedalEvent};

/// Analysis output for a single dimension.
#[derive(Debug, Clone, Serialize)]
pub struct DimensionAnalysis {
    pub dimension: String,
    pub analysis: String,           // Human-readable sentence for subagent
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_marking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_comparison: Option<String>,
}

/// Full analysis output for a chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkAnalysis {
    pub tier: u8,   // 1, 2, or 3
    pub bar_range: Option<String>,
    pub dimensions: Vec<DimensionAnalysis>,
}

/// Analyze a chunk with full score context (Tier 1).
pub fn analyze_tier1(
    bar_map: &BarMap,
    perf_notes: &[PerfNote],
    perf_pedal: &[PerfPedalEvent],
    scores: &[f64; 6], // [dynamics, timing, pedaling, articulation, phrasing, interpretation]
    score_ctx: &ScoreContext,
) -> ChunkAnalysis {
    let bar_range_str = format!("{}-{}", bar_map.bar_start, bar_map.bar_end);

    // Get score bars in range
    let score_bars: Vec<&ScoreBar> = score_ctx.score.bars.iter()
        .filter(|b| b.bar_number >= bar_map.bar_start && b.bar_number <= bar_map.bar_end)
        .collect();

    // Get reference bars if available
    let ref_bars: Vec<&ReferenceBar> = score_ctx.reference.as_ref()
        .map(|r| r.bars.iter()
            .filter(|b| b.bar_number >= bar_map.bar_start && b.bar_number <= bar_map.bar_end)
            .collect())
        .unwrap_or_default();

    let dims = vec![
        analyze_dynamics(&bar_map.alignments, &score_bars, &ref_bars, scores[0]),
        analyze_timing(&bar_map.alignments, &score_bars, &ref_bars, scores[1]),
        analyze_pedaling(perf_pedal, &score_bars, &ref_bars, scores[2]),
        analyze_articulation(perf_notes, &bar_map.alignments, &score_bars, &ref_bars, scores[3]),
        analyze_phrasing(&bar_map.alignments, &score_bars, &ref_bars, scores[4]),
        analyze_interpretation(&bar_map.alignments, &score_bars, &ref_bars, scores[5]),
    ];

    ChunkAnalysis {
        tier: 1,
        bar_range: Some(bar_range_str),
        dimensions: dims,
    }
}

/// Analyze with AMT data but no score context (Tier 2).
pub fn analyze_tier2(
    perf_notes: &[PerfNote],
    perf_pedal: &[PerfPedalEvent],
    scores: &[f64; 6],
) -> ChunkAnalysis {
    let dims = vec![
        analyze_dynamics_absolute(perf_notes, scores[0]),
        analyze_timing_absolute(perf_notes, scores[1]),
        analyze_pedaling_absolute(perf_pedal, scores[2]),
        analyze_articulation_absolute(perf_notes, scores[3]),
        analyze_phrasing_absolute(perf_notes, scores[4]),
        analyze_interpretation_absolute(scores[5]),
    ];

    ChunkAnalysis {
        tier: 2,
        bar_range: None,
        dimensions: dims,
    }
}

// --- Tier 1 dimension analyzers (with score + reference) ---

fn analyze_dynamics(
    alignments: &[NoteAlignment],
    score_bars: &[&ScoreBar],
    ref_bars: &[&ReferenceBar],
    model_score: f64,
) -> DimensionAnalysis {
    // Performance velocity stats from aligned notes (velocity carried in NoteAlignment)
    let perf_velocities: Vec<u8> = alignments.iter()
        .map(|a| a.perf_velocity)
        .collect();

    // Score velocity stats
    let score_velocities: Vec<u8> = score_bars.iter()
        .flat_map(|b| b.notes.iter().map(|n| n.velocity))
        .collect();

    let perf_mean = mean_u8(&perf_velocities);
    let score_mean = mean_u8(&score_velocities);

    let mut analysis = format!(
        "Performance velocity avg {:.0}, score reference avg {:.0}.",
        perf_mean, score_mean,
    );

    // Velocity curve direction (crescendo/diminuendo detection)
    if alignments.len() >= 4 {
        let mid = alignments.len() / 2;
        let first_half = mean_u8(&perf_velocities[..mid]);
        let second_half = mean_u8(&perf_velocities[mid..]);
        if second_half > first_half * 1.15 {
            analysis.push_str(" Crescendo detected.");
        } else if first_half > second_half * 1.15 {
            analysis.push_str(" Diminuendo detected.");
        } else {
            analysis.push_str(" Dynamic level relatively flat.");
        }
    }

    let ref_comparison = ref_bars.first().map(|r| {
        format!(
            "Reference velocity avg {:.0} (std {:.1}, {} performers).",
            r.velocity_mean, r.velocity_std, r.performer_count
        )
    });

    DimensionAnalysis {
        dimension: "dynamics".into(),
        analysis,
        score_marking: None, // Requires MusicXML for pp/ff markings (future)
        reference_comparison: ref_comparison,
    }
}

fn analyze_timing(
    alignments: &[NoteAlignment],
    _score_bars: &[&ScoreBar],
    ref_bars: &[&ReferenceBar],
    model_score: f64,
) -> DimensionAnalysis {
    let deviations: Vec<f64> = alignments.iter()
        .map(|a| a.onset_deviation_ms)
        .collect();

    let mean_dev = mean_f64(&deviations);
    let std_dev = std_f64(&deviations);

    let tendency = if mean_dev > 30.0 {
        "tending late (dragging)"
    } else if mean_dev < -30.0 {
        "tending early (rushing)"
    } else {
        "close to score timing"
    };

    let analysis = format!(
        "Mean onset deviation {:.0}ms (std {:.0}ms), {}. Model score {:.2}.",
        mean_dev, std_dev, tendency, model_score,
    );

    let ref_comparison = ref_bars.first().map(|r| {
        format!(
            "Reference deviation {:.0}ms (std {:.0}ms, {} performers).",
            r.onset_deviation_mean_ms, r.onset_deviation_std_ms, r.performer_count,
        )
    });

    DimensionAnalysis {
        dimension: "timing".into(),
        analysis,
        score_marking: None,
        reference_comparison: ref_comparison,
    }
}

fn analyze_pedaling(
    perf_pedal: &[PerfPedalEvent],
    score_bars: &[&ScoreBar],
    ref_bars: &[&ReferenceBar],
    model_score: f64,
) -> DimensionAnalysis {
    // Count pedal on/off pairs in performance
    let perf_changes = perf_pedal.len();

    // Count score pedal events in bar range
    let score_pedal_count: usize = score_bars.iter()
        .flat_map(|b| b.pedal_events.iter())
        .count();

    // Compute average pedal duration from on/off pairs
    let mut durations = Vec::new();
    let mut last_on: Option<f64> = None;
    for event in perf_pedal {
        if event.value > 64 {
            last_on = Some(event.time);
        } else if let Some(on_time) = last_on {
            durations.push(event.time - on_time);
            last_on = None;
        }
    }
    let avg_duration = if durations.is_empty() { 0.0 } else { mean_f64(&durations) };

    let analysis = if perf_changes == 0 && score_pedal_count > 0 {
        format!(
            "No pedal detected in performance. Score has {} pedal markings. Model score {:.2}.",
            score_pedal_count, model_score,
        )
    } else if perf_changes > 0 {
        format!(
            "{} pedal changes detected (avg duration {:.1}s). Score has {} markings. Model score {:.2}.",
            perf_changes, avg_duration, score_pedal_count, model_score,
        )
    } else {
        format!(
            "No pedal events in score or performance. Model score {:.2}.",
            model_score,
        )
    };

    let ref_comparison = ref_bars.first().and_then(|r| {
        r.pedal_duration_mean_beats.map(|pd| {
            format!(
                "Reference pedal avg {:.1} beats, {} changes ({} performers).",
                pd, r.pedal_changes.unwrap_or(0), r.performer_count,
            )
        })
    });

    DimensionAnalysis {
        dimension: "pedaling".into(),
        analysis,
        score_marking: None,
        reference_comparison: ref_comparison,
    }
}

fn analyze_articulation(
    perf_notes: &[PerfNote],
    alignments: &[NoteAlignment],
    score_bars: &[&ScoreBar],
    ref_bars: &[&ReferenceBar],
    model_score: f64,
) -> DimensionAnalysis {
    // Compare note durations: performance vs score
    // Legato = duration ratio > 0.9, staccato < 0.5
    let score_notes: Vec<f64> = score_bars.iter()
        .flat_map(|b| b.notes.iter().map(|n| n.duration_seconds))
        .collect();

    let perf_durations: Vec<f64> = perf_notes.iter()
        .map(|n| n.offset - n.onset)
        .collect();

    let perf_mean_dur = mean_f64(&perf_durations);
    let score_mean_dur = mean_f64(&score_notes);

    let ratio = if score_mean_dur > 0.0 { perf_mean_dur / score_mean_dur } else { 1.0 };

    let style = if ratio > 1.1 {
        "more legato than written"
    } else if ratio < 0.6 {
        "more detached/staccato than written"
    } else {
        "close to written articulation"
    };

    let analysis = format!(
        "Note duration ratio {:.2}x vs score ({}). Model score {:.2}.",
        ratio, style, model_score,
    );

    let ref_comparison = ref_bars.first().map(|r| {
        format!(
            "Reference duration ratio {:.2}x ({} performers).",
            r.note_duration_ratio_mean, r.performer_count,
        )
    });

    DimensionAnalysis {
        dimension: "articulation".into(),
        analysis,
        score_marking: None,
        reference_comparison: ref_comparison,
    }
}

fn analyze_phrasing(
    alignments: &[NoteAlignment],
    _score_bars: &[&ScoreBar],
    _ref_bars: &[&ReferenceBar],
    model_score: f64,
) -> DimensionAnalysis {
    // Phrasing: multi-bar patterns in dynamics + timing
    // Simplified: detect if there's a velocity arc (shape) across the passage
    let deviations: Vec<f64> = alignments.iter()
        .map(|a| a.onset_deviation_ms)
        .collect();

    let analysis = if deviations.len() >= 6 {
        let first_third = mean_f64(&deviations[..deviations.len() / 3]);
        let last_third = mean_f64(&deviations[deviations.len() * 2 / 3..]);
        let diff = last_third - first_third;
        if diff.abs() > 50.0 {
            format!(
                "Timing shape detected across passage (shift of {:.0}ms). Model score {:.2}.",
                diff, model_score,
            )
        } else {
            format!(
                "Timing relatively even across passage. Model score {:.2}.",
                model_score,
            )
        }
    } else {
        format!("Passage too short for phrasing analysis. Model score {:.2}.", model_score)
    };

    DimensionAnalysis {
        dimension: "phrasing".into(),
        analysis,
        score_marking: None,
        reference_comparison: None,
    }
}

fn analyze_interpretation(
    alignments: &[NoteAlignment],
    _score_bars: &[&ScoreBar],
    _ref_bars: &[&ReferenceBar],
    model_score: f64,
) -> DimensionAnalysis {
    // Interpretation: aggregate signal. Summarize overall deviation from score.
    let total_dev: f64 = alignments.iter()
        .map(|a| a.onset_deviation_ms.abs())
        .sum::<f64>() / alignments.len().max(1) as f64;

    let analysis = format!(
        "Average absolute deviation {:.0}ms from score. Model score {:.2}.",
        total_dev, model_score,
    );

    DimensionAnalysis {
        dimension: "interpretation".into(),
        analysis,
        score_marking: None,
        reference_comparison: None,
    }
}

// --- Tier 2 analyzers (absolute, no score) ---

fn analyze_dynamics_absolute(perf_notes: &[PerfNote], model_score: f64) -> DimensionAnalysis {
    let velocities: Vec<f64> = perf_notes.iter().map(|n| n.velocity as f64).collect();
    let mean_vel = mean_f64(&velocities);
    let range = velocities.iter().cloned().fold(f64::MAX, f64::min)
        .max(0.0);
    let max_vel = velocities.iter().cloned().fold(0.0f64, f64::max);

    DimensionAnalysis {
        dimension: "dynamics".into(),
        analysis: format!(
            "Velocity range {:.0}-{:.0} (avg {:.0}). Dynamic range {:.0}. Model score {:.2}.",
            range, max_vel, mean_vel, max_vel - range, model_score,
        ),
        score_marking: None,
        reference_comparison: None,
    }
}

fn analyze_timing_absolute(perf_notes: &[PerfNote], model_score: f64) -> DimensionAnalysis {
    // Inter-onset intervals
    let iois: Vec<f64> = perf_notes.windows(2)
        .map(|w| w[1].onset - w[0].onset)
        .collect();
    let ioi_std = std_f64(&iois);

    DimensionAnalysis {
        dimension: "timing".into(),
        analysis: format!(
            "Inter-onset interval std {:.0}ms (lower = more regular). Model score {:.2}.",
            ioi_std * 1000.0, model_score,
        ),
        score_marking: None,
        reference_comparison: None,
    }
}

fn analyze_pedaling_absolute(perf_pedal: &[PerfPedalEvent], model_score: f64) -> DimensionAnalysis {
    let changes = perf_pedal.len();
    DimensionAnalysis {
        dimension: "pedaling".into(),
        analysis: format!(
            "{} pedal events detected. Model score {:.2}.",
            changes, model_score,
        ),
        score_marking: None,
        reference_comparison: None,
    }
}

fn analyze_articulation_absolute(perf_notes: &[PerfNote], model_score: f64) -> DimensionAnalysis {
    let durations: Vec<f64> = perf_notes.iter().map(|n| n.offset - n.onset).collect();
    let mean_dur = mean_f64(&durations);
    DimensionAnalysis {
        dimension: "articulation".into(),
        analysis: format!(
            "Average note duration {:.2}s. Model score {:.2}.",
            mean_dur, model_score,
        ),
        score_marking: None,
        reference_comparison: None,
    }
}

fn analyze_phrasing_absolute(perf_notes: &[PerfNote], model_score: f64) -> DimensionAnalysis {
    DimensionAnalysis {
        dimension: "phrasing".into(),
        analysis: format!(
            "{} notes in passage. Model score {:.2}.",
            perf_notes.len(), model_score,
        ),
        score_marking: None,
        reference_comparison: None,
    }
}

fn analyze_interpretation_absolute(model_score: f64) -> DimensionAnalysis {
    DimensionAnalysis {
        dimension: "interpretation".into(),
        analysis: format!("Model score {:.2}.", model_score),
        score_marking: None,
        reference_comparison: None,
    }
}

// --- Utility functions ---

fn mean_u8(vals: &[u8]) -> f64 {
    if vals.is_empty() { return 0.0; }
    vals.iter().map(|v| *v as f64).sum::<f64>() / vals.len() as f64
}

fn mean_f64(vals: &[f64]) -> f64 {
    if vals.is_empty() { return 0.0; }
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn std_f64(vals: &[f64]) -> f64 {
    if vals.len() < 2 { return 0.0; }
    let m = mean_f64(vals);
    let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
    var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::practice::score_context::*;
    use crate::practice::score_follower::*;

    #[test]
    fn tier2_produces_all_6_dimensions() {
        let perf = vec![
            PerfNote { pitch: 60, onset: 0.0, offset: 0.3, velocity: 80 },
            PerfNote { pitch: 62, onset: 0.5, offset: 0.8, velocity: 90 },
        ];
        let pedal = vec![];
        let scores = [0.5, 0.6, 0.4, 0.7, 0.5, 0.6];

        let result = analyze_tier2(&perf, &pedal, &scores);
        assert_eq!(result.tier, 2);
        assert_eq!(result.dimensions.len(), 6);
        assert!(result.bar_range.is_none());
    }

    #[test]
    fn tier2_includes_model_scores() {
        let perf = vec![
            PerfNote { pitch: 60, onset: 0.0, offset: 0.3, velocity: 80 },
            PerfNote { pitch: 62, onset: 0.5, offset: 0.8, velocity: 90 },
            PerfNote { pitch: 64, onset: 1.0, offset: 1.3, velocity: 85 },
        ];
        let scores = [0.35, 0.6, 0.4, 0.7, 0.5, 0.6];

        let result = analyze_tier2(&perf, &[], &scores);
        // Each dimension analysis should mention its model score
        for dim in &result.dimensions {
            assert!(dim.analysis.contains("Model score"), "dim {} missing model score", dim.dimension);
        }
    }

    #[test]
    fn timing_detects_rushing() {
        let alignments = vec![
            NoteAlignment { perf_onset: 0.0, perf_pitch: 60, perf_velocity: 80, score_bar: 1, score_beat: 0.0, score_pitch: 60, onset_deviation_ms: -50.0 },
            NoteAlignment { perf_onset: 0.5, perf_pitch: 62, perf_velocity: 80, score_bar: 1, score_beat: 1.0, score_pitch: 62, onset_deviation_ms: -40.0 },
            NoteAlignment { perf_onset: 1.0, perf_pitch: 64, perf_velocity: 80, score_bar: 1, score_beat: 2.0, score_pitch: 64, onset_deviation_ms: -35.0 },
        ];

        let result = analyze_timing(&alignments, &[], &[], 0.5);
        assert!(result.analysis.contains("rushing"), "should detect rushing: {}", result.analysis);
    }

    #[test]
    fn pedaling_detects_no_pedal_vs_score() {
        let score_bar = ScoreBar {
            bar_number: 1,
            start_tick: 0, start_seconds: 0.0,
            time_signature: "4/4".into(),
            notes: vec![],
            pedal_events: vec![
                ScorePedalEvent { event_type: "on".into(), tick: 0, seconds: 0.0 },
                ScorePedalEvent { event_type: "off".into(), tick: 480, seconds: 0.5 },
            ],
            note_count: 0, pitch_range: vec![], mean_velocity: 0,
        };

        let result = analyze_pedaling(&[], &[&score_bar], &[], 0.35);
        assert!(result.analysis.contains("No pedal detected"), "{}", result.analysis);
        assert!(result.analysis.contains("2 pedal markings"), "{}", result.analysis);
    }
}
```

- [ ] **Step 2: Register module**

Add to `apps/api/src/practice/mod.rs`:

```rust
pub mod analysis;
```

- [ ] **Step 3: Run tests**

Run: `cd apps/api && cargo test analysis -- --nocapture`
Expected: All 4 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/practice/analysis.rs apps/api/src/practice/mod.rs
git commit -m "feat: add bar-aligned musical analysis engine (all 6 dimensions, Tier 1+2)"
```

---

## Chunk 6: Session Integration + Prompt Enrichment

### Task 7: Wire Score Context + Follower + Analysis into PracticeSession DO

This is the integration task: modify `session.rs` to parse the full HF response (midi_notes + pedal_events), load score context on first chunk, run score following per chunk, run analysis, and pass enriched piece_context to `handle_ask_inner`.

> **IMPORTANT:** Steps 1-6 of this task must be applied atomically (single commit). Step 3 changes the `call_hf_inference` return type, which breaks callers until Step 4 replaces `handle_chunk_ready`. Do not pause between these steps.
>
> **Prerequisite:** Tasks 3-6 must be complete and merged before starting this task (modules must be registered in `mod.rs`).

**Files:**
- Modify: `apps/api/src/practice/session.rs`

- [ ] **Step 1: Add new fields to SessionState**

In `session.rs`, add to the `SessionState` struct (line 29):

```rust
use crate::practice::score_context::ScoreContext;
use crate::practice::score_follower::{FollowerState, PerfNote, PerfPedalEvent};
use crate::practice::analysis;
```

Add fields to `SessionState`:

```rust
struct SessionState {
    session_id: String,
    student_id: String,
    piece_query: Option<String>,          // NEW: student's piece name
    score_context: Option<ScoreContext>,   // NEW: loaded once per session
    score_context_loaded: bool,            // NEW: prevent repeated lookups
    follower_state: FollowerState,         // NEW: DTW continuity
    baselines: Option<StudentBaselines>,
    baselines_loaded: bool,
    scored_chunks: Vec<ScoredChunk>,
    observations: Vec<ObservationRecord>,
    dim_stats: DimStats,
    last_observation_at: Option<u64>,
}
```

Update `Default` impl accordingly with `piece_query: None, score_context: None, score_context_loaded: false, follower_state: FollowerState::default()`.

- [ ] **Step 2: Add `set_piece` WebSocket message handler**

In the `websocket_message` match block (after line 128), add:

```rust
"set_piece" => {
    let query = parsed.get("query").and_then(|v| v.as_str()).unwrap_or("");
    if !query.is_empty() {
        let mut s = self.inner.borrow_mut();
        s.piece_query = Some(query.to_string());
        // Reset score context so it's loaded on next chunk
        s.score_context = None;
        s.score_context_loaded = false;
        s.follower_state = FollowerState::default();
    }
    // Acknowledge
    let ack = serde_json::json!({"type": "piece_set", "query": query});
    let _ = ws.send_with_str(&ack.to_string());
}
```

- [ ] **Step 3: Update call_hf_inference to return full response**

Change `call_hf_inference` to return the full HF response body (not just scores):

```rust
async fn call_hf_inference(
    &self,
    audio_bytes: &[u8],
) -> std::result::Result<serde_json::Value, String> {
    // ... same HTTP call ...
    // Return full body instead of extracting just predictions
    let body: serde_json::Value = response.json().await
        .map_err(|e| format!("HF response parse failed: {:?}", e))?;
    Ok(body)
}
```

- [ ] **Step 4: Update handle_chunk_ready to use full pipeline**

Replace `handle_chunk_ready` to extract scores, midi_notes, and pedal_events; load score context once; run score following and analysis:

```rust
async fn handle_chunk_ready(&self, ws: &WebSocket, index: usize, r2_key: &str) -> Result<()> {
    // 1. Fetch audio from R2
    let audio_bytes = match self.fetch_audio_from_r2(r2_key).await {
        Ok(bytes) => bytes,
        Err(e) => {
            console_error!("R2 fetch failed for {}: {}", r2_key, e);
            self.send_zeroed_chunk_processed(ws, index)?;
            return Ok(());
        }
    };

    // 2. Call HF inference (returns full response with scores + AMT)
    let hf_response = match self.call_hf_inference(&audio_bytes).await {
        Ok(resp) => resp,
        Err(e) => {
            console_error!("HF inference failed for chunk {}: {}", index, e);
            self.send_zeroed_chunk_processed(ws, index)?;
            return Ok(());
        }
    };

    // 3. Extract scores
    let predictions = hf_response.get("predictions").unwrap_or(&hf_response);
    let mut scores_map = std::collections::HashMap::new();
    for dim in DIMS_6 {
        if let Some(val) = predictions.get(dim).and_then(|v| v.as_f64()) {
            scores_map.insert(dim.to_string(), val);
        }
    }
    let scores_array: [f64; 6] = DIMS_6.map(|dim| {
        scores_map.get(dim).copied().unwrap_or(0.0)
    });

    // 4. Extract AMT notes + pedal events
    let perf_notes: Vec<PerfNote> = hf_response.get("midi_notes")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|n| {
            Some(PerfNote {
                pitch: n.get("pitch")?.as_u64()? as u8,
                onset: n.get("onset")?.as_f64()?,
                offset: n.get("offset")?.as_f64()?,
                velocity: n.get("velocity")?.as_u64()? as u8,
            })
        }).collect())
        .unwrap_or_default();

    let perf_pedal: Vec<PerfPedalEvent> = hf_response.get("pedal_events")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|p| {
            Some(PerfPedalEvent {
                time: p.get("time")?.as_f64()?,
                value: p.get("value")?.as_u64()? as u8,
            })
        }).collect())
        .unwrap_or_default();

    // 5. Send chunk_processed immediately (same as before)
    let scores_json = serde_json::json!({
        "dynamics": scores_array[0], "timing": scores_array[1],
        "pedaling": scores_array[2], "articulation": scores_array[3],
        "phrasing": scores_array[4], "interpretation": scores_array[5],
    });
    let response = serde_json::json!({
        "type": "chunk_processed",
        "index": index,
        "scores": scores_json,
    });
    ws.send_with_str(&response.to_string())?;

    // 6. Load score context (one-time per session)
    {
        let needs_load = {
            let s = self.inner.borrow();
            !s.score_context_loaded && s.piece_query.is_some()
        };
        if needs_load {
            let (query, student_id) = {
                let s = self.inner.borrow();
                (s.piece_query.clone().unwrap_or_default(), s.student_id.clone())
            };
            let ctx = crate::practice::score_context::resolve_piece(
                &self.env, &query, &student_id,
            ).await;
            let mut s = self.inner.borrow_mut();
            s.score_context = ctx;
            s.score_context_loaded = true;
        }
    }

    // 7. Run score following + analysis
    let chunk_analysis = {
        let mut s = self.inner.borrow_mut();
        let chunk_offset = index as f64 * 15.0;

        match &s.score_context {
            Some(ctx) if !perf_notes.is_empty() => {
                // Tier 1: full bar-aligned analysis
                let bar_map = crate::practice::score_follower::align_chunk(
                    index, chunk_offset, &perf_notes, &ctx.score, &mut s.follower_state,
                );
                match bar_map {
                    Some(bm) => Some(analysis::analyze_tier1(
                        &bm, &perf_notes, &perf_pedal, &scores_array, ctx,
                    )),
                    None => Some(analysis::analyze_tier2(&perf_notes, &perf_pedal, &scores_array)),
                }
            }
            _ if !perf_notes.is_empty() => {
                // Tier 2: absolute analysis (no score context)
                Some(analysis::analyze_tier2(&perf_notes, &perf_pedal, &scores_array))
            }
            _ => None, // Tier 3: scores only (AMT failed)
        }
    };

    // 8-9. Update stats + store chunk (same as before)
    {
        let mut s = self.inner.borrow_mut();
        s.dim_stats.update(&scores_map);
        s.scored_chunks.push(ScoredChunk {
            chunk_index: index,
            scores: scores_array,
        });
    }

    // 10. Load baselines (one-time)
    {
        let needs_load = !self.inner.borrow().baselines_loaded;
        if needs_load {
            let student_id = self.inner.borrow().student_id.clone();
            let baselines = self.load_baselines(&student_id).await;
            let mut s = self.inner.borrow_mut();
            s.baselines = Some(baselines);
            s.baselines_loaded = true;
        }
    }

    // 11. STOP + generate observation (enriched with analysis)
    let stop_result = stop::classify(&scores_array);
    let should_generate = {
        let s = self.inner.borrow();
        stop_result.triggered && s.baselines.is_some() && self.throttle_allows(&s)
    };

    if should_generate {
        self.generate_observation(ws, chunk_analysis.as_ref()).await;
    }

    let _ = self.state.storage().set_alarm(ALARM_DURATION_MS).await;
    Ok(())
}
```

- [ ] **Step 5: Update generate_observation to pass enriched piece_context**

Modify `generate_observation` (line 277) to accept and use the chunk analysis:

```rust
async fn generate_observation(&self, ws: &WebSocket, chunk_analysis: Option<&analysis::ChunkAnalysis>) {
    // ... existing code to get scored_chunks, baselines, recent_obs ...

    let moment = match crate::services::teaching_moments::select_teaching_moment(
        &scored_chunks, &baselines, &recent_obs,
    ) {
        Some(m) => m,
        None => return,
    };

    // Build enriched piece_context with musical analysis
    let piece_context = {
        let s = self.inner.borrow();
        let mut ctx = serde_json::Map::new();

        if let Some(score_ctx) = &s.score_context {
            ctx.insert("composer".into(), serde_json::json!(score_ctx.composer));
            ctx.insert("title".into(), serde_json::json!(score_ctx.title));
            ctx.insert("piece_id".into(), serde_json::json!(score_ctx.piece_id));
        }

        if let Some(analysis) = chunk_analysis {
            if let Some(bar_range) = &analysis.bar_range {
                ctx.insert("bar_range".into(), serde_json::json!(bar_range));
            }
            ctx.insert("analysis_tier".into(), serde_json::json!(analysis.tier));
            ctx.insert("musical_analysis".into(), serde_json::to_value(&analysis.dimensions).unwrap_or_default());
        }

        if ctx.is_empty() { None } else { Some(serde_json::Value::Object(ctx)) }
    };

    // Extract bar_range for the WS event
    let bar_range_str = chunk_analysis.as_ref()
        .and_then(|a| a.bar_range.clone());

    let tm_json = serde_json::json!({
        "dimension": moment.dimension,
        "dimension_score": moment.score,
        "chunk_index": moment.chunk_index,
        "deviation": moment.deviation,
        "stop_probability": moment.stop_probability,
        "is_positive": moment.is_positive,
        "reasoning": moment.reasoning,
    });

    let inner_req = crate::services::ask::AskInnerRequest {
        teaching_moment: tm_json,
        student_id: student_id.clone(),
        session_id: session_id.clone(),
        piece_context,  // NOW POPULATED
    };

    let inner_resp = crate::services::ask::handle_ask_inner(&self.env, &inner_req).await;

    // Push observation to client (include barRange for UI display)
    let mut obs_event = serde_json::json!({
        "type": "observation",
        "text": inner_resp.observation_text,
        "dimension": inner_resp.dimension,
        "framing": inner_resp.framing,
    });
    if let Some(br) = &bar_range_str {
        obs_event["barRange"] = serde_json::json!(br);
    }
    let _ = ws.send_with_str(&obs_event.to_string());

    // Store in session state (rest unchanged) ...
}
```

- [ ] **Step 6: Verify compilation**

Run: `cd apps/api && cargo check`
Expected: Compiles.

- [ ] **Step 7: Commit**

```bash
git add apps/api/src/practice/session.rs
git commit -m "feat: wire score following + analysis engine into PracticeSession DO pipeline"
```

### Task 8: Enrich Subagent Prompt with Musical Analysis

**Files:**
- Modify: `apps/api/src/services/prompts.rs:122-138`

- [ ] **Step 1: Expand piece_context block in subagent prompt**

Replace the piece_context block in `build_subagent_user_prompt` (lines 122-138) to include musical analysis:

```rust
// Piece context + musical analysis
if let Some(piece) = piece_context {
    prompt.push_str("<piece_context>\n");
    if let Some(composer) = piece.get("composer").and_then(|v| v.as_str()) {
        prompt.push_str(&format!("Composer: {}\n", composer));
    }
    if let Some(title) = piece.get("title").and_then(|v| v.as_str()) {
        prompt.push_str(&format!("Title: {}\n", title));
    }
    if let Some(bar_range) = piece.get("bar_range").and_then(|v| v.as_str()) {
        prompt.push_str(&format!("Bar range: {}\n", bar_range));
    }
    if let Some(tier) = piece.get("analysis_tier").and_then(|v| v.as_u64()) {
        prompt.push_str(&format!("Analysis tier: {} (1=full score context, 2=absolute, 3=scores only)\n", tier));
    }

    // Per-dimension musical analysis
    if let Some(analysis) = piece.get("musical_analysis").and_then(|v| v.as_array()) {
        prompt.push_str("\n<musical_analysis>\n");
        for dim_analysis in analysis {
            if let Some(dim) = dim_analysis.get("dimension").and_then(|v| v.as_str()) {
                prompt.push_str(&format!("<{}>\n", dim));
                if let Some(a) = dim_analysis.get("analysis").and_then(|v| v.as_str()) {
                    prompt.push_str(&format!("  {}\n", a));
                }
                if let Some(sm) = dim_analysis.get("score_marking").and_then(|v| v.as_str()) {
                    prompt.push_str(&format!("  Score marking: {}\n", sm));
                }
                if let Some(rc) = dim_analysis.get("reference_comparison").and_then(|v| v.as_str()) {
                    prompt.push_str(&format!("  Reference: {}\n", rc));
                }
                prompt.push_str(&format!("</{}>\n", dim));
            }
        }
        prompt.push_str("</musical_analysis>\n");
    }
    prompt.push_str("</piece_context>\n\n");
}
```

- [ ] **Step 2: Verify compilation**

Run: `cd apps/api && cargo check`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/services/prompts.rs
git commit -m "feat: enrich subagent prompt with bar-aligned musical analysis"
```

---

## Chunk 7: Reference Performance Cache (Offline)

### Task 9: Build Reference Cache Generation Script

Offline Python job: for each ASAP piece that also has MAESTRO recordings, align professional performances to the score and compute per-bar statistics.

**Files:**
- Create: `model/src/score_library/reference_cache.py`

- [ ] **Step 1: Write reference cache generator**

```python
"""Generate reference performance profiles from MAESTRO recordings.

For each piece in the score library that has MAESTRO recordings:
1. Load the score JSON
2. For each MAESTRO recording of that piece:
   a. Load the performance MIDI
   b. Align performance to score via onset-based DTW
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

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import mido
from dtw import dtw  # pip install dtw-python


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


def load_score(path: Path) -> dict:
    """Load a score JSON file."""
    with open(path) as f:
        return json.load(f)


def load_performance_midi(midi_path: Path) -> list[dict]:
    """Load a MIDI file and extract note events with onset times in seconds."""
    mid = mido.MidiFile(str(midi_path))
    notes = []
    abs_time = 0.0
    tempo = 500000  # default 120 BPM

    for track in mid.tracks:
        abs_time = 0.0
        active = {}
        for msg in track:
            abs_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = {"pitch": msg.note, "onset": abs_time, "velocity": msg.velocity}
            elif msg.type in ("note_off", "note_on") and (msg.type == "note_off" or msg.velocity == 0):
                if msg.note in active:
                    note = active.pop(msg.note)
                    note["offset"] = abs_time
                    note["duration"] = abs_time - note["onset"]
                    notes.append(note)

    notes.sort(key=lambda n: n["onset"])
    return notes


def align_to_score(perf_notes: list[dict], score_data: dict) -> dict[int, list[dict]]:
    """Align performance notes to score bars using onset-based DTW.

    Returns dict mapping bar_number -> list of aligned performance notes.
    """
    # Flatten score notes with bar assignments
    score_onsets = []
    score_bars_map = []
    for bar in score_data["bars"]:
        for note in bar["notes"]:
            score_onsets.append(note["onset_seconds"])
            score_bars_map.append(bar["bar_number"])

    if not score_onsets or not perf_notes:
        return {}

    perf_onsets = [n["onset"] for n in perf_notes]

    # DTW alignment
    score_arr = np.array(score_onsets).reshape(-1, 1)
    perf_arr = np.array(perf_onsets).reshape(-1, 1)

    alignment = dtw(perf_arr, score_arr)

    # Map performance notes to bars
    bar_notes: dict[int, list[dict]] = {}
    for pi, si in zip(alignment.index1, alignment.index2):
        bar_num = score_bars_map[si]
        note = perf_notes[pi].copy()
        note["score_onset"] = score_onsets[si]
        note["onset_deviation_ms"] = (note["onset"] - score_onsets[si]) * 1000
        if bar_num not in bar_notes:
            bar_notes[bar_num] = []
        bar_notes[bar_num].append(note)

    return bar_notes


def compute_bar_stats(bar_num: int, notes: list[dict], score_bar: dict) -> BarStats:
    """Compute statistics for one bar from one performance."""
    velocities = [n["velocity"] for n in notes]
    deviations = [n["onset_deviation_ms"] for n in notes]

    # Duration ratio: performance / score
    score_durations = [n["duration_seconds"] for n in score_bar.get("notes", [])]
    perf_durations = [n.get("duration", 0.25) for n in notes]
    if score_durations and perf_durations:
        ratio = np.mean(perf_durations) / max(np.mean(score_durations), 0.01)
    else:
        ratio = 1.0

    return BarStats(
        bar_number=bar_num,
        velocity_mean=float(np.mean(velocities)) if velocities else 0.0,
        velocity_std=float(np.std(velocities)) if len(velocities) > 1 else 0.0,
        onset_deviation_mean_ms=float(np.mean(deviations)) if deviations else 0.0,
        onset_deviation_std_ms=float(np.std(deviations)) if len(deviations) > 1 else 0.0,
        note_duration_ratio_mean=float(ratio),
        performer_count=1,
    )


def aggregate_bar_stats(all_stats: list[list[BarStats]]) -> list[BarStats]:
    """Aggregate per-bar stats across multiple performers."""
    bar_map: dict[int, list[BarStats]] = {}
    for performer_bars in all_stats:
        for bs in performer_bars:
            bar_map.setdefault(bs.bar_number, []).append(bs)

    aggregated = []
    for bar_num in sorted(bar_map.keys()):
        stats_list = bar_map[bar_num]
        n = len(stats_list)
        aggregated.append(BarStats(
            bar_number=bar_num,
            velocity_mean=float(np.mean([s.velocity_mean for s in stats_list])),
            velocity_std=float(np.mean([s.velocity_std for s in stats_list])),
            onset_deviation_mean_ms=float(np.mean([s.onset_deviation_mean_ms for s in stats_list])),
            onset_deviation_std_ms=float(np.mean([s.onset_deviation_std_ms for s in stats_list])),
            note_duration_ratio_mean=float(np.mean([s.note_duration_ratio_mean for s in stats_list])),
            performer_count=n,
        ))

    return aggregated


def build_reference_for_piece(
    piece_id: str,
    score_path: Path,
    midi_paths: list[Path],
) -> ReferenceProfile | None:
    """Build a reference profile for one piece from multiple MAESTRO recordings."""
    score_data = load_score(score_path)
    score_bars_by_num = {b["bar_number"]: b for b in score_data["bars"]}

    all_performer_stats = []
    for midi_path in midi_paths:
        perf_notes = load_performance_midi(midi_path)
        if not perf_notes:
            continue

        bar_notes = align_to_score(perf_notes, score_data)
        performer_bars = []
        for bar_num, notes in bar_notes.items():
            score_bar = score_bars_by_num.get(bar_num, {})
            performer_bars.append(compute_bar_stats(bar_num, notes, score_bar))

        if performer_bars:
            all_performer_stats.append(performer_bars)

    if not all_performer_stats:
        return None

    bars = aggregate_bar_stats(all_performer_stats)

    return ReferenceProfile(
        piece_id=piece_id,
        performer_count=len(all_performer_stats),
        bars=bars,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate reference performance profiles")
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--maestro-dir", type=Path, required=True)
    parser.add_argument("--maestro-meta", type=Path, help="Path to maestro-v3.0.0.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load MAESTRO metadata to map canonical_title -> MIDI paths
    if args.maestro_meta and args.maestro_meta.exists():
        with open(args.maestro_meta) as f:
            maestro = json.load(f)
    else:
        print("MAESTRO metadata not provided; skipping reference generation")
        return

    # Build piece_id -> MAESTRO MIDI paths mapping
    # This requires matching ASAP piece_ids to MAESTRO canonical_titles
    # (dataset-specific mapping -- implement per ASAP/MAESTRO naming conventions)
    print(f"Score dir: {args.score_dir}")
    print(f"MAESTRO dir: {args.maestro_dir}")
    print(f"Output dir: {args.output_dir}")
    print("Reference cache generation requires piece_id <-> MAESTRO title mapping.")
    print("Implement dataset-specific matching for your catalog.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add dtw-python dependency**

Run: `cd model && uv add dtw-python`

- [ ] **Step 3: Commit**

```bash
git add model/src/score_library/reference_cache.py model/pyproject.toml
git commit -m "feat: add reference performance cache generation script (MAESTRO)"
```

---

## Chunk 8: Web Client Integration

### Task 10: Add set_piece WebSocket Message + Type Updates to Web Client

The web client needs to send a `set_piece` message over the WebSocket when the student identifies their piece (via chat or UI). This enables score context loading in the DO.

**Files:**
- Modify: `apps/web/src/lib/practice-api.ts` (add `piece_set` to WS event union, add `barRange` to observation event)
- Modify: `apps/web/src/hooks/usePracticeSession.ts` (add `setPiece` method, handle `piece_set` ack)

- [ ] **Step 1: Update TypeScript types in practice-api.ts**

Add `piece_set` to the `PracticeWsEvent` union type, and add `barRange` to the observation event:

```typescript
// In PracticeWsEvent union, add:
| { type: "piece_set"; query: string }

// Update the observation event to include barRange:
| { type: "observation"; text: string; dimension: string; framing: string; barRange?: string }
```

- [ ] **Step 2: Add setPiece method to practice session hook**

In the WebSocket hook, add a method that sends the `set_piece` message:

```typescript
// Add to the practice session hook's returned API
const setPiece = useCallback((query: string) => {
  if (wsRef.current?.readyState === WebSocket.OPEN) {
    wsRef.current.send(JSON.stringify({
      type: "set_piece",
      query,
    }));
  }
}, []);
```

Add `setPiece` to both the `UsePracticeSessionReturn` interface and the hook's return statement.

- [ ] **Step 3: Handle piece_set acknowledgment**

In the WebSocket message handler, add:

```typescript
case "piece_set":
  console.log("Piece context set:", data.query);
  break;
```

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/lib/practice-api.ts apps/web/src/hooks/usePracticeSession.ts
git commit -m "feat: add setPiece WebSocket message and bar-range types to web client"
```

---

## Dependency Graph

```
Task 1 (AMT pedal)          Task 2 (D1 migration)    Task 3 (fuzzy match)
     |                            |                        |
     v                            v                        v
Task 5 (score follower) <-- Task 4 (score context) <------/
     |                            |
     v                            v
Task 6 (analysis engine)         /
     |                          /
     v                         v
Task 7 (session integration + prompt enrichment)
     |
     v
Task 8 (subagent prompt)    Task 9 (reference cache, parallel)
     |
     v
Task 10 (web client)
```

Tasks 1, 2, 3 can run in parallel. Task 9 can run in parallel with 7-8.
