# Phase 1a: Score MIDI Library

> **Status:** DESIGNED (not implemented)
> **Depends on:** Nothing (independent foundation)
> **Enables:** 1c (Score Following), 1d (Bar-Aligned Analysis Engine), 1e (Reference Performance Cache)
> **Pipeline phase:** Phase 1 -- Score Infrastructure

---

## Purpose

Build a searchable library of parsed score MIDIs that downstream pipeline components (score following, bar-aligned analysis, reference comparison) can query by piece ID. This is the foundation for transforming teacher feedback from "dynamics score 0.35" to "the crescendo in bars 12-16 doesn't reach the forte Chopin marked."

80% of user-facing improvement comes from giving the LLM better context (bar-aligned musical facts, reference comparisons), not from changing the model. The Score MIDI Library is the first step: structured, bar-indexed score data that the subagent can reason about.

## Scope

**In scope:**
- Parse ASAP dataset score MIDIs (242 unique pieces, 16 composers)
- Extract bar structure, time/key signatures, tempo markings, notes per bar, pedal events
- Store piece catalog in D1 (searchable index) and full per-piece score data in R2 (JSON)
- API endpoints for piece lookup and score data retrieval
- Graceful degradation: pieces not in the library fall back to absolute scoring (current behavior)

**Out of scope (with future notes):**
- MAESTRO/IMSLP/MuseScore piece expansion (future: add sources as library grows)
- MusicXML import for richer annotations (dynamics text like pp/ff/cresc., articulation marks, section labels) -- MIDI doesn't encode these reliably; MusicXML is the right future path
- Piece identification/matching UX -- future options: AMT-based pitch sequence matching or audio fingerprinting (Shazam-style). The library provides a clean `piece_id -> score_data` lookup API; how the student selects a piece is a product/UX concern for the app layer
- Score-conditioned model changes (Phase 3)

## Data Source: ASAP Dataset

The ASAP dataset in `model/data/asap_cache/` contains:
- **1,066 score MIDI files** across **242 unique pieces** and **16 composers**
- Directory structure: `{Composer}/{Collection}/{Piece}/score_{performer}.mid`
- Score files for the same piece are identical in note content across performers (verified via MD5)
- One canonical score per piece: pick the first `score_*.mid` in each piece directory

Composers: Bach, Balakirev, Beethoven, Brahms, Chopin, Debussy, Glinka, Haydn, Liszt, Mozart, Prokofiev, Rachmaninoff, Ravel, Schubert, Schumann, Scriabin.

MIDI files contain: time signatures, key signatures, tempo markings, note data (pitch, velocity, onset, duration) across 2 tracks (RH/LH), pedal CC events. No annotation files (beat positions, etc.) are present in the current cache -- bar structure is derived from time signature events in the MIDI.

## Data Model

### Piece Catalog (D1 table: `pieces`)

Searchable index for piece discovery. One row per piece.

```sql
CREATE TABLE pieces (
  piece_id TEXT PRIMARY KEY,
  composer TEXT NOT NULL,
  title TEXT NOT NULL,
  key_signature TEXT,
  time_signature TEXT,
  tempo_bpm INTEGER,
  bar_count INTEGER NOT NULL,
  duration_seconds REAL,
  note_count INTEGER NOT NULL,
  pitch_range_low INTEGER,
  pitch_range_high INTEGER,
  source TEXT NOT NULL DEFAULT 'asap',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

- `piece_id` derived from ASAP directory path, normalized to lowercase: `chopin/etudes_op_10/3`
- `time_signature` and `tempo_bpm` reflect the initial values; pieces with changes are noted in the full score data
- `source` tracks provenance for when additional sources are added

### Per-Piece Score Data (R2: `scores/{piece_id}.json`)

Full parsed score, bar-centric structure. This is what downstream consumers (score following, bar-aligned analysis engine) use.

```json
{
  "piece_id": "chopin/etudes_op_10/3",
  "composer": "Chopin",
  "title": "Etude Op. 10 No. 3",
  "key_signature": "E",
  "time_signatures": [
    {"bar": 1, "numerator": 2, "denominator": 4}
  ],
  "tempo_markings": [
    {"bar": 1, "bpm": 100}
  ],
  "total_bars": 77,
  "bars": [
    {
      "bar_number": 1,
      "start_tick": 0,
      "start_seconds": 0.0,
      "time_signature": "2/4",
      "notes": [
        {
          "pitch": 64,
          "pitch_name": "E4",
          "velocity": 49,
          "onset_tick": 0,
          "onset_seconds": 0.0,
          "duration_ticks": 240,
          "duration_seconds": 0.42,
          "track": 0
        }
      ],
      "pedal_events": [],
      "note_count": 8,
      "pitch_range": [40, 64],
      "mean_velocity": 49
    }
  ]
}
```

Design choices:
- **Bar-centric structure.** Everything indexed by bar number -- the natural unit for score following and teacher feedback ("bars 12-16").
- **Both ticks and seconds.** Ticks for score following alignment, seconds for mapping to audio chunks.
- **Per-bar summary stats** (note_count, pitch_range, mean_velocity) pre-computed for the analysis engine.
- **Track preserved.** ASAP scores use 2 tracks (RH/LH). Enables future hand-separation analysis.

## Pipeline Architecture

Batch Python CLI tool in `model/src/score_library/`. Three stages, stateless -- always regenerates from source.

```
STAGE 1: DISCOVER          STAGE 2: PARSE              STAGE 3: UPLOAD

Scan ASAP dirs       -->   For each piece:        -->   Seed D1 pieces table
Pick one score MIDI        Parse MIDI (mido)            Upload JSONs to R2
per unique piece           Extract bar structure
Generate piece_id          Extract notes per bar
                           Compute summary stats
                           Write {piece_id}.json
```

### Stage 1: Discovery

- Walk `asap_cache/{Composer}/{Collection}/{Piece}/`
- For each unique piece directory, pick the first `score_*.mid` file
- Derive `piece_id` from path: lowercase, forward-slash separated (e.g., `chopin/etudes_op_10/3`)
- Derive human-readable `title` from directory names (e.g., `Etude Op. 10 No. 3`)
- Output: list of `(piece_id, title, composer, score_midi_path)`

### Stage 2: Parse

For each piece, using `mido`:
1. Read time signatures, key signatures, tempo changes from track 0
2. Build a bar grid from time signature events (tick positions of each bar boundary)
3. Assign each note to a bar by onset tick
4. Convert ticks to seconds using tempo map
5. Extract pedal events (CC64) and assign to bars
6. Compute per-bar summaries (note_count, pitch_range, mean_velocity)
7. Write structured JSON to `model/data/score_library/{piece_id}.json`

Error handling: if a MIDI file fails to parse, log the error and skip. Report summary at the end. Do not halt the batch for individual failures.

### Stage 3: Upload

- Generate D1 seed SQL from parsed metadata
- Upload per-piece JSONs to R2 bucket at `scores/{piece_id}.json`
- Verify upload count matches parsed count

### CLI Interface

```bash
# Full pipeline: discover, parse, upload
uv run python -m score_library.cli build --asap-dir data/asap_cache

# Parse only (local JSON output, no upload)
uv run python -m score_library.cli parse --asap-dir data/asap_cache

# Upload only (from previously parsed local JSONs)
uv run python -m score_library.cli upload --source data/score_library

# Stats: print piece count, composer distribution, bar count range
uv run python -m score_library.cli stats --source data/score_library
```

### Dependencies

- `mido` -- already in the project for MIDI parsing
- `wrangler` -- for D1 seeding and R2 upload (already configured in `apps/api/wrangler.toml`)
- No new Python dependencies needed

## API Worker Integration

### Endpoints

**Piece lookup (D1):**
```
GET /api/scores/:piece_id
  -> 200: { piece_id, composer, title, key_signature, bar_count, ... }
  -> 404: { error: "piece_not_found" }

GET /api/scores?composer=Chopin
  -> 200: [{ piece_id, composer, title, ... }, ...]
```

**Full score data (R2):**
```
GET /api/scores/:piece_id/data
  -> 200: full JSON from R2 (bar-centric score data)
  -> 404: { error: "piece_not_found" }
```

The worker fetches from R2 and caches using the Cloudflare Cache API. Score data is immutable -- cache indefinitely until a library rebuild.

### Graceful Degradation

When `piece_id` is not found in D1:
- Score following, bar-aligned analysis, and reference comparison are skipped
- System falls back to current behavior: absolute scoring with raw dimension scores fed to the subagent
- Track "piece not found" events via `console_error!` for prioritizing library expansion

### D1 Migration

New migration in `apps/api/migrations/` adding the `pieces` table. No changes to existing tables. `piece_id` can be referenced from `sessions` and `observations` via an optional column added later when the piece selection UX is built.

## Testing & Validation

### Unit Tests

- **Discovery:** Given a mock ASAP directory structure, correctly identifies unique pieces, picks one score per piece, derives correct `piece_id` and `title`
- **MIDI parsing:** For known score MIDIs, verify:
  - Correct bar count
  - Time signature and key signature extracted
  - Notes assigned to correct bars
  - Tick-to-seconds conversion matches expected values
  - Pedal events extracted
  - Per-bar summary stats correct
- **JSON schema:** Output JSON validates against defined schema (all required fields present, types correct)
- **Edge cases:** Tempo changes mid-piece, time signature changes, anacrusis (pickup bars), empty bars

### Integration Test

Run the full parse pipeline on the actual ASAP cache:
- All 242 pieces parsed without errors (or document which fail and why)
- Spot-check 5-10 pieces across different composers against manual inspection
- Total JSON output size is reasonable (~5-50KB per piece)

### Validation Metrics

After initial build, report:
- Pieces parsed: X / 242
- Composer distribution (table)
- Bar count range (min/median/max)
- Note count range (min/median/max)
- Pieces with time signature changes
- Pieces with tempo changes
- Parse failures with error details

## Future Enhancements

| Enhancement | When | Notes |
|-------------|------|-------|
| MAESTRO/IMSLP/MuseScore expansion | After core library proves useful | Source score MIDIs externally for pieces not in ASAP |
| MusicXML import | When richer annotations needed | Explicit dynamics (pp, ff, cresc.), articulation text, section labels |
| Piece identification | When piece selection UX is built | AMT-based pitch sequence matching or audio fingerprinting |
| Score difficulty annotations | When exercise database (Slice 07) is built | Per-bar difficulty ratings for adaptive feedback |
