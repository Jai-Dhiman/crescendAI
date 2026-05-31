//! Isolated bar-analysis tests driven by real AMT-transcribed notes.
//! Tier 2: absolute MIDI path (no score context) — used when piece-ID fails.
//! Tier 1: bar-aligned path — uses real BWV 846 score JSON + a synthetic bar_map
//!         (since DTW/piece-ID currently fails for this prelude). This isolates the
//!         downstream analysis engine from the piece-ID + DTW gate.
//!
//! Inputs via env vars (paths to JSON):
//!   NOTES_JSON   — AMT output: {notes:[{pitch,onset,offset,velocity}], pedals:[{time,value}]}
//!   SCORES_CSV   — six comma-separated f64 scores
//!   SCORE_JSON   — score data (e.g. model/data/scores/bach.prelude.bwv_846.json) [tier1 only]
//!   BAR_RANGE    — synthetic bar range, e.g. "6,11"  [tier1 only]

use crate::bar_analysis;
use crate::types::{
    BarMap, NoteAlignment, PerfNote, PerfPedalEvent, ReferenceProfile, ScoreBar, ScoreContext,
    ScoreData, ScoreNote, ScorePedalEvent,
};
use std::fs;

fn read(key: &str) -> String {
    let p = std::env::var(key).unwrap_or_else(|_| panic!("env {key} not set"));
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {p}: {e}"))
}

fn parse_scores(env_var: &str) -> [f64; 6] {
    let csv = std::env::var(env_var).unwrap_or_else(|_| "0.5,0.5,0.5,0.5,0.5,0.5".into());
    let v: Vec<f64> = csv.split(',').map(|s| s.trim().parse().unwrap()).collect();
    assert_eq!(v.len(), 6, "need 6 scores");
    v.try_into().unwrap()
}

#[derive(serde::Deserialize)]
struct AmtDump {
    notes: Vec<PerfNote>,
    #[serde(default)]
    pedals: Vec<PerfPedalEvent>,
}

#[test]
fn bar_analysis_tier2_on_real_notes() {
    let dump: AmtDump = serde_json::from_str(&read("NOTES_JSON")).expect("parse notes");
    let scores = parse_scores("SCORES_CSV");

    eprintln!(
        "--- Tier 2 bar-analysis on {} notes, {} pedals ---",
        dump.notes.len(),
        dump.pedals.len()
    );
    let result = bar_analysis::analyze_tier2(&dump.notes, &dump.pedals, &scores);
    eprintln!(
        "ChunkAnalysis (Tier 2):\n{}",
        serde_json::to_string_pretty(&result).unwrap()
    );
}

// ---------- Tier 1 path (synthetic bar_map + real score) ----------

#[derive(serde::Deserialize)]
struct ScoreJsonRaw {
    piece_id: String,
    composer: String,
    title: String,
    #[serde(default)]
    key_signature: Option<String>,
    total_bars: u32,
    bars: Vec<ScoreBarRaw>,
}
#[derive(serde::Deserialize)]
struct ScoreBarRaw {
    bar_number: u32,
    start_tick: u32,
    start_seconds: f64,
    time_signature: String,
    notes: Vec<ScoreNoteRaw>,
    #[serde(default)]
    pedal_events: Vec<ScorePedalEventRaw>,
}
#[derive(serde::Deserialize)]
struct ScoreNoteRaw {
    pitch: u8,
    pitch_name: String,
    velocity: u8,
    onset_tick: u32,
    onset_seconds: f64,
    duration_ticks: u32,
    duration_seconds: f64,
    track: u8,
}
#[derive(serde::Deserialize)]
struct ScorePedalEventRaw {
    r#type: String,
    tick: u32,
    seconds: f64,
}

fn build_score_data(raw: &ScoreJsonRaw) -> ScoreData {
    let bars: Vec<ScoreBar> = raw
        .bars
        .iter()
        .map(|b| {
            let notes: Vec<ScoreNote> = b
                .notes
                .iter()
                .map(|n| ScoreNote {
                    pitch: n.pitch,
                    pitch_name: n.pitch_name.clone(),
                    velocity: n.velocity,
                    onset_tick: n.onset_tick,
                    onset_seconds: n.onset_seconds,
                    duration_ticks: n.duration_ticks,
                    duration_seconds: n.duration_seconds,
                    track: n.track,
                })
                .collect();
            let pedals: Vec<ScorePedalEvent> = b
                .pedal_events
                .iter()
                .map(|p| ScorePedalEvent {
                    event_type: p.r#type.clone(),
                    tick: p.tick,
                    seconds: p.seconds,
                })
                .collect();
            let pitch_range: Vec<u8> = if notes.is_empty() {
                vec![0, 0]
            } else {
                let lo = notes.iter().map(|n| n.pitch).min().unwrap();
                let hi = notes.iter().map(|n| n.pitch).max().unwrap();
                vec![lo, hi]
            };
            let mean_velocity: u8 = if notes.is_empty() {
                0
            } else {
                (notes.iter().map(|n| n.velocity as u32).sum::<u32>() / notes.len() as u32) as u8
            };
            let note_count: u32 = notes.len() as u32;
            ScoreBar {
                bar_number: b.bar_number,
                start_tick: b.start_tick,
                start_seconds: b.start_seconds,
                time_signature: b.time_signature.clone(),
                notes,
                pedal_events: pedals,
                note_count,
                pitch_range,
                mean_velocity,
            }
        })
        .collect();

    ScoreData {
        piece_id: raw.piece_id.clone(),
        composer: raw.composer.clone(),
        title: raw.title.clone(),
        key_signature: raw.key_signature.clone(),
        time_signatures: vec![],
        tempo_markings: vec![],
        total_bars: raw.total_bars,
        bars,
    }
}

fn synth_alignments(
    perf: &[PerfNote],
    bar_start: u32,
    bar_end: u32,
    slice_start_s: f64,
    slice_end_s: f64,
    score_bars: &[ScoreBar],
) -> Vec<NoteAlignment> {
    // Uniformly distribute perf notes across the bar range based on onset time.
    // Pair each with the nearest-pitch note in the corresponding score bar (if any).
    let span_s = (slice_end_s - slice_start_s).max(1e-6);
    let n_bars = (bar_end - bar_start + 1) as f64;
    perf.iter()
        .map(|n| {
            let frac = ((n.onset - slice_start_s) / span_s).clamp(0.0, 0.999_999);
            let offset_bars = (frac * n_bars) as u32;
            let bar = (bar_start + offset_bars).min(bar_end);
            let score_pitch = score_bars
                .iter()
                .find(|sb| sb.bar_number == bar)
                .and_then(|sb| {
                    sb.notes
                        .iter()
                        .min_by_key(|sn| (sn.pitch as i32 - n.pitch as i32).abs())
                        .map(|sn| sn.pitch)
                })
                .unwrap_or(n.pitch);
            NoteAlignment {
                perf_onset: n.onset,
                perf_pitch: n.pitch,
                perf_velocity: n.velocity,
                score_bar: bar,
                score_beat: frac * 4.0,
                score_pitch,
                onset_deviation_ms: 0.0,
            }
        })
        .collect()
}

#[test]
fn bar_analysis_tier1_on_real_notes_synthetic_alignment() {
    let dump: AmtDump = serde_json::from_str(&read("NOTES_JSON")).expect("parse notes");
    let scores = parse_scores("SCORES_CSV");
    let raw: ScoreJsonRaw = serde_json::from_str(&read("SCORE_JSON")).expect("parse score");
    let bar_range_csv = std::env::var("BAR_RANGE").unwrap_or_else(|_| "6,11".into());
    let parts: Vec<u32> = bar_range_csv
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let (bar_start, bar_end) = (parts[0], parts[1]);

    let score = build_score_data(&raw);
    let slice_start_s = dump.notes.first().map(|n| n.onset).unwrap_or(0.0);
    let slice_end_s = dump.notes.last().map(|n| n.offset).unwrap_or(slice_start_s + 25.0);

    let alignments = synth_alignments(
        &dump.notes,
        bar_start,
        bar_end,
        slice_start_s,
        slice_end_s,
        &score.bars,
    );

    let bar_map = BarMap {
        chunk_index: 0,
        bar_start,
        bar_end,
        alignments,
        confidence: 0.8,
        is_reanchored: false,
    };

    let score_ctx = ScoreContext {
        piece_id: score.piece_id.clone(),
        composer: score.composer.clone(),
        title: score.title.clone(),
        score,
        reference: None::<ReferenceProfile>,
        match_confidence: 0.9,
    };

    eprintln!(
        "--- Tier 1 bar-analysis on {} notes, bars {}-{} ---",
        dump.notes.len(),
        bar_start,
        bar_end
    );
    let result =
        bar_analysis::analyze_tier1(&bar_map, &dump.notes, &dump.pedals, &scores, &score_ctx);
    eprintln!(
        "ChunkAnalysis (Tier 1):\n{}",
        serde_json::to_string_pretty(&result).unwrap()
    );
}
