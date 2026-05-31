// CLI wrapper over chroma_dtw_native for the Python eval harness.
//
// argv: <score_bars_json_path> <frame_rate_hz> <decim_hz> <n_audio_frames>
// stdin: raw little-endian float32, row-major 12 x n_audio_frames
// stdout: one JSON object: bar_min, bar_max, cost, bar_per_frame,
//         score_frame_per_audio_frame, predicted_score_frame

use std::io::Read;
use std::process::ExitCode;

use score_analysis::chroma_dtw_native;
use score_analysis::types::ScoreBar;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!(
            "usage: dtw_chunk_cli <score_json> <frame_rate_hz> <decim_hz> <n_audio_frames>"
        );
        return ExitCode::from(2);
    }
    let score_path = &args[1];
    let frame_rate: f32 = match args[2].parse() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bad frame_rate: {e}");
            return ExitCode::from(2);
        }
    };
    let decim: f32 = match args[3].parse() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bad decim: {e}");
            return ExitCode::from(2);
        }
    };
    let n_audio: u32 = match args[4].parse() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bad n_audio: {e}");
            return ExitCode::from(2);
        }
    };

    let score_text = match std::fs::read_to_string(score_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("read score: {e}");
            return ExitCode::from(2);
        }
    };
    let score_bars: Vec<ScoreBar> = match serde_json::from_str(&score_text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("parse score: {e}");
            return ExitCode::from(2);
        }
    };

    let mut buf = Vec::new();
    if let Err(e) = std::io::stdin().read_to_end(&mut buf) {
        eprintln!("read stdin: {e}");
        return ExitCode::from(2);
    }
    let expected_bytes = (n_audio as usize) * 12 * 4;
    if buf.len() != expected_bytes {
        eprintln!(
            "stdin {} bytes != expected {}",
            buf.len(),
            expected_bytes
        );
        return ExitCode::from(2);
    }
    let audio: Vec<f32> = buf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let result = match chroma_dtw_native(&audio, n_audio, &score_bars, frame_rate, decim) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("dtw: {e}");
            return ExitCode::from(3);
        }
    };

    // predicted_score_frame: mid-chunk score-side frame index at the audio frame rate,
    // read from the warping path now exposed on BarMapChroma.
    let mid_audio = result.score_frame_per_audio_frame.len() / 2;
    let predicted_score_frame: i64 = result
        .score_frame_per_audio_frame
        .get(mid_audio)
        .copied()
        .unwrap_or(0) as i64;

    let out = serde_json::json!({
        "bar_min": result.bar_min,
        "bar_max": result.bar_max,
        "cost": result.cost,
        "bar_per_frame": result.bar_per_frame,
        "score_frame_per_audio_frame": result.score_frame_per_audio_frame,
        "predicted_score_frame": predicted_score_frame,
    });
    println!("{}", out);
    ExitCode::from(0)
}
