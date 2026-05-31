// End-to-end: spawn the release binary, pipe a tiny chroma to stdin,
// parse JSON from stdout, assert it has the required fields.

use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn dtw_chunk_cli_prints_expected_json_fields() {
    // Build the binary first.
    let status = Command::new("cargo")
        .args(["build", "--release", "--bin", "dtw_chunk_cli"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("cargo build");
    assert!(status.success(), "cargo build failed");

    // Tiny score bar JSON: one bar with one C4 note.
    let score_json = r#"[{
        "bar_number": 1, "start_tick": 0, "start_seconds": 0.0,
        "time_signature": "4/4",
        "notes": [{
            "pitch": 60, "pitch_name": "C4", "velocity": 80,
            "onset_tick": 0, "onset_seconds": 0.0,
            "duration_ticks": 480, "duration_seconds": 0.2, "track": 0
        }],
        "pedal_events": [], "note_count": 1,
        "pitch_range": [60], "mean_velocity": 80
    }]"#;
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), score_json).unwrap();

    // 2 audio frames, both strongly C (pitch class 0).
    // Row-major 12 x 2: pc=0 at index 0 for each column.
    let mut audio = vec![0.0_f32; 12 * 2];
    audio[0] = 1.0; // pc=0, af=0
    audio[1] = 1.0; // pc=0, af=1
    let audio_bytes: Vec<u8> = audio.iter().flat_map(|f| f.to_le_bytes()).collect();

    let bin = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/release/dtw_chunk_cli");
    let mut child = Command::new(&bin)
        .args([tmp.path().to_str().unwrap(), "10.0", "10.0", "2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn dtw_chunk_cli");
    child.stdin.as_mut().unwrap().write_all(&audio_bytes).unwrap();
    let out = child.wait_with_output().expect("wait_with_output");
    assert!(
        out.status.success(),
        "cli failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect(&stdout);
    assert!(v.get("bar_min").is_some());
    assert!(v.get("bar_max").is_some());
    assert!(v.get("cost").is_some());
    assert!(v.get("bar_per_frame").is_some());
    assert!(v.get("score_frame_per_audio_frame").is_some());
    let sfpaf = v
        .get("score_frame_per_audio_frame")
        .unwrap()
        .as_array()
        .unwrap();
    // 2 audio frames in, 2 score-frame indices out.
    assert_eq!(sfpaf.len(), 2, "expected one score-frame index per audio frame");
    assert!(v.get("predicted_score_frame").is_some());
}

// Carry-over risk: when frame_rate != decim the length of score_frame_per_audio_frame
// must still equal n_audio (full rate, not decimated).
#[test]
fn score_frame_per_audio_frame_length_equals_n_audio_when_rates_differ() {
    // Build the binary first.
    let status = Command::new("cargo")
        .args(["build", "--release", "--bin", "dtw_chunk_cli"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("cargo build");
    assert!(status.success(), "cargo build failed");

    let score_json = r#"[{
        "bar_number": 1, "start_tick": 0, "start_seconds": 0.0,
        "time_signature": "4/4",
        "notes": [{
            "pitch": 60, "pitch_name": "C4", "velocity": 80,
            "onset_tick": 0, "onset_seconds": 0.0,
            "duration_ticks": 480, "duration_seconds": 0.5, "track": 0
        }],
        "pedal_events": [], "note_count": 1,
        "pitch_range": [60], "mean_velocity": 80
    }]"#;
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), score_json).unwrap();

    // 10 audio frames at frame_rate=50 Hz, decim=5 Hz (ratio=10:1)
    let n_audio: usize = 10;
    let mut audio = vec![0.0_f32; 12 * n_audio];
    // All frames strongly C
    for af in 0..n_audio {
        audio[0 * n_audio + af] = 1.0;
    }
    let audio_bytes: Vec<u8> = audio.iter().flat_map(|f| f.to_le_bytes()).collect();

    let bin = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/release/dtw_chunk_cli");
    let mut child = Command::new(&bin)
        // frame_rate=50, decim=5 — unequal rates
        .args([tmp.path().to_str().unwrap(), "50.0", "5.0", &n_audio.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn dtw_chunk_cli");
    child.stdin.as_mut().unwrap().write_all(&audio_bytes).unwrap();
    let out = child.wait_with_output().expect("wait_with_output");
    assert!(
        out.status.success(),
        "cli failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect(&stdout);
    let sfpaf = v
        .get("score_frame_per_audio_frame")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(
        sfpaf.len(),
        n_audio,
        "score_frame_per_audio_frame length {} != n_audio {}",
        sfpaf.len(),
        n_audio
    );
    // bar_per_frame is at decim rate: ceil(10/10) = 1 entry
    let bpf = v.get("bar_per_frame").unwrap().as_array().unwrap();
    assert_eq!(bpf.len(), 1, "bar_per_frame should have ceil(n_audio/decim_step)=1 entry");
}
