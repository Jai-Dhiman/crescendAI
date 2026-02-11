use anyhow::{Context, Result};
use std::process::Command;

use crate::store::MasterclassStore;

pub async fn download_audio(store: &MasterclassStore, video_id: &str, force: bool) -> Result<()> {
    let output_path = store.audio_path(video_id);

    // Skip if already downloaded (non-empty file exists)
    if !force && output_path.exists() {
        let metadata = std::fs::metadata(&output_path)?;
        if metadata.len() > 0 {
            tracing::info!("Audio already exists for {}, skipping", video_id);
            return Ok(());
        }
    }

    let video = store
        .get_video(video_id)?
        .with_context(|| format!("Video metadata not found for {}", video_id))?;

    tracing::info!("Downloading audio for: {} ({})", video.title, video_id);

    let output_template = output_path
        .to_str()
        .with_context(|| "Invalid output path")?;

    let result = Command::new("yt-dlp")
        .args([
            "-x",
            "--audio-format",
            "wav",
            "--postprocessor-args",
            "ffmpeg:-ar 16000 -ac 1",
            "-o",
            output_template,
            &video.url,
        ])
        .output()
        .with_context(|| "Failed to run yt-dlp. Is it installed?")?;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        anyhow::bail!("yt-dlp download failed for {}: {}", video_id, stderr);
    }

    // Verify file was created
    anyhow::ensure!(
        output_path.exists(),
        "yt-dlp completed but output file not found at {}",
        output_path.display()
    );

    let size = std::fs::metadata(&output_path)?.len();
    tracing::info!(
        "Downloaded {} ({:.1} MB)",
        video_id,
        size as f64 / 1_048_576.0
    );

    Ok(())
}
