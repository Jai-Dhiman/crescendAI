use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use crate::config::{MASTERCLASS_KEYWORDS, MAX_VIDEO_DURATION_SECS, MIN_VIDEO_DURATION_SECS};
use crate::schemas::*;
use crate::store::MasterclassStore;

pub async fn discover(store: &MasterclassStore, data_dir: &Path) -> Result<Vec<VideoMetadata>> {
    let sources_path = data_dir.join("sources.yaml");
    anyhow::ensure!(
        sources_path.exists(),
        "sources.yaml not found at {}. Create it first.",
        sources_path.display()
    );

    let content = std::fs::read_to_string(&sources_path)
        .with_context(|| format!("Failed to read {}", sources_path.display()))?;
    let sources: SourcesConfig = serde_yaml::from_str(&content)
        .with_context(|| "Failed to parse sources.yaml")?;

    let existing = store.load_video_map()?;
    let mut seen_ids: HashSet<String> = existing.keys().cloned().collect();
    let mut new_videos = Vec::new();

    // Curated individual videos
    for video_src in &sources.videos {
        let video_id = extract_video_id(&video_src.url)?;
        if seen_ids.contains(&video_id) {
            tracing::debug!("Skipping already-known video: {}", video_id);
            continue;
        }

        match fetch_video_metadata(&video_id, &video_src.url).await {
            Ok(mut meta) => {
                meta.source = "curated".to_string();
                if let Some(ref teacher) = video_src.teacher {
                    meta.teacher = Some(teacher.clone());
                }
                if let Some(ref piece) = video_src.piece {
                    meta.pieces = vec![piece.clone()];
                }
                if let Some(ref composer) = video_src.composer {
                    meta.composers = vec![composer.clone()];
                }
                seen_ids.insert(video_id);
                store.save_video(&meta)?;
                store.mark_stage_complete(&meta.video_id, &PipelineStage::Discover)?;
                new_videos.push(meta);
            }
            Err(e) => {
                tracing::warn!("Failed to fetch metadata for {}: {}", video_src.url, e);
            }
        }
    }

    // Curated channels
    for channel in &sources.channels {
        tracing::info!("Discovering from channel: {}", channel.url);
        match discover_from_channel(&channel.url, channel.max_videos).await {
            Ok(entries) => {
                for entry in entries {
                    let video_id = entry.video_id.clone();
                    if seen_ids.contains(&video_id) {
                        continue;
                    }
                    if !passes_filters(&entry) {
                        tracing::debug!("Filtered out: {} ({})", entry.title, video_id);
                        continue;
                    }
                    let mut meta = entry;
                    meta.source = "curated".to_string();
                    if let Some(ref teacher) = channel.teacher {
                        meta.teacher = Some(teacher.clone());
                    }
                    seen_ids.insert(video_id.clone());
                    store.save_video(&meta)?;
                    store.mark_stage_complete(&meta.video_id, &PipelineStage::Discover)?;
                    new_videos.push(meta);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to discover from channel {}: {}", channel.url, e);
            }
        }
    }

    // Search queries
    for query in &sources.search_queries {
        tracing::info!("Searching: {}", query);
        match search_videos(query).await {
            Ok(entries) => {
                for entry in entries {
                    let video_id = entry.video_id.clone();
                    if seen_ids.contains(&video_id) {
                        continue;
                    }
                    if !passes_filters(&entry) {
                        continue;
                    }
                    let mut meta = entry;
                    meta.source = "search".to_string();
                    seen_ids.insert(video_id.clone());
                    store.save_video(&meta)?;
                    store.mark_stage_complete(&meta.video_id, &PipelineStage::Discover)?;
                    new_videos.push(meta);
                }
            }
            Err(e) => {
                tracing::warn!("Failed search '{}': {}", query, e);
            }
        }
    }

    Ok(new_videos)
}

fn extract_video_id(url: &str) -> Result<String> {
    // Handle various YouTube URL formats
    if let Some(id) = url.strip_prefix("https://www.youtube.com/watch?v=") {
        Ok(id.split('&').next().unwrap_or(id).to_string())
    } else if let Some(id) = url.strip_prefix("https://youtube.com/watch?v=") {
        Ok(id.split('&').next().unwrap_or(id).to_string())
    } else if let Some(id) = url.strip_prefix("https://youtu.be/") {
        Ok(id.split('?').next().unwrap_or(id).to_string())
    } else {
        anyhow::bail!("Cannot extract video ID from URL: {}", url);
    }
}

async fn fetch_video_metadata(video_id: &str, url: &str) -> Result<VideoMetadata> {
    let output = Command::new("yt-dlp")
        .args(["--dump-json", "--no-download", url])
        .output()
        .with_context(|| "Failed to run yt-dlp. Is it installed?")?;

    anyhow::ensure!(
        output.status.success(),
        "yt-dlp failed for {}: {}",
        url,
        String::from_utf8_lossy(&output.stderr)
    );

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)
        .with_context(|| "Failed to parse yt-dlp JSON output")?;

    Ok(VideoMetadata {
        video_id: video_id.to_string(),
        url: url.to_string(),
        title: json["title"].as_str().unwrap_or("").to_string(),
        channel: json["channel"].as_str().unwrap_or("").to_string(),
        duration_seconds: json["duration"].as_f64().unwrap_or(0.0),
        upload_date: json["upload_date"].as_str().map(|s| s.to_string()),
        description: json["description"].as_str().map(|s| s.to_string()),
        teacher: None,
        pieces: Vec::new(),
        composers: Vec::new(),
        source: String::new(),
        discovered_at: chrono::Utc::now().to_rfc3339(),
    })
}

async fn discover_from_channel(channel_url: &str, max_videos: u32) -> Result<Vec<VideoMetadata>> {
    let output = Command::new("yt-dlp")
        .args([
            "--dump-json",
            "--flat-playlist",
            "--playlist-end",
            &max_videos.to_string(),
            channel_url,
        ])
        .output()
        .with_context(|| format!("Failed to run yt-dlp for channel {}", channel_url))?;

    if !output.status.success() {
        anyhow::bail!(
            "yt-dlp failed for channel {}: {}",
            channel_url,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut videos = Vec::new();

    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(json) => {
                let video_id = json["id"].as_str().unwrap_or("").to_string();
                if video_id.is_empty() {
                    continue;
                }
                let title = json["title"].as_str().unwrap_or("").to_string();
                let duration = json["duration"].as_f64().unwrap_or(0.0);
                let url = format!("https://www.youtube.com/watch?v={}", video_id);

                videos.push(VideoMetadata {
                    video_id,
                    url,
                    title,
                    channel: json["channel"].as_str().unwrap_or("").to_string(),
                    duration_seconds: duration,
                    upload_date: json["upload_date"].as_str().map(|s| s.to_string()),
                    description: json["description"].as_str().map(|s| s.to_string()),
                    teacher: None,
                    pieces: Vec::new(),
                    composers: Vec::new(),
                    source: String::new(),
                    discovered_at: chrono::Utc::now().to_rfc3339(),
                });
            }
            Err(e) => {
                tracing::debug!("Failed to parse channel entry: {}", e);
            }
        }
    }

    Ok(videos)
}

async fn search_videos(query: &str) -> Result<Vec<VideoMetadata>> {
    let search_query = format!("ytsearch20:{}", query);
    let output = Command::new("yt-dlp")
        .args(["--dump-json", "--flat-playlist", &search_query])
        .output()
        .with_context(|| format!("Failed to run yt-dlp search for '{}'", query))?;

    if !output.status.success() {
        anyhow::bail!(
            "yt-dlp search failed for '{}': {}",
            query,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut videos = Vec::new();

    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            let video_id = json["id"].as_str().unwrap_or("").to_string();
            if video_id.is_empty() {
                continue;
            }
            let url = format!("https://www.youtube.com/watch?v={}", video_id);
            videos.push(VideoMetadata {
                video_id,
                url,
                title: json["title"].as_str().unwrap_or("").to_string(),
                channel: json["channel"].as_str().unwrap_or("").to_string(),
                duration_seconds: json["duration"].as_f64().unwrap_or(0.0),
                upload_date: json["upload_date"].as_str().map(|s| s.to_string()),
                description: json["description"].as_str().map(|s| s.to_string()),
                teacher: None,
                pieces: Vec::new(),
                composers: Vec::new(),
                source: String::new(),
                discovered_at: chrono::Utc::now().to_rfc3339(),
            });
        }
    }

    Ok(videos)
}

fn passes_filters(video: &VideoMetadata) -> bool {
    // Duration filter
    if video.duration_seconds > 0.0
        && (video.duration_seconds < MIN_VIDEO_DURATION_SECS
            || video.duration_seconds > MAX_VIDEO_DURATION_SECS)
    {
        return false;
    }

    // Keyword filter: title must contain at least one masterclass keyword
    let title_lower = video.title.to_lowercase();
    let has_keyword = MASTERCLASS_KEYWORDS
        .iter()
        .any(|kw| title_lower.contains(kw));

    if !has_keyword {
        // Also check description if available
        if let Some(ref desc) = video.description {
            let desc_lower = desc.to_lowercase();
            return MASTERCLASS_KEYWORDS
                .iter()
                .any(|kw| desc_lower.contains(kw));
        }
        return false;
    }

    true
}
