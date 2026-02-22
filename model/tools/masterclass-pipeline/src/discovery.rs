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
                if let Some(ref level) = video_src.student_level {
                    meta.student_level = Some(level.clone());
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

/// Refresh existing video metadata from sources.yaml.
///
/// Matches each VideoSource to existing videos by video_id (extracted from URL),
/// and patches teacher/piece/composer/student_level using store.update_video().
/// This makes sources.yaml the permanent source of truth for curated metadata.
pub fn refresh_metadata_from_sources(store: &MasterclassStore, data_dir: &Path) -> Result<()> {
    let sources_path = data_dir.join("sources.yaml");
    if !sources_path.exists() {
        tracing::debug!("No sources.yaml found, skipping metadata refresh");
        return Ok(());
    }

    let content = std::fs::read_to_string(&sources_path)
        .with_context(|| format!("Failed to read {}", sources_path.display()))?;
    let sources: SourcesConfig = serde_yaml::from_str(&content)
        .with_context(|| "Failed to parse sources.yaml")?;

    let mut refreshed = 0u32;

    for video_src in &sources.videos {
        let video_id = match extract_video_id(&video_src.url) {
            Ok(id) => id,
            Err(_) => continue,
        };

        let updated = store.update_video(&video_id, |video| {
            if let Some(ref teacher) = video_src.teacher {
                video.teacher = Some(teacher.clone());
            }
            if let Some(ref piece) = video_src.piece {
                video.pieces = vec![piece.clone()];
            }
            if let Some(ref composer) = video_src.composer {
                video.composers = vec![composer.clone()];
            }
            if let Some(ref level) = video_src.student_level {
                video.student_level = Some(level.clone());
            }
        })?;

        if updated {
            refreshed += 1;
        }
    }

    if refreshed > 0 {
        tracing::info!("Refreshed metadata for {} videos from sources.yaml", refreshed);
    }

    Ok(())
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
        student_level: None,
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
                    student_level: None,
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
                student_level: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_video(id: &str) -> VideoMetadata {
        VideoMetadata {
            video_id: id.to_string(),
            url: format!("https://www.youtube.com/watch?v={}", id),
            title: format!("Video {}", id),
            channel: "test".to_string(),
            duration_seconds: 60.0,
            upload_date: None,
            description: None,
            teacher: None,
            pieces: vec![],
            composers: vec![],
            source: "test".to_string(),
            discovered_at: "now".to_string(),
            student_level: None,
        }
    }

    #[test]
    fn extract_video_id_standard() {
        assert_eq!(
            extract_video_id("https://www.youtube.com/watch?v=abc123").unwrap(),
            "abc123"
        );
    }

    #[test]
    fn extract_video_id_with_params() {
        assert_eq!(
            extract_video_id("https://www.youtube.com/watch?v=abc123&t=10").unwrap(),
            "abc123"
        );
    }

    #[test]
    fn extract_video_id_short_url() {
        assert_eq!(
            extract_video_id("https://youtu.be/abc123").unwrap(),
            "abc123"
        );
    }

    #[test]
    fn extract_video_id_invalid() {
        assert!(extract_video_id("https://example.com/video").is_err());
    }

    #[test]
    fn refresh_metadata_patches_student_level() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = MasterclassStore::new(tmp.path()).unwrap();

        // Save a video without student_level
        let video = make_video("IsE7XOmTwjQ");
        store.save_video(&video).unwrap();

        // Write sources.yaml with student_level
        let sources_yaml = r#"
channels: []
videos:
  - url: "https://www.youtube.com/watch?v=IsE7XOmTwjQ"
    teacher: "Leon Fleisher"
    piece: "Beethoven Pathetique"
    composer: "Beethoven"
    student_level: "advanced"
search_queries: []
"#;
        std::fs::write(tmp.path().join("sources.yaml"), sources_yaml).unwrap();

        refresh_metadata_from_sources(&store, tmp.path()).unwrap();

        let updated = store.get_video("IsE7XOmTwjQ").unwrap().unwrap();
        assert_eq!(updated.student_level.as_deref(), Some("advanced"));
        assert_eq!(updated.teacher.as_deref(), Some("Leon Fleisher"));
        assert_eq!(updated.pieces, vec!["Beethoven Pathetique"]);
        assert_eq!(updated.composers, vec!["Beethoven"]);
    }

    #[test]
    fn refresh_metadata_unknown_video_id_no_error() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = MasterclassStore::new(tmp.path()).unwrap();

        let sources_yaml = r#"
channels: []
videos:
  - url: "https://www.youtube.com/watch?v=nonexistent"
    teacher: "Nobody"
    student_level: "beginner"
search_queries: []
"#;
        std::fs::write(tmp.path().join("sources.yaml"), sources_yaml).unwrap();

        // Should not error, just skip unknown IDs
        refresh_metadata_from_sources(&store, tmp.path()).unwrap();
    }

    #[test]
    fn refresh_metadata_no_sources_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = MasterclassStore::new(tmp.path()).unwrap();

        // No sources.yaml -- should succeed silently
        refresh_metadata_from_sources(&store, tmp.path()).unwrap();
    }

    #[test]
    fn refresh_metadata_does_not_overwrite_unset_fields() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = MasterclassStore::new(tmp.path()).unwrap();

        // Save a video with existing teacher
        let mut video = make_video("abc123");
        video.teacher = Some("Original Teacher".to_string());
        video.pieces = vec!["Original Piece".to_string()];
        store.save_video(&video).unwrap();

        // Sources.yaml only has student_level, no teacher/piece/composer
        let sources_yaml = r#"
channels: []
videos:
  - url: "https://www.youtube.com/watch?v=abc123"
    student_level: "intermediate"
search_queries: []
"#;
        std::fs::write(tmp.path().join("sources.yaml"), sources_yaml).unwrap();

        refresh_metadata_from_sources(&store, tmp.path()).unwrap();

        let updated = store.get_video("abc123").unwrap().unwrap();
        // student_level should be set
        assert_eq!(updated.student_level.as_deref(), Some("intermediate"));
        // teacher should be preserved (source didn't set it)
        assert_eq!(updated.teacher.as_deref(), Some("Original Teacher"));
        // pieces should be preserved
        assert_eq!(updated.pieces, vec!["Original Piece"]);
    }
}
