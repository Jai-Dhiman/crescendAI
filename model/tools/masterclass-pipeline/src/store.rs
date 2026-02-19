use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::schemas::*;

pub struct MasterclassStore {
    pub data_dir: PathBuf,
    pub audio_dir: PathBuf,
    pub transcripts_dir: PathBuf,
    pub segments_dir: PathBuf,
    pub moments_dir: PathBuf,
    pub state_dir: PathBuf,
    pub models_dir: PathBuf,
}

impl MasterclassStore {
    pub fn new(data_dir: &Path) -> Result<Self> {
        let store = Self {
            data_dir: data_dir.to_path_buf(),
            audio_dir: data_dir.join("audio"),
            transcripts_dir: data_dir.join("transcripts"),
            segments_dir: data_dir.join("segments"),
            moments_dir: data_dir.join("teaching_moments"),
            state_dir: data_dir.join("state"),
            models_dir: data_dir.join("models"),
        };
        store.ensure_dirs()?;
        Ok(store)
    }

    fn ensure_dirs(&self) -> Result<()> {
        for dir in [
            &self.data_dir,
            &self.audio_dir,
            &self.transcripts_dir,
            &self.segments_dir,
            &self.moments_dir,
            &self.state_dir,
            &self.models_dir,
        ] {
            fs::create_dir_all(dir)
                .with_context(|| format!("Failed to create directory: {}", dir.display()))?;
        }
        Ok(())
    }

    // --- JSONL I/O ---

    pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let file = fs::File::open(path)
            .with_context(|| format!("Failed to open {}", path.display()))?;
        let reader = BufReader::new(file);
        let mut items = Vec::new();
        for (i, line) in reader.lines().enumerate() {
            let line = line.with_context(|| format!("Failed to read line {} of {}", i + 1, path.display()))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let item: T = serde_json::from_str(trimmed)
                .with_context(|| format!("Failed to parse line {} of {}: {}", i + 1, path.display(), trimmed))?;
            items.push(item);
        }
        Ok(items)
    }

    pub fn append_jsonl<T: Serialize>(path: &Path, item: &T) -> Result<()> {
        let tmp_path = path.with_extension("jsonl.tmp");
        // Copy existing content + new line to tmp, then rename
        {
            let mut tmp_file = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)
                .with_context(|| format!("Failed to create tmp file: {}", tmp_path.display()))?;

            // Copy existing lines
            if path.exists() {
                let existing = fs::read(path)
                    .with_context(|| format!("Failed to read {}", path.display()))?;
                tmp_file.write_all(&existing)?;
            }

            // Append new line
            let line = serde_json::to_string(item)?;
            writeln!(tmp_file, "{}", line)?;
            tmp_file.flush()?;
        }

        fs::rename(&tmp_path, path)
            .with_context(|| format!("Failed to rename {} to {}", tmp_path.display(), path.display()))?;
        Ok(())
    }

    pub fn write_jsonl<T: Serialize>(path: &Path, items: &[T]) -> Result<()> {
        let tmp_path = path.with_extension("jsonl.tmp");
        {
            let mut file = fs::File::create(&tmp_path)
                .with_context(|| format!("Failed to create {}", tmp_path.display()))?;
            for item in items {
                let line = serde_json::to_string(item)?;
                writeln!(file, "{}", line)?;
            }
            file.flush()?;
        }
        fs::rename(&tmp_path, path)?;
        Ok(())
    }

    // --- Video metadata ---

    fn videos_path(&self) -> PathBuf {
        self.data_dir.join("videos.jsonl")
    }

    pub fn load_videos(&self) -> Result<Vec<VideoMetadata>> {
        Self::read_jsonl(&self.videos_path())
    }

    pub fn save_video(&self, video: &VideoMetadata) -> Result<()> {
        Self::append_jsonl(&self.videos_path(), video)
    }

    pub fn load_video_map(&self) -> Result<HashMap<String, VideoMetadata>> {
        let videos = self.load_videos()?;
        Ok(videos.into_iter().map(|v| (v.video_id.clone(), v)).collect())
    }

    pub fn get_video(&self, video_id: &str) -> Result<Option<VideoMetadata>> {
        let map = self.load_video_map()?;
        Ok(map.get(video_id).cloned())
    }

    // --- Pipeline state ---

    fn state_path(&self) -> PathBuf {
        self.state_dir.join("pipeline_state.jsonl")
    }

    fn load_state(&self) -> Result<Vec<StageState>> {
        Self::read_jsonl(&self.state_path())
    }

    pub fn mark_stage_complete(&self, video_id: &str, stage: &PipelineStage) -> Result<()> {
        let state = StageState {
            video_id: video_id.to_string(),
            stage: stage.clone(),
            status: StageStatus::Completed,
            completed_at: Some(chrono::Utc::now().to_rfc3339()),
            error: None,
        };
        Self::append_jsonl(&self.state_path(), &state)
    }

    pub fn mark_stage_failed(&self, video_id: &str, stage: &PipelineStage, error: &str) -> Result<()> {
        let state = StageState {
            video_id: video_id.to_string(),
            stage: stage.clone(),
            status: StageStatus::Failed,
            completed_at: Some(chrono::Utc::now().to_rfc3339()),
            error: Some(error.to_string()),
        };
        Self::append_jsonl(&self.state_path(), &state)
    }

    #[allow(dead_code)]
    pub fn is_stage_complete(&self, video_id: &str, stage: &PipelineStage) -> Result<bool> {
        let states = self.load_state()?;
        // Last state entry for this video+stage wins
        Ok(states
            .iter()
            .rev()
            .find(|s| s.video_id == video_id && s.stage == *stage)
            .map(|s| s.status == StageStatus::Completed)
            .unwrap_or(false))
    }

    pub fn get_videos_needing_stage(&self, stage: &PipelineStage) -> Result<Vec<String>> {
        let videos = self.load_videos()?;
        let states = self.load_state()?;

        // Build set of completed video_ids for this stage (last entry wins)
        let mut completed: HashMap<String, bool> = HashMap::new();
        for s in &states {
            if s.stage == *stage {
                completed.insert(s.video_id.clone(), s.status == StageStatus::Completed);
            }
        }

        // For stages after Discover, also require the previous stage to be complete
        let prev_stage = match stage {
            PipelineStage::Discover => None,
            PipelineStage::Download => Some(PipelineStage::Discover),
            PipelineStage::Transcribe => Some(PipelineStage::Download),
            PipelineStage::Segment => Some(PipelineStage::Transcribe),
            PipelineStage::Extract => Some(PipelineStage::Segment),
            PipelineStage::Identify => Some(PipelineStage::Transcribe),
        };

        let mut prev_completed: HashMap<String, bool> = HashMap::new();
        if let Some(prev) = &prev_stage {
            for s in &states {
                if s.stage == *prev {
                    prev_completed.insert(s.video_id.clone(), s.status == StageStatus::Completed);
                }
            }
        }

        let mut needed = Vec::new();
        for v in &videos {
            let already_done = completed.get(&v.video_id).copied().unwrap_or(false);
            if already_done {
                continue;
            }
            // Check prerequisite
            if prev_stage.is_some() {
                let prev_done = prev_completed.get(&v.video_id).copied().unwrap_or(false);
                if !prev_done {
                    continue;
                }
            }
            needed.push(v.video_id.clone());
        }
        Ok(needed)
    }

    /// Like `get_videos_needing_stage`, but with an explicit prerequisite override.
    /// Useful when `--no-transcript` mode needs Segment videos with only Download complete.
    pub fn get_videos_needing_stage_with_prereq(
        &self,
        stage: &PipelineStage,
        prereq: &PipelineStage,
    ) -> Result<Vec<String>> {
        let videos = self.load_videos()?;
        let states = self.load_state()?;

        let mut completed: HashMap<String, bool> = HashMap::new();
        for s in &states {
            if s.stage == *stage {
                completed.insert(s.video_id.clone(), s.status == StageStatus::Completed);
            }
        }

        let mut prev_completed: HashMap<String, bool> = HashMap::new();
        for s in &states {
            if s.stage == *prereq {
                prev_completed.insert(s.video_id.clone(), s.status == StageStatus::Completed);
            }
        }

        let mut needed = Vec::new();
        for v in &videos {
            let already_done = completed.get(&v.video_id).copied().unwrap_or(false);
            if already_done {
                continue;
            }
            let prev_done = prev_completed.get(&v.video_id).copied().unwrap_or(false);
            if !prev_done {
                continue;
            }
            needed.push(v.video_id.clone());
        }
        Ok(needed)
    }

    // --- Transcripts ---

    pub fn transcript_path(&self, video_id: &str) -> PathBuf {
        self.transcripts_dir.join(format!("{}.json", video_id))
    }

    pub fn save_transcript(&self, transcript: &Transcript) -> Result<()> {
        let path = self.transcript_path(&transcript.video_id);
        let tmp_path = path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(transcript)?;
        fs::write(&tmp_path, json)?;
        fs::rename(&tmp_path, &path)?;
        Ok(())
    }

    pub fn load_transcript(&self, video_id: &str) -> Result<Option<Transcript>> {
        let path = self.transcript_path(video_id);
        if !path.exists() {
            return Ok(None);
        }
        let content = fs::read_to_string(&path)?;
        let transcript: Transcript = serde_json::from_str(&content)?;
        Ok(Some(transcript))
    }

    // --- Segmentation ---

    pub fn segment_path(&self, video_id: &str) -> PathBuf {
        self.segments_dir.join(format!("{}.json", video_id))
    }

    pub fn save_segmentation(&self, result: &SegmentationResult) -> Result<()> {
        let path = self.segment_path(&result.video_id);
        let tmp_path = path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(result)?;
        fs::write(&tmp_path, json)?;
        fs::rename(&tmp_path, &path)?;
        Ok(())
    }

    pub fn load_segmentation(&self, video_id: &str) -> Result<Option<SegmentationResult>> {
        let path = self.segment_path(video_id);
        if !path.exists() {
            return Ok(None);
        }
        let content = fs::read_to_string(&path)?;
        let result: SegmentationResult = serde_json::from_str(&content)?;
        Ok(Some(result))
    }

    // --- Teaching moments ---

    pub fn moments_path(&self, video_id: &str) -> PathBuf {
        self.moments_dir.join(format!("{}.jsonl", video_id))
    }

    pub fn save_teaching_moments(&self, video_id: &str, moments: &[TeachingMoment]) -> Result<()> {
        let path = self.moments_path(video_id);
        Self::write_jsonl(&path, moments)
    }

    pub fn load_teaching_moments(&self, video_id: &str) -> Result<Vec<TeachingMoment>> {
        let path = self.moments_path(video_id);
        Self::read_jsonl(&path)
    }

    // --- Audio path ---

    pub fn audio_path(&self, video_id: &str) -> PathBuf {
        self.audio_dir.join(format!("{}.wav", video_id))
    }

    // --- Export ---

    pub fn export_all_moments(&self, output: &Path) -> Result<usize> {
        let videos = self.load_videos()?;
        let mut all_moments = Vec::new();
        for v in &videos {
            let moments = self.load_teaching_moments(&v.video_id)?;
            all_moments.extend(moments);
        }
        let count = all_moments.len();
        Self::write_jsonl(output, &all_moments)?;
        Ok(count)
    }

    // --- Status summary ---

    pub fn status_summary(&self) -> Result<StatusSummary> {
        let videos = self.load_videos()?;
        let states = self.load_state()?;

        let stages = [
            PipelineStage::Discover,
            PipelineStage::Download,
            PipelineStage::Transcribe,
            PipelineStage::Segment,
            PipelineStage::Extract,
            PipelineStage::Identify,
        ];

        let mut stage_counts: HashMap<String, StageCounts> = HashMap::new();

        for stage in &stages {
            let mut completed = 0usize;
            let mut failed = 0usize;
            let mut pending = 0usize;

            for v in &videos {
                let last_state = states
                    .iter()
                    .rev()
                    .find(|s| s.video_id == v.video_id && s.stage == *stage);
                match last_state {
                    Some(s) if s.status == StageStatus::Completed => completed += 1,
                    Some(_) => failed += 1,
                    None => pending += 1,
                }
            }

            stage_counts.insert(
                stage.to_string(),
                StageCounts { completed, failed, pending },
            );
        }

        Ok(StatusSummary {
            total_videos: videos.len(),
            stage_counts,
        })
    }
}

pub struct StageCounts {
    pub completed: usize,
    pub failed: usize,
    pub pending: usize,
}

pub struct StatusSummary {
    pub total_videos: usize,
    pub stage_counts: HashMap<String, StageCounts>,
}

impl std::fmt::Display for StatusSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Pipeline Status ({} total videos)", self.total_videos)?;
        writeln!(f, "{:-<50}", "")?;
        let stage_order = ["discover", "download", "transcribe", "segment", "extract", "identify"];
        for stage in &stage_order {
            if let Some(counts) = self.stage_counts.get(*stage) {
                writeln!(
                    f,
                    "  {:12} | done: {:3} | failed: {:3} | pending: {:3}",
                    stage, counts.completed, counts.failed, counts.pending
                )?;
            }
        }
        Ok(())
    }
}
