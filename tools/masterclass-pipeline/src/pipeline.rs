use anyhow::Result;
use std::fmt;
use std::path::PathBuf;

use crate::schemas::PipelineStage;
use crate::store::MasterclassStore;
use crate::{discovery, download, extract, identify, llm_client, segment, transcribe};

pub struct Pipeline {
    store: MasterclassStore,
    data_dir: PathBuf,
    whisper_model: String,
    llm_model: String,
    llm_url: String,
    force: bool,
    max_videos: Option<usize>,
    dry_run: bool,
    piece_filter: Option<String>,
    openai_api_key: Option<String>,
    local: bool,
}

pub struct PipelineReport {
    pub stages: Vec<StageReport>,
}

pub struct StageReport {
    pub stage: String,
    pub processed: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub skipped: usize,
}

impl fmt::Display for PipelineReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pipeline Report")?;
        writeln!(f, "{:-<50}", "")?;
        for s in &self.stages {
            writeln!(
                f,
                "  {:12} | processed: {:3} | ok: {:3} | failed: {:3} | skipped: {:3}",
                s.stage, s.processed, s.succeeded, s.failed, s.skipped
            )?;
        }
        Ok(())
    }
}

impl Pipeline {
    pub fn new(
        store: MasterclassStore,
        data_dir: PathBuf,
        whisper_model: String,
        llm_model: String,
        llm_url: String,
        force: bool,
        max_videos: Option<usize>,
        dry_run: bool,
        piece_filter: Option<String>,
        openai_api_key: Option<String>,
        local: bool,
    ) -> Self {
        Self {
            store,
            data_dir,
            whisper_model,
            llm_model,
            llm_url,
            force,
            max_videos,
            dry_run,
            piece_filter,
            openai_api_key,
            local,
        }
    }

    pub async fn run(self) -> Result<PipelineReport> {
        let mut stages = Vec::new();

        if let Some(ref piece) = self.piece_filter {
            tracing::info!("=== Piece filter: {} ===", piece);
        }

        tracing::info!("=== Stage: Discover ===");
        stages.push(self.run_discover().await?);

        tracing::info!("=== Stage: Download ===");
        stages.push(self.run_download().await?);

        tracing::info!("=== Stage: Transcribe ===");
        if self.local {
            stages.push(self.run_transcribe_local()?);
        } else {
            stages.push(self.run_transcribe_api().await?);
        }

        if self.local {
            tracing::info!("=== Stage: Segment ===");
            stages.push(self.run_segment()?);
            tracing::info!("=== Stage: Extract ===");
            stages.push(self.run_extract().await?);
        } else {
            tracing::info!("=== Stage: Identify ===");
            stages.push(self.run_identify().await?);
        }

        Ok(PipelineReport { stages })
    }

    async fn run_discover(&self) -> Result<StageReport> {
        if self.dry_run {
            tracing::info!("[dry-run] Would discover videos");
            return Ok(StageReport {
                stage: "discover".to_string(),
                processed: 0,
                succeeded: 0,
                failed: 0,
                skipped: 0,
            });
        }

        match discovery::discover(&self.store, &self.data_dir).await {
            Ok(videos) => Ok(StageReport {
                stage: "discover".to_string(),
                processed: videos.len(),
                succeeded: videos.len(),
                failed: 0,
                skipped: 0,
            }),
            Err(e) => {
                tracing::error!("Discovery failed: {}", e);
                Ok(StageReport {
                    stage: "discover".to_string(),
                    processed: 0,
                    succeeded: 0,
                    failed: 1,
                    skipped: 0,
                })
            }
        }
    }

    async fn run_download(&self) -> Result<StageReport> {
        let videos = self.get_stage_videos(&PipelineStage::Download)?;
        let mut succeeded = 0;
        let mut failed = 0;
        let skipped = 0;

        for video_id in &videos {
            if self.dry_run {
                tracing::info!("[dry-run] Would download {}", video_id);
                continue;
            }
            match download::download_audio(&self.store, video_id, self.force).await {
                Ok(_) => {
                    self.store.mark_stage_complete(video_id, &PipelineStage::Download)?;
                    succeeded += 1;
                }
                Err(e) => {
                    tracing::error!("Download failed for {}: {}", video_id, e);
                    self.store.mark_stage_failed(video_id, &PipelineStage::Download, &e.to_string())?;
                    failed += 1;
                }
            }
        }

        Ok(StageReport {
            stage: "download".to_string(),
            processed: videos.len(),
            succeeded,
            failed,
            skipped,
        })
    }

    fn run_transcribe_local(&self) -> Result<StageReport> {
        let videos = self.get_stage_videos(&PipelineStage::Transcribe)?;
        if videos.is_empty() {
            return Ok(StageReport {
                stage: "transcribe".to_string(),
                processed: 0,
                succeeded: 0,
                failed: 0,
                skipped: 0,
            });
        }

        let model_path = self.data_dir.join("models").join(format!("ggml-{}.bin", self.whisper_model));
        let ctx = transcribe::load_whisper_context(&model_path)?;

        let mut succeeded = 0;
        let mut failed = 0;

        for video_id in &videos {
            if self.dry_run {
                tracing::info!("[dry-run] Would transcribe {}", video_id);
                continue;
            }
            match transcribe::transcribe_video(&ctx, &self.store, video_id) {
                Ok(_) => {
                    self.store.mark_stage_complete(video_id, &PipelineStage::Transcribe)?;
                    succeeded += 1;
                }
                Err(e) => {
                    tracing::error!("Transcription failed for {}: {}", video_id, e);
                    self.store.mark_stage_failed(video_id, &PipelineStage::Transcribe, &e.to_string())?;
                    failed += 1;
                }
            }
        }

        Ok(StageReport {
            stage: "transcribe".to_string(),
            processed: videos.len(),
            succeeded,
            failed,
            skipped: 0,
        })
    }

    async fn run_transcribe_api(&self) -> Result<StageReport> {
        let videos = self.get_stage_videos(&PipelineStage::Transcribe)?;
        if videos.is_empty() {
            return Ok(StageReport {
                stage: "transcribe".to_string(),
                processed: 0, succeeded: 0, failed: 0, skipped: 0,
            });
        }

        let api_key = self.openai_api_key.as_deref()
            .ok_or_else(|| anyhow::anyhow!("OpenAI API key required for Whisper API"))?;
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()?;

        let mut succeeded = 0;
        let mut failed = 0;

        for video_id in &videos {
            if self.dry_run {
                tracing::info!("[dry-run] Would transcribe {} (API)", video_id);
                continue;
            }
            match transcribe::transcribe_video_api(&http_client, api_key, &self.store, video_id).await {
                Ok(_) => {
                    self.store.mark_stage_complete(video_id, &PipelineStage::Transcribe)?;
                    succeeded += 1;
                }
                Err(e) => {
                    tracing::error!("Transcription (API) failed for {}: {}", video_id, e);
                    self.store.mark_stage_failed(video_id, &PipelineStage::Transcribe, &e.to_string())?;
                    failed += 1;
                }
            }
        }

        Ok(StageReport {
            stage: "transcribe".to_string(),
            processed: videos.len(), succeeded, failed, skipped: 0,
        })
    }

    fn run_segment(&self) -> Result<StageReport> {
        let videos = self.get_stage_videos(&PipelineStage::Segment)?;
        let mut succeeded = 0;
        let mut failed = 0;

        for video_id in &videos {
            if self.dry_run {
                tracing::info!("[dry-run] Would segment {}", video_id);
                continue;
            }
            match segment::segment_video(&self.store, video_id) {
                Ok(_) => {
                    self.store.mark_stage_complete(video_id, &PipelineStage::Segment)?;
                    succeeded += 1;
                }
                Err(e) => {
                    tracing::error!("Segmentation failed for {}: {}", video_id, e);
                    self.store.mark_stage_failed(video_id, &PipelineStage::Segment, &e.to_string())?;
                    failed += 1;
                }
            }
        }

        Ok(StageReport {
            stage: "segment".to_string(),
            processed: videos.len(),
            succeeded,
            failed,
            skipped: 0,
        })
    }

    async fn run_extract(&self) -> Result<StageReport> {
        let videos = self.get_stage_videos(&PipelineStage::Extract)?;
        if videos.is_empty() {
            return Ok(StageReport {
                stage: "extract".to_string(),
                processed: 0,
                succeeded: 0,
                failed: 0,
                skipped: 0,
            });
        }

        let client = llm_client::LlmClient::new(Some(&self.llm_url), &self.llm_model, None)?;
        let mut succeeded = 0;
        let mut failed = 0;

        for video_id in &videos {
            if self.dry_run {
                tracing::info!("[dry-run] Would extract from {}", video_id);
                continue;
            }
            match extract::extract_teaching_moments(&client, &self.store, video_id).await {
                Ok(moments) => {
                    tracing::info!("Extracted {} moments from {}", moments.len(), video_id);
                    self.store.mark_stage_complete(video_id, &PipelineStage::Extract)?;
                    succeeded += 1;
                }
                Err(e) => {
                    tracing::error!("Extraction failed for {}: {}", video_id, e);
                    self.store.mark_stage_failed(video_id, &PipelineStage::Extract, &e.to_string())?;
                    failed += 1;
                }
            }
        }

        Ok(StageReport {
            stage: "extract".to_string(),
            processed: videos.len(),
            succeeded,
            failed,
            skipped: 0,
        })
    }

    async fn run_identify(&self) -> Result<StageReport> {
        let videos = self.get_stage_videos(&PipelineStage::Identify)?;
        if videos.is_empty() {
            return Ok(StageReport {
                stage: "identify".to_string(),
                processed: 0, succeeded: 0, failed: 0, skipped: 0,
            });
        }

        let api_key = self.openai_api_key.as_ref().cloned();
        let client = llm_client::LlmClient::new(Some(&self.llm_url), &self.llm_model, api_key)?;
        let mut succeeded = 0;
        let mut failed = 0;

        for video_id in &videos {
            if self.dry_run {
                tracing::info!("[dry-run] Would identify moments in {}", video_id);
                continue;
            }
            match identify::identify_teaching_moments(&client, &self.store, video_id).await {
                Ok(moments) => {
                    tracing::info!("Identified {} moments in {}", moments.len(), video_id);
                    self.store.mark_stage_complete(video_id, &PipelineStage::Identify)?;
                    succeeded += 1;
                }
                Err(e) => {
                    tracing::error!("Identification failed for {}: {}", video_id, e);
                    self.store.mark_stage_failed(video_id, &PipelineStage::Identify, &e.to_string())?;
                    failed += 1;
                }
            }
        }

        Ok(StageReport {
            stage: "identify".to_string(),
            processed: videos.len(), succeeded, failed, skipped: 0,
        })
    }

    fn get_stage_videos(&self, stage: &PipelineStage) -> Result<Vec<String>> {
        let mut videos = if self.force {
            self.store
                .load_videos()?
                .into_iter()
                .map(|v| v.video_id)
                .collect()
        } else {
            self.store.get_videos_needing_stage(stage)?
        };

        // Filter by piece if --piece is set
        if let Some(ref filter) = self.piece_filter {
            let filter_lower = filter.to_lowercase();
            let video_map = self.store.load_video_map()?;
            videos.retain(|id| {
                if let Some(meta) = video_map.get(id) {
                    meta.pieces.iter().any(|p| p.to_lowercase().contains(&filter_lower))
                } else {
                    false
                }
            });
        }

        if let Some(max) = self.max_videos {
            videos.truncate(max);
        }

        Ok(videos)
    }
}

pub fn print_status(store: &MasterclassStore) -> Result<()> {
    let summary = store.status_summary()?;
    println!("{}", summary);
    Ok(())
}
