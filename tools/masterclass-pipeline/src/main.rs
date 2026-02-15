mod audio_features;
mod llm_client;
mod config;
mod discovery;
mod download;
mod extract;
mod pipeline;
mod schemas;
mod segment;
mod store;
mod transcribe;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "masterclass-pipeline")]
#[command(about = "Extract teaching moments from piano masterclass recordings")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Data directory path
    #[arg(long, default_value = "./data", global = true)]
    data_dir: PathBuf,

    /// Whisper model name
    #[arg(long, default_value = "large-v3", global = true)]
    whisper_model: String,

    /// LLM model name (Ollama model, e.g. qwen2.5:32b, llama3.1, mistral)
    #[arg(long, default_value = "qwen2.5:32b", global = true)]
    llm_model: String,

    /// LLM server URL (OpenAI-compatible endpoint)
    #[arg(long, default_value = "http://localhost:11434", global = true)]
    llm_url: String,

    /// Re-run even if cached results exist
    #[arg(long, global = true)]
    force: bool,

    /// Limit number of videos to process
    #[arg(long, global = true)]
    max_videos: Option<usize>,

    /// Show what would be done without actually doing it
    #[arg(long, global = true)]
    dry_run: bool,

    /// Only process videos matching this piece (substring match, case-insensitive)
    #[arg(long, global = true)]
    piece: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Download Whisper model file
    Setup,

    /// Find videos from sources.yaml and/or search
    Discover,

    /// Download audio for discovered videos
    Download,

    /// Run Whisper transcription
    Transcribe,

    /// Run audio segmentation
    Segment,

    /// Run LLM extraction of teaching moments
    Extract,

    /// Run full pipeline (all stages)
    Run,

    /// Show pipeline progress
    Status,

    /// Export all teaching moments to single JSONL
    Export {
        /// Output file path
        #[arg(long, default_value = "all_moments.jsonl")]
        output: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let store = store::MasterclassStore::new(&cli.data_dir)?;

    match cli.command {
        Commands::Setup => {
            transcribe::download_model(&cli.data_dir, &cli.whisper_model).await?;
        }
        Commands::Discover => {
            let videos = discovery::discover(&store, &cli.data_dir).await?;
            tracing::info!("Discovered {} videos", videos.len());
        }
        Commands::Download => {
            let videos = get_videos(&store, &schemas::PipelineStage::Download, cli.force, cli.max_videos, cli.piece.as_deref())?;
            tracing::info!("Downloading audio for {} videos", videos.len());
            for video_id in &videos {
                if cli.dry_run {
                    tracing::info!("[dry-run] Would download {}", video_id);
                    continue;
                }
                match download::download_audio(&store, video_id, cli.force).await {
                    Ok(_) => store.mark_stage_complete(video_id, &schemas::PipelineStage::Download)?,
                    Err(e) => {
                        tracing::error!("Failed to download {}: {}", video_id, e);
                        store.mark_stage_failed(video_id, &schemas::PipelineStage::Download, &e.to_string())?;
                    }
                }
            }
        }
        Commands::Transcribe => {
            let videos = get_videos(&store, &schemas::PipelineStage::Transcribe, cli.force, cli.max_videos, cli.piece.as_deref())?;
            tracing::info!("Transcribing {} videos", videos.len());
            if !videos.is_empty() {
                let model_path = cli.data_dir.join("models").join(format!("ggml-{}.bin", cli.whisper_model));
                let ctx = transcribe::load_whisper_context(&model_path)?;
                for video_id in &videos {
                    if cli.dry_run {
                        tracing::info!("[dry-run] Would transcribe {}", video_id);
                        continue;
                    }
                    match transcribe::transcribe_video(&ctx, &store, video_id) {
                        Ok(_) => store.mark_stage_complete(video_id, &schemas::PipelineStage::Transcribe)?,
                        Err(e) => {
                            tracing::error!("Failed to transcribe {}: {}", video_id, e);
                            store.mark_stage_failed(video_id, &schemas::PipelineStage::Transcribe, &e.to_string())?;
                        }
                    }
                }
            }
        }
        Commands::Segment => {
            let videos = get_videos(&store, &schemas::PipelineStage::Segment, cli.force, cli.max_videos, cli.piece.as_deref())?;
            tracing::info!("Segmenting {} videos", videos.len());
            for video_id in &videos {
                if cli.dry_run {
                    tracing::info!("[dry-run] Would segment {}", video_id);
                    continue;
                }
                match segment::segment_video(&store, video_id) {
                    Ok(_) => store.mark_stage_complete(video_id, &schemas::PipelineStage::Segment)?,
                    Err(e) => {
                        tracing::error!("Failed to segment {}: {}", video_id, e);
                        store.mark_stage_failed(video_id, &schemas::PipelineStage::Segment, &e.to_string())?;
                    }
                }
            }
        }
        Commands::Extract => {
            let videos = get_videos(&store, &schemas::PipelineStage::Extract, cli.force, cli.max_videos, cli.piece.as_deref())?;
            tracing::info!("Extracting teaching moments from {} videos", videos.len());
            if !videos.is_empty() {
                let client = llm_client::LlmClient::new(Some(&cli.llm_url), &cli.llm_model, None)?;
                for video_id in &videos {
                    if cli.dry_run {
                        tracing::info!("[dry-run] Would extract from {}", video_id);
                        continue;
                    }
                    match extract::extract_teaching_moments(&client, &store, video_id).await {
                        Ok(moments) => {
                            tracing::info!("Extracted {} moments from {}", moments.len(), video_id);
                            store.mark_stage_complete(video_id, &schemas::PipelineStage::Extract)?;
                        }
                        Err(e) => {
                            tracing::error!("Failed to extract from {}: {}", video_id, e);
                            store.mark_stage_failed(video_id, &schemas::PipelineStage::Extract, &e.to_string())?;
                        }
                    }
                }
            }
        }
        Commands::Run => {
            let pipe = pipeline::Pipeline::new(
                store,
                cli.data_dir.clone(),
                cli.whisper_model.clone(),
                cli.llm_model.clone(),
                cli.llm_url.clone(),
                cli.force,
                cli.max_videos,
                cli.dry_run,
                cli.piece.clone(),
            );
            let report = pipe.run().await?;
            tracing::info!("{}", report);
            return Ok(());
        }
        Commands::Status => {
            pipeline::print_status(&store)?;
        }
        Commands::Export { output } => {
            let count = store.export_all_moments(&output)?;
            tracing::info!("Exported {} teaching moments to {}", count, output.display());
        }
    }

    Ok(())
}

/// Get videos to process for a given stage, respecting --force, --max-videos, and --piece.
fn get_videos(
    store: &store::MasterclassStore,
    stage: &schemas::PipelineStage,
    force: bool,
    max_videos: Option<usize>,
    piece_filter: Option<&str>,
) -> Result<Vec<String>> {
    let mut videos = if force {
        store
            .load_videos()?
            .into_iter()
            .map(|v| v.video_id)
            .collect()
    } else {
        store.get_videos_needing_stage(stage)?
    };

    // Filter by piece if --piece is set
    if let Some(filter) = piece_filter {
        let filter_lower = filter.to_lowercase();
        let video_map = store.load_video_map()?;
        videos.retain(|id| {
            if let Some(meta) = video_map.get(id) {
                meta.pieces.iter().any(|p| p.to_lowercase().contains(&filter_lower))
            } else {
                false
            }
        });
    }

    if let Some(n) = max_videos {
        videos.truncate(n);
    }
    Ok(videos)
}
