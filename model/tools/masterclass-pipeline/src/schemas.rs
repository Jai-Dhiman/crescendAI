use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VideoMetadata {
    pub video_id: String,
    pub url: String,
    pub title: String,
    pub channel: String,
    pub duration_seconds: f64,
    pub upload_date: Option<String>,
    pub description: Option<String>,
    pub teacher: Option<String>,
    pub pieces: Vec<String>,
    pub composers: Vec<String>,
    pub source: String,
    pub discovered_at: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TranscriptToken {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub probability: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TranscriptSegment {
    pub id: u32,
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub tokens: Vec<TranscriptToken>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Transcript {
    pub video_id: String,
    pub model: String,
    pub language: String,
    pub transcribed_at: String,
    pub segments: Vec<TranscriptSegment>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SegmentLabel {
    Playing,
    Talking,
    Silence,
    Mixed,
}

impl std::fmt::Display for SegmentLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SegmentLabel::Playing => write!(f, "playing"),
            SegmentLabel::Talking => write!(f, "talking"),
            SegmentLabel::Silence => write!(f, "silence"),
            SegmentLabel::Mixed => write!(f, "mixed"),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AudioSegmentLabel {
    pub start: f64,
    pub end: f64,
    pub label: SegmentLabel,
    pub confidence: f32,
    pub energy_db: f32,
    pub spectral_centroid_hz: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StoppingPoint {
    pub timestamp: f64,
    pub playing_start: f64,
    pub playing_end: f64,
    pub talking_start: f64,
    pub talking_end: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SegmentationResult {
    pub video_id: String,
    pub segments: Vec<AudioSegmentLabel>,
    pub stopping_points: Vec<StoppingPoint>,
    pub segmented_at: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TeachingMoment {
    pub moment_id: String,
    pub video_id: String,
    pub video_title: String,
    pub teacher: String,

    pub stop_timestamp: f64,
    pub feedback_start: f64,
    pub feedback_end: f64,
    pub playing_before_start: f64,
    pub playing_before_end: f64,

    pub transcript_text: String,
    pub feedback_summary: String,
    pub musical_dimension: String,
    pub secondary_dimensions: Vec<String>,
    pub severity: String,
    pub feedback_type: String,

    pub piece: Option<String>,
    pub composer: Option<String>,
    pub passage_description: Option<String>,
    pub student_level: Option<String>,

    pub stop_order: u32,
    pub total_stops: u32,
    pub time_spent_seconds: f64,
    pub demonstrated: bool,

    pub extracted_at: String,
    pub extraction_model: String,
    pub confidence: f32,

    #[serde(default)]
    pub open_description: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStage {
    Discover,
    Download,
    Transcribe,
    Segment,
    Extract,
    Identify,
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineStage::Discover => write!(f, "discover"),
            PipelineStage::Download => write!(f, "download"),
            PipelineStage::Transcribe => write!(f, "transcribe"),
            PipelineStage::Segment => write!(f, "segment"),
            PipelineStage::Extract => write!(f, "extract"),
            PipelineStage::Identify => write!(f, "identify"),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StageState {
    pub video_id: String,
    pub stage: PipelineStage,
    pub status: StageStatus,
    pub completed_at: Option<String>,
    pub error: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Completed,
    Failed,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SourcesConfig {
    #[serde(default)]
    pub channels: Vec<ChannelSource>,
    #[serde(default)]
    pub videos: Vec<VideoSource>,
    #[serde(default)]
    pub search_queries: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChannelSource {
    pub url: String,
    pub teacher: Option<String>,
    #[serde(default = "default_max_videos")]
    pub max_videos: u32,
}

fn default_max_videos() -> u32 {
    50
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VideoSource {
    pub url: String,
    pub teacher: Option<String>,
    pub piece: Option<String>,
    pub composer: Option<String>,
}
