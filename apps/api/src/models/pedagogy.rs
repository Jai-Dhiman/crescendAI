use serde::{Deserialize, Serialize};

/// Source type for pedagogical content
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    Book,
    Letter,
    Masterclass,
    Journal,
}

impl SourceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            SourceType::Book => "book",
            SourceType::Letter => "letter",
            SourceType::Masterclass => "masterclass",
            SourceType::Journal => "journal",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "book" => Some(SourceType::Book),
            "letter" => Some(SourceType::Letter),
            "masterclass" => Some(SourceType::Masterclass),
            "journal" => Some(SourceType::Journal),
            _ => None,
        }
    }
}

/// A chunk of pedagogical content with full citation metadata
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PedagogyChunk {
    pub chunk_id: String,
    pub text: String,
    pub text_with_context: String,
    pub source_type: SourceType,
    pub source_title: String,
    pub source_author: String,
    pub source_url: Option<String>,
    pub page_number: Option<i32>,
    pub section_title: Option<String>,
    pub paragraph_index: Option<i32>,
    pub char_start: Option<i32>,
    pub char_end: Option<i32>,
    pub timestamp_start: Option<f64>,
    pub timestamp_end: Option<f64>,
    pub speaker: Option<String>,
    pub composers: Vec<String>,
    pub pieces: Vec<String>,
    pub techniques: Vec<String>,
    pub ingested_at: String,
    pub source_hash: String,
}

/// Citation metadata for LLM-generated feedback
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Citation {
    pub number: i32,
    pub source_type: SourceType,
    pub title: String,
    pub author: String,
    pub url: Option<String>,
    pub page_number: Option<i32>,
    pub timestamp_start: Option<f64>,
    /// The quoted text from the source (for inline expansion)
    pub quote: Option<String>,
}

impl Citation {
    /// Create a citation from a pedagogy chunk
    pub fn from_chunk(chunk: &PedagogyChunk, number: i32) -> Self {
        // Truncate quote to reasonable length for display
        let quote = if chunk.text.len() > 300 {
            Some(format!("{}...", &chunk.text[..297]))
        } else {
            Some(chunk.text.clone())
        };

        Self {
            number,
            source_type: chunk.source_type.clone(),
            title: chunk.source_title.clone(),
            author: chunk.source_author.clone(),
            url: chunk.source_url.clone(),
            page_number: chunk.page_number,
            timestamp_start: chunk.timestamp_start,
            quote,
        }
    }

    /// Format as a footnote string for the sources footer
    pub fn format_footnote(&self) -> String {
        match self.source_type {
            SourceType::Book | SourceType::Letter | SourceType::Journal => {
                let mut s = format!("[{}] {} by {}", self.number, self.title, self.author);
                if let Some(page) = self.page_number {
                    s.push_str(&format!(", p.{}", page));
                }
                s
            }
            SourceType::Masterclass => {
                let mut s = format!("[{}] {} - {}", self.number, self.author, self.title);
                if let Some(ts) = self.timestamp_start {
                    let mins = (ts / 60.0).floor() as i32;
                    let secs = (ts % 60.0).floor() as i32;
                    s.push_str(&format!(" ({:02}:{:02})", mins, secs));
                }
                s
            }
        }
    }

    /// Generate URL with timestamp for video sources
    pub fn get_timestamped_url(&self) -> Option<String> {
        match (&self.url, self.timestamp_start) {
            (Some(url), Some(ts)) if url.contains("youtube.com") || url.contains("youtu.be") => {
                let secs = ts.floor() as i32;
                if url.contains('?') {
                    Some(format!("{}&t={}s", url, secs))
                } else {
                    Some(format!("{}?t={}s", url, secs))
                }
            }
            (Some(url), _) => Some(url.clone()),
            _ => None,
        }
    }
}

/// Cited feedback returned from RAG-augmented LLM
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CitedFeedback {
    pub html: String,
    pub plain_text: String,
    pub citations: Vec<Citation>,
}

/// Result of hybrid retrieval (BM25 + vector search with RRF)
#[derive(Clone, Debug)]
pub struct RetrievalResult {
    pub chunk: PedagogyChunk,
    pub bm25_rank: Option<usize>,
    pub vector_rank: Option<usize>,
    pub rrf_score: f64,
}
