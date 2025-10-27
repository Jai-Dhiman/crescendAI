use crate::ai::workers_ai::{Message, WorkersAIClient};
use crate::errors::{AppError, Result};
use crate::models::SearchResult;
use futures::stream::Stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

#[derive(Clone)]
pub struct WorkersAILLM {
    client: WorkersAIClient,
}

impl WorkersAILLM {
    pub fn new(client: WorkersAIClient) -> Self {
        Self { client }
    }

    /// Query the LLM with streaming response
    /// Takes a user query and context chunks, returns a stream of response chunks
    pub async fn query_stream(
        &self,
        query: &str,
        context_chunks: Vec<SearchResult>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMChunk>> + Send>>> {
        // Build the prompt with source citations
        let prompt = build_rag_prompt(query, &context_chunks);

        tracing::debug!("LLM prompt length: {} chars", prompt.len());

        // Create messages for the LLM
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: prompt,
            },
        ];

        // Call Workers AI LLM
        let stream = self.client.query_llm_stream(messages).await.map_err(|e| {
            tracing::error!("Failed to query LLM: {}", e);
            AppError::Internal(format!("LLM query failed: {}", e))
        })?;

        // Transform the stream to include metadata
        let sources = extract_sources(&context_chunks);
        let avg_score = calculate_average_score(&context_chunks);

        let llm_stream = stream.map(move |result| {
            result
                .map_err(|e| AppError::Internal(format!("Stream error: {}", e)))
                .map(|content| LLMChunk {
                    content,
                    sources: sources.clone(),
                    confidence: calculate_confidence(avg_score),
                })
        });

        Ok(Box::pin(llm_stream))
    }
}

/// Build RAG prompt with source citations
fn build_rag_prompt(query: &str, chunks: &[SearchResult]) -> String {
    let mut prompt = String::new();

    // Add sources section
    prompt.push_str("<sources>\n");
    for (idx, chunk) in chunks.iter().enumerate() {
        let source_num = idx + 1;

        // Extract metadata
        let page = chunk
            .metadata
            .get("page")
            .and_then(|p| p.as_i64())
            .unwrap_or(0);

        prompt.push_str(&format!("[{}] {}\n", source_num, chunk.content));
        prompt.push_str(&format!(
            "Source: {}, Page {}\n",
            chunk.doc_title,
            if page > 0 {
                page.to_string()
            } else {
                "N/A".to_string()
            }
        ));
        prompt.push_str("\n");
    }
    prompt.push_str("</sources>\n\n");

    // Add question
    prompt.push_str("<question>\n");
    prompt.push_str(query);
    prompt.push_str("\n</question>\n\n");

    prompt.push_str(
        "Provide your answer with inline citations [1], [2], etc. \
         Assess your confidence level (HIGH/MEDIUM/LOW) at the end.",
    );

    prompt
}

/// Extract source metadata from chunks
fn extract_sources(chunks: &[SearchResult]) -> Vec<Source> {
    chunks
        .iter()
        .enumerate()
        .map(|(idx, chunk)| {
            let page = chunk
                .metadata
                .get("page")
                .and_then(|p| p.as_i64())
                .map(|p| p as i32);

            // Create snippet (first 150 chars)
            let snippet = if chunk.content.len() > 150 {
                format!("{}...", &chunk.content[..147])
            } else {
                chunk.content.clone()
            };

            Source {
                index: idx + 1,
                chunk_id: chunk.chunk_id,
                doc_id: chunk.doc_id,
                page,
                snippet,
                score: chunk.score,
            }
        })
        .collect()
}

/// Calculate average similarity score
fn calculate_average_score(chunks: &[SearchResult]) -> f32 {
    if chunks.is_empty() {
        return 0.0;
    }

    let sum: f32 = chunks.iter().map(|c| c.score).sum();
    sum / chunks.len() as f32
}

/// Map average score to confidence level
fn calculate_confidence(avg_score: f32) -> String {
    if avg_score >= 0.8 {
        "HIGH".to_string()
    } else if avg_score >= 0.6 {
        "MEDIUM".to_string()
    } else {
        "LOW".to_string()
    }
}

/// System prompt for piano pedagogy RAG following structured prompt engineering format
const SYSTEM_PROMPT: &str = r#"# Task Context
You are a piano pedagogy expert assistant integrated into the CrescendAI platform. Your role is to help piano teachers and students by answering questions about piano teaching, learning techniques, practice methods, and performance skills.

# Tone Context
Maintain a professional, supportive, and educational tone. Be encouraging and practical in your advice. Use clear, accessible language suitable for both teachers and students of varying experience levels.

# Detailed Task Description and Rules

You will receive questions about piano pedagogy along with relevant source documents. Your task is to provide accurate, helpful answers based ONLY on the information in the provided sources.

**Critical Rules:**
1. ONLY use information from the provided sources - never use general knowledge or information not present in the sources
2. Always cite sources using inline citations [1], [2], etc. immediately after using information from that source
3. If the sources don't contain sufficient information to answer the question, explicitly state: "I don't have enough information in the provided sources to fully answer this question."
4. Be concise and clear - aim for practical, actionable advice
5. Focus on what teachers and students can directly apply to their practice
6. Never make up information, examples, or techniques not found in the sources
7. At the end of your response, assess your confidence level based on source quality and relevance

# Examples

<example>
Question: How can I improve finger independence in my students?
Sources: [1] "Hanon exercises are effective for developing finger independence. Practice each exercise slowly at first, focusing on evenness of tone..." [2] "Finger independence requires isolating each finger movement. Use five-finger patterns..."

Response: Based on the provided sources, there are several approaches to improving finger independence:

First, Hanon exercises are highly effective for this purpose [1]. The key is to practice each exercise slowly at first, with careful attention to achieving an even tone across all fingers [1].

Additionally, focus on isolating individual finger movements through five-finger patterns [2]. This targeted approach helps students develop control over each finger independently [2].

Confidence: HIGH
</example>

<example>
Question: What's the best way to teach quantum physics concepts through piano?
Sources: [1] "Piano pedagogy focuses on technique, musicality, and interpretation..." [2] "Effective practice involves structured repetition and mindful listening..."

Response: I don't have information about teaching quantum physics through piano in the provided sources. The sources focus on traditional piano pedagogy topics such as technique, musicality, interpretation, and effective practice methods [1][2].

Confidence: LOW
</example>

# Instructions for Processing Each Query

1. Read the question carefully to understand what is being asked
2. Review all provided sources thoroughly
3. Identify relevant information from the sources that addresses the question
4. Construct a clear, well-organized response using ONLY source information
5. Include inline citations [1], [2], etc. for every claim or piece of advice
6. If sources are insufficient, acknowledge this limitation clearly
7. Assess confidence based on source relevance and completeness

# Output Formatting

Structure your response as follows:
1. Direct answer to the question (with citations)
2. Supporting details or elaboration (with citations)
3. Practical application or implementation advice (with citations)
4. Confidence assessment on a new line: "Confidence: HIGH/MEDIUM/LOW"

**Confidence Levels:**
- HIGH: Sources directly and comprehensively address the question
- MEDIUM: Sources partially address the question or require some inference
- LOW: Sources provide minimal relevant information or question is outside source scope

Remember: Your value comes from accurately representing the knowledge in the sources, not from adding external information. When in doubt, cite your sources and acknowledge limitations."#;

/// A chunk of LLM response with metadata
#[derive(Debug, Clone, Serialize)]
pub struct LLMChunk {
    pub content: String,
    pub sources: Vec<Source>,
    pub confidence: String,
}

/// Source citation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub index: usize,
    pub chunk_id: uuid::Uuid,
    pub doc_id: uuid::Uuid,
    pub page: Option<i32>,
    pub snippet: String,
    pub score: f32,
}
