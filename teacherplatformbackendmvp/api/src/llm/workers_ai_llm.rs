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

        let doc_title = "Unknown Document"; // TODO: Join with knowledge_base_docs to get title

        prompt.push_str(&format!("[{}] {}\n", source_num, chunk.content));
        prompt.push_str(&format!(
            "Source: {}, Page {}\n",
            doc_title,
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

/// System prompt for piano pedagogy RAG
const SYSTEM_PROMPT: &str = r#"You are a piano pedagogy expert assistant. Your role is to answer questions about piano teaching and learning based ONLY on the provided sources.

Rules:
1. Always cite your sources using [1], [2], etc. inline in your response
2. If the sources don't contain information to answer the question, explicitly say "I don't have information about this in the provided sources"
3. Never make up information that isn't in the sources
4. Be concise and clear in your explanations
5. Focus on practical, actionable advice for piano teachers and students
6. At the end, assess your confidence level: HIGH, MEDIUM, or LOW

Remember: You must base your answer ONLY on the provided sources. Do not use general knowledge about piano pedagogy unless it's explicitly mentioned in the sources."#;

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
