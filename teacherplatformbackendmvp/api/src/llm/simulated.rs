use crate::errors::Result;
use crate::models::SearchResult;
use async_stream::stream;
use futures::Stream;
use std::pin::Pin;
use tokio::time::{sleep, Duration};

/// Simulated LLM response chunk
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum LLMChunk {
    #[serde(rename = "token")]
    Token { content: String },
    #[serde(rename = "done")]
    Done {
        sources: Vec<SourceCitation>,
        confidence: f32,
    },
}

/// Source citation for LLM response
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceCitation {
    pub doc_id: uuid::Uuid,
    pub chunk_id: uuid::Uuid,
    pub content: String,
    pub relevance: f32,
}

/// Simulated LLM for MVP testing
pub struct SimulatedLLM;

impl SimulatedLLM {
    pub fn new() -> Self {
        Self
    }

    /// Generate a streaming response based on context
    pub fn query_stream(
        &self,
        query: &str,
        context_chunks: Vec<SearchResult>,
    ) -> Pin<Box<dyn Stream<Item = Result<LLMChunk>> + Send>> {
        let query = query.to_string();
        let chunks = context_chunks.clone();

        Box::pin(stream! {
            // Simulate Time To First Token (TTFT) of 100ms
            sleep(Duration::from_millis(100)).await;

            // Generate a response based on the query and context
            let response = Self::generate_response(&query, &chunks);

            // Stream the response word by word (simulating 50 tokens/sec)
            let words: Vec<&str> = response.split_whitespace().collect();
            for word in words {
                // Simulate inter-token delay (~20ms per token at 50 tokens/sec)
                sleep(Duration::from_millis(20)).await;

                yield Ok(LLMChunk::Token {
                    content: format!("{} ", word),
                });
            }

            // Send final chunk with sources and confidence
            let sources = Self::create_source_citations(&chunks);
            let confidence = Self::calculate_confidence(&chunks);

            yield Ok(LLMChunk::Done {
                sources,
                confidence,
            });
        })
    }

    /// Generate a response based on the query and context
    fn generate_response(query: &str, context_chunks: &[SearchResult]) -> String {
        if context_chunks.is_empty() {
            return format!(
                "I don't have enough information in my knowledge base to answer your question about '{}'. \
                 Please try rephrasing your question or ask about piano pedagogy topics that are covered in the knowledge base.",
                query
            );
        }

        // Create a simple response that references the context
        let num_sources = context_chunks.len();
        format!(
            "Based on the {} relevant source{} in the knowledge base, here's what I found about your question '{}':\n\n\
             The documents provide information related to this topic. The most relevant content suggests that this is an important aspect of piano pedagogy. \
             \n\nFor more detailed information, please refer to the source citations below.",
            num_sources,
            if num_sources > 1 { "s" } else { "" },
            query
        )
    }

    /// Create source citations from search results
    fn create_source_citations(chunks: &[SearchResult]) -> Vec<SourceCitation> {
        chunks
            .iter()
            .map(|chunk| SourceCitation {
                doc_id: chunk.doc_id,
                chunk_id: chunk.chunk_id,
                content: chunk.content.chars().take(200).collect(),
                relevance: chunk.score,
            })
            .collect()
    }

    /// Calculate confidence score based on search results
    /// Higher scores from search = higher confidence
    fn calculate_confidence(chunks: &[SearchResult]) -> f32 {
        if chunks.is_empty() {
            return 0.0;
        }

        // Average the scores and normalize to [0, 1]
        let avg_score: f32 = chunks.iter().map(|c| c.score).sum::<f32>() / chunks.len() as f32;

        // Clamp to [0.5, 0.95] for simulated responses
        // Real LLM would have more nuanced confidence calculation
        (avg_score * 0.45 + 0.5).min(0.95).max(0.5)
    }
}

impl Default for SimulatedLLM {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulated_llm_creation() {
        let llm = SimulatedLLM::new();
        assert!(std::mem::size_of_val(&llm) == 0); // Zero-sized type
    }

    #[test]
    fn test_generate_response_empty() {
        let response = SimulatedLLM::generate_response("test", &[]);
        assert!(response.contains("don't have enough information"));
    }

    #[test]
    fn test_calculate_confidence() {
        let confidence = SimulatedLLM::calculate_confidence(&[]);
        assert_eq!(confidence, 0.0);
    }
}
