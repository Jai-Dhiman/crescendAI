use crate::errors::{AppError, Result};
use tiktoken_rs::cl100k_base;

/// A chunk of text with metadata
#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub start_char: usize,
    pub end_char: usize,
    pub page_number: Option<i32>,
}

/// Configuration for text chunking
pub struct ChunkConfig {
    pub chunk_size: usize,     // Tokens per chunk
    pub overlap: usize,         // Token overlap between chunks
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 128,
        }
    }
}

/// Chunk text into overlapping segments based on token count
pub fn chunk_text(text: &str, config: &ChunkConfig, page_number: Option<i32>) -> Result<Vec<Chunk>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    // Get tiktoken tokenizer (cl100k_base is used by GPT-4 and compatible models)
    let bpe = cl100k_base()
        .map_err(|e| AppError::Internal(format!("Failed to load tokenizer: {}", e)))?;

    // Tokenize the entire text
    let tokens = bpe.encode_with_special_tokens(text);

    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    let mut chunks = Vec::new();
    let mut start_idx = 0;

    while start_idx < tokens.len() {
        // Calculate end index for this chunk
        let end_idx = (start_idx + config.chunk_size).min(tokens.len());

        // Extract tokens for this chunk
        let chunk_tokens = &tokens[start_idx..end_idx];

        // Decode tokens back to text
        let chunk_text = bpe.decode(chunk_tokens.to_vec())
            .map_err(|e| AppError::Internal(format!("Failed to decode tokens: {}", e)))?;

        // Find the character positions in the original text
        // This is an approximation since we're working with tokens
        let start_char = estimate_char_position(text, &bpe, start_idx, &tokens);
        let end_char = estimate_char_position(text, &bpe, end_idx, &tokens);

        chunks.push(Chunk {
            content: chunk_text,
            start_char,
            end_char,
            page_number,
        });

        // Move to next chunk with overlap
        if end_idx >= tokens.len() {
            break;
        }

        start_idx += config.chunk_size - config.overlap;
    }

    Ok(chunks)
}

/// Estimate character position in original text based on token index
/// This is an approximation since tiktoken works with bytes
fn estimate_char_position(
    text: &str,
    bpe: &tiktoken_rs::CoreBPE,
    token_idx: usize,
    all_tokens: &[usize],
) -> usize {
    if token_idx == 0 {
        return 0;
    }
    if token_idx >= all_tokens.len() {
        return text.len();
    }

    // Decode up to this token index and count characters
    let partial_tokens = &all_tokens[..token_idx];
    match bpe.decode(partial_tokens.to_vec()) {
        Ok(partial_text) => partial_text.len(),
        Err(_) => {
            // Fallback: estimate based on token ratio
            (text.len() * token_idx) / all_tokens.len()
        }
    }
}

/// Chunk multiple pages of text
pub fn chunk_pages(
    pages: Vec<crate::ingestion::extractors::PageText>,
    config: &ChunkConfig,
) -> Result<Vec<Chunk>> {
    let mut all_chunks = Vec::new();

    for page in pages {
        let mut page_chunks = chunk_text(&page.text, config, Some(page.page_number))?;
        all_chunks.append(&mut page_chunks);
    }

    Ok(all_chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_empty_text() {
        let config = ChunkConfig::default();
        let result = chunk_text("", &config, None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_chunk_short_text() {
        let config = ChunkConfig {
            chunk_size: 10,
            overlap: 3,
        };
        let text = "This is a short test text.";
        let result = chunk_text(text, &config, Some(1)).unwrap();

        // Should create at least one chunk
        assert!(!result.is_empty());
        assert_eq!(result[0].page_number, Some(1));
    }

    #[test]
    fn test_chunk_overlap() {
        let config = ChunkConfig {
            chunk_size: 20,
            overlap: 5,
        };
        // Create a longer text that will require multiple chunks
        let text = "word ".repeat(100);
        let result = chunk_text(&text, &config, None).unwrap();

        // Should create multiple chunks
        assert!(result.len() > 1);

        // Verify chunks have content
        for chunk in &result {
            assert!(!chunk.content.is_empty());
        }
    }
}
