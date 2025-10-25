pub mod chunker;
pub mod extractors;
pub mod embedder;
pub mod processor;

pub use chunker::{Chunk, ChunkConfig, chunk_text, chunk_pages};
pub use extractors::{PageText, extract_pdf_text, extract_youtube_transcript, extract_web_content};
pub use embedder::{generate_embeddings, store_chunks};
pub use processor::process_pdf_document;
