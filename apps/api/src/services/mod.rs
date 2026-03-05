mod embedding;
mod feedback;
pub mod goals;
mod huggingface;
pub mod llm;
pub mod prompts;
mod r2;
mod rag;
mod reranker;
pub mod sync;
mod vectorize;
mod vectorize_binding;

pub use embedding::{generate_embedding, generate_embeddings, EMBEDDING_DIM, EMBEDDING_MODEL};
pub use feedback::{generate_chat_response, generate_cited_feedback, generate_fallback_feedback};
pub use huggingface::{get_performance_dimensions_from_hf, HFInferenceResult};
pub use r2::{
    delete_upload, extension_from_content_type, generate_upload_key, get_audio, upload_audio,
    validate_content_type, UploadResult, ALLOWED_MIME_TYPES, MAX_FILE_SIZE,
};
pub use rag::{
    bm25_search, build_retrieval_query, hybrid_retrieve, ingest_chunk, ingest_chunks_batch,
    retrieve_for_analysis, retrieve_for_chat,
};
pub use reranker::{rerank_passages, rerank_results, RerankedItem, RERANKER_MODEL};
pub use vectorize::get_practice_tips;
pub use vectorize_binding::{
    get_vectorize_index, query_vectors, upsert_vectors, VectorMatch, VectorMetadata, VectorRecord,
};
