mod feedback;
mod huggingface;
mod rag;
mod vectorize;
#[cfg(feature = "ssr")]
mod embedding;
#[cfg(feature = "ssr")]
mod reranker;
#[cfg(feature = "ssr")]
mod vectorize_binding;

pub use feedback::generate_fallback_feedback;
#[cfg(feature = "ssr")]
pub use feedback::{generate_chat_response, generate_cited_feedback};
pub use huggingface::get_performance_dimensions;
pub use rag::{bm25_search, build_retrieval_query, hybrid_retrieve, retrieve_for_analysis, retrieve_for_chat};
#[cfg(feature = "ssr")]
pub use rag::{ingest_chunk, ingest_chunks_batch};
pub use vectorize::get_practice_tips;
#[cfg(feature = "ssr")]
pub use embedding::{generate_embedding, generate_embeddings, EMBEDDING_DIM, EMBEDDING_MODEL};
#[cfg(feature = "ssr")]
pub use reranker::{rerank_passages, rerank_results, RerankedItem, RERANKER_MODEL};
#[cfg(feature = "ssr")]
pub use vectorize_binding::{
    get_vectorize_index, query_vectors, upsert_vectors, VectorMatch, VectorMetadata, VectorRecord,
};
