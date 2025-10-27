use crate::errors::{AppError, Result};
use crate::models::SearchResult;
use sqlx::PgPool;
use uuid::Uuid;

/// User context for filtering search results
pub struct UserContext {
    pub user_id: Uuid,
    pub is_public_only: bool,
}

/// Perform vector similarity search using HNSW index
pub async fn vector_search(
    pool: &PgPool,
    query_embedding: Vec<f32>,
    user_context: &UserContext,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    // Convert embedding to pgvector format string
    let embedding_str = format!(
        "[{}]",
        query_embedding
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    // Perform vector similarity search with user filtering
    // Using cosine distance operator: <=>
    // Note: We need to set hnsw.ef_search for optimal recall
    sqlx::query("SET hnsw.ef_search = 40")
        .execute(pool)
        .await?;

    let results = sqlx::query_as::<_, (Uuid, Uuid, String, String, f32, serde_json::Value)>(
        r#"
        SELECT
            c.id as chunk_id,
            c.doc_id,
            kb.title as doc_title,
            c.content,
            (c.embedding <=> $1::vector) as score,
            c.metadata
        FROM document_chunks c
        JOIN knowledge_base_docs kb ON c.doc_id = kb.id
        WHERE
            (kb.is_public = true OR kb.owner_id = $2
             OR EXISTS (
                 SELECT 1 FROM teacher_student_relationships tsr
                 WHERE tsr.student_id = $2 AND tsr.teacher_id = kb.owner_id
             ))
            AND kb.status = 'completed'
        ORDER BY c.embedding <=> $1::vector
        LIMIT $3
        "#,
    )
    .bind(&embedding_str)
    .bind(user_context.user_id)
    .bind(limit as i64)
    .fetch_all(pool)
    .await?;

    Ok(results
        .into_iter()
        .map(|(chunk_id, doc_id, doc_title, content, score, metadata)| SearchResult {
            chunk_id,
            doc_id,
            doc_title,
            content,
            score,
            metadata,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_context_creation() {
        let ctx = UserContext {
            user_id: Uuid::new_v4(),
            is_public_only: false,
        };
        assert!(!ctx.is_public_only);
    }
}
