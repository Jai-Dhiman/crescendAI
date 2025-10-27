use crate::errors::Result;
use crate::models::SearchResult;
use crate::search::vector::UserContext;
use sqlx::PgPool;
use uuid::Uuid;

/// Perform BM25 keyword search using PostgreSQL full-text search
pub async fn bm25_search(
    pool: &PgPool,
    query_text: &str,
    user_context: &UserContext,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    // Sanitize and prepare query for full-text search
    // Convert query to tsquery format
    let tsquery = query_text
        .split_whitespace()
        .map(|word| word.trim())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" & ");

    if tsquery.is_empty() {
        return Ok(Vec::new());
    }

    // Perform full-text search with BM25-like ranking
    // PostgreSQL's ts_rank_cd provides BM25-like scoring
    let results = sqlx::query_as::<_, (Uuid, Uuid, String, String, f32, serde_json::Value)>(
        r#"
        SELECT
            c.id as chunk_id,
            c.doc_id,
            kb.title as doc_title,
            c.content,
            ts_rank_cd(to_tsvector('english', c.content), query) as score,
            c.metadata
        FROM
            document_chunks c,
            plainto_tsquery('english', $1) query
        JOIN knowledge_base_docs kb ON c.doc_id = kb.id
        WHERE
            to_tsvector('english', c.content) @@ query
            AND (kb.is_public = true OR kb.owner_id = $2
                 OR EXISTS (
                     SELECT 1 FROM teacher_student_relationships tsr
                     WHERE tsr.student_id = $2 AND tsr.teacher_id = kb.owner_id
                 ))
            AND kb.status = 'completed'
        ORDER BY score DESC
        LIMIT $3
        "#,
    )
    .bind(&query_text)
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
    fn test_tsquery_construction() {
        let query = "piano finger independence";
        let tsquery: Vec<&str> = query.split_whitespace().collect();
        assert_eq!(tsquery.len(), 3);
    }
}
