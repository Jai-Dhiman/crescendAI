use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    auth::jwt::JwtClaims,
    errors::{AppError, Result},
    ingestion::processor::process_pdf_document,
    models::{
        CreateKnowledgeRequest, CreateKnowledgeResponse, KnowledgeDoc, ProcessingStatus,
        ProcessingStatusResponse,
    },
    state::AppState,
    storage::BucketType,
};

/// List knowledge base documents with filtering
#[derive(Debug, Deserialize)]
pub struct ListKnowledgeQuery {
    pub limit: Option<i64>,
    pub offset: Option<i64>,
    pub source_type: Option<String>,
}

/// Create a new knowledge base document
pub async fn create_knowledge_doc(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Json(payload): Json<CreateKnowledgeRequest>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Validate source type
    if !["pdf", "video", "text", "web"].contains(&payload.source_type.as_str()) {
        return Err(AppError::BadRequest(
            "Invalid source_type. Must be one of: pdf, video, text, web".to_string(),
        ));
    }

    // Create database record
    let doc = sqlx::query_as::<_, KnowledgeDoc>(
        r#"
        INSERT INTO knowledge_base_docs (title, source_type, source_url, owner_id, is_public, status)
        VALUES ($1, $2, $3, $4, $5, 'pending')
        RETURNING *
        "#,
    )
    .bind(&payload.title)
    .bind(&payload.source_type)
    .bind(&payload.source_url)
    .bind(user_id)
    .bind(payload.is_public)
    .fetch_one(&state.pool)
    .await?;

    // Generate presigned R2 URL if PDF (1 hour expiry)
    let upload_url = if payload.source_type == "pdf" {
        let key = format!("knowledge/{}/{}.pdf", user_id, doc.id);
        let url = state
            .r2
            .generate_upload_url(BucketType::Knowledge, &key, 3600)
            .await?;

        // Update the document's source_url with the R2 key for later retrieval
        sqlx::query!(
            r#"
            UPDATE knowledge_base_docs
            SET source_url = $1
            WHERE id = $2
            "#,
            key,
            doc.id
        )
        .execute(&state.pool)
        .await?;

        tracing::info!(
            doc_id = %doc.id,
            user_id = %user_id,
            r2_key = %key,
            "Generated presigned upload URL for knowledge document"
        );

        Some(url)
    } else {
        None
    };

    Ok(Json(CreateKnowledgeResponse { doc, upload_url }))
}

/// List knowledge base documents
pub async fn list_knowledge_docs(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Query(query): Query<ListKnowledgeQuery>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;
    let limit = query.limit.unwrap_or(50).min(100);
    let offset = query.offset.unwrap_or(0);

    // Get docs that are either:
    // 1. Public
    // 2. Owned by the user
    // 3. Owned by the user's teachers (if user is a student)
    let docs = if let Some(source_type) = query.source_type {
        sqlx::query_as::<_, KnowledgeDoc>(
            r#"
            SELECT DISTINCT kb.*
            FROM knowledge_base_docs kb
            LEFT JOIN teacher_student_relationships tsr
                ON kb.owner_id = tsr.teacher_id AND tsr.student_id = $1
            WHERE (kb.is_public = true
                   OR kb.owner_id = $1
                   OR tsr.teacher_id IS NOT NULL)
              AND kb.source_type = $2
            ORDER BY kb.created_at DESC
            LIMIT $3 OFFSET $4
            "#,
        )
        .bind(user_id)
        .bind(&source_type)
        .bind(limit)
        .bind(offset)
        .fetch_all(&state.pool)
        .await?
    } else {
        sqlx::query_as::<_, KnowledgeDoc>(
            r#"
            SELECT DISTINCT kb.*
            FROM knowledge_base_docs kb
            LEFT JOIN teacher_student_relationships tsr
                ON kb.owner_id = tsr.teacher_id AND tsr.student_id = $1
            WHERE kb.is_public = true
                   OR kb.owner_id = $1
                   OR tsr.teacher_id IS NOT NULL
            ORDER BY kb.created_at DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(user_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(&state.pool)
        .await?
    };

    Ok(Json(docs))
}

/// Get a specific knowledge base document
pub async fn get_knowledge_doc(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    let doc = sqlx::query_as::<_, KnowledgeDoc>(
        r#"
        SELECT DISTINCT kb.*
        FROM knowledge_base_docs kb
        LEFT JOIN teacher_student_relationships tsr
            ON kb.owner_id = tsr.teacher_id AND tsr.student_id = $1
        WHERE kb.id = $2
          AND (kb.is_public = true
               OR kb.owner_id = $1
               OR tsr.teacher_id IS NOT NULL)
        "#,
    )
    .bind(user_id)
    .bind(id)
    .fetch_optional(&state.pool)
    .await?;

    match doc {
        Some(doc) => Ok(Json(doc)),
        None => Err(AppError::NotFound(
            "Knowledge base document not found or access denied".to_string(),
        )),
    }
}

/// Delete a knowledge base document
pub async fn delete_knowledge_doc(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Only owner can delete
    let result = sqlx::query(
        r#"
        DELETE FROM knowledge_base_docs
        WHERE id = $1 AND owner_id = $2
        "#,
    )
    .bind(id)
    .bind(user_id)
    .execute(&state.pool)
    .await?;

    if result.rows_affected() == 0 {
        return Err(AppError::NotFound(
            "Knowledge base document not found or access denied".to_string(),
        ));
    }

    Ok(StatusCode::NO_CONTENT)
}

/// Trigger processing for a knowledge base document
pub async fn process_knowledge_doc(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Verify ownership
    let doc = sqlx::query_as::<_, KnowledgeDoc>(
        r#"
        SELECT * FROM knowledge_base_docs
        WHERE id = $1 AND owner_id = $2
        "#,
    )
    .bind(id)
    .bind(user_id)
    .fetch_optional(&state.pool)
    .await?;

    let doc = match doc {
        Some(doc) => doc,
        None => {
            return Err(AppError::NotFound(
                "Knowledge base document not found or access denied".to_string(),
            ))
        }
    };

    // Check if already processing or completed
    if doc.status == ProcessingStatus::Processing {
        return Err(AppError::BadRequest(
            "Document is already being processed".to_string(),
        ));
    }

    // Verify we have Workers AI client for embeddings
    let workers_ai = state.workers_ai.as_ref().ok_or_else(|| {
        AppError::BadRequest("Workers AI is not configured - cannot process document".to_string())
    })?;

    // Update status to processing
    sqlx::query(
        r#"
        UPDATE knowledge_base_docs
        SET status = 'processing'
        WHERE id = $1
        "#,
    )
    .bind(id)
    .execute(&state.pool)
    .await?;

    // Spawn background task to process the document
    let pool = state.pool.clone();
    let workers_ai = workers_ai.clone();
    let r2_client = state.r2.clone();
    let source_url = doc.source_url.clone();

    tokio::spawn(async move {
        // Fetch PDF from R2
        let pdf_result = if let Some(r2_key) = source_url {
            // Download the PDF from R2
            match r2_client.download_object(BucketType::Knowledge, &r2_key).await {
                Ok(pdf_bytes) => {
                    tracing::info!(
                        doc_id = %id,
                        r2_key = %r2_key,
                        size_bytes = %pdf_bytes.len(),
                        "Fetched PDF from R2 for processing"
                    );
                    Ok(pdf_bytes)
                }
                Err(e) => {
                    tracing::error!(
                        doc_id = %id,
                        r2_key = %r2_key,
                        error = ?e,
                        "Failed to fetch PDF from R2"
                    );
                    Err(format!("Failed to fetch PDF from R2: {}", e))
                }
            }
        } else {
            tracing::error!(doc_id = %id, "No source_url found for document");
            Err("No source_url found for document".to_string())
        };

        match pdf_result {
            Ok(pdf_bytes) => {
                // Process the PDF
                match process_pdf_document(&pool, &workers_ai, id, &pdf_bytes).await {
                    Ok(chunk_count) => {
                        // Update status to completed and set total chunks
                        if let Err(e) = sqlx::query(
                            r#"
                            UPDATE knowledge_base_docs
                            SET status = 'completed', total_chunks = $2
                            WHERE id = $1
                            "#,
                        )
                        .bind(id)
                        .bind(chunk_count as i32)
                        .execute(&pool)
                        .await
                        {
                            tracing::error!("Failed to update document status to completed: {:?}", e);
                        }
                    }
                    Err(e) => {
                        // Error handling is already done in process_pdf_document
                        tracing::error!("Failed to process document {}: {:?}", id, e);
                    }
                }
            }
            Err(error_msg) => {
                // Update status to failed with error message
                if let Err(e) = sqlx::query(
                    r#"
                    UPDATE knowledge_base_docs
                    SET status = 'failed', error_message = $2
                    WHERE id = $1
                    "#,
                )
                .bind(id)
                .bind(&error_msg)
                .execute(&pool)
                .await
                {
                    tracing::error!("Failed to update document status to failed: {:?}", e);
                }
            }
        }
    });

    // Return 202 Accepted immediately
    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "message": "Processing started",
            "doc_id": id
        })),
    ))
}

/// Get processing status for a knowledge base document
pub async fn get_processing_status(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    let doc = sqlx::query_as::<_, KnowledgeDoc>(
        r#"
        SELECT DISTINCT kb.*
        FROM knowledge_base_docs kb
        LEFT JOIN teacher_student_relationships tsr
            ON kb.owner_id = tsr.teacher_id AND tsr.student_id = $1
        WHERE kb.id = $2
          AND (kb.is_public = true
               OR kb.owner_id = $1
               OR tsr.teacher_id IS NOT NULL)
        "#,
    )
    .bind(user_id)
    .bind(id)
    .fetch_optional(&state.pool)
    .await?;

    let doc = match doc {
        Some(doc) => doc,
        None => {
            return Err(AppError::NotFound(
                "Knowledge base document not found or access denied".to_string(),
            ))
        }
    };

    // Count processed chunks
    let chunk_count: (i64,) = sqlx::query_as(
        r#"
        SELECT COUNT(*) FROM document_chunks
        WHERE doc_id = $1
        "#,
    )
    .bind(id)
    .fetch_one(&state.pool)
    .await?;

    let response = ProcessingStatusResponse {
        status: doc.status.clone(),
        progress: chunk_count.0 as i32,
        total_chunks: doc.total_chunks,
        error_message: None,
    };

    Ok(Json(response))
}
