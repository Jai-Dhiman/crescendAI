use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    auth::{authz::check_project_access, jwt::JwtClaims},
    errors::{AppError, Result},
    models::{
        AccessLevel, Annotation, AnnotationType, CreateAnnotationRequest, UpdateAnnotationRequest,
    },
    state::AppState,
};

/// Query parameters for listing annotations
#[derive(Debug, Deserialize)]
pub struct ListAnnotationsQuery {
    pub project_id: Uuid,
    pub page_number: Option<i32>,
}

/// Create a new annotation
pub async fn create_annotation(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Json(payload): Json<CreateAnnotationRequest>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Validate the request
    payload.validate().map_err(AppError::BadRequest)?;

    // Check edit access to the project
    check_project_access(
        &state.pool,
        user_id,
        payload.project_id,
        AccessLevel::Edit,
    )
    .await?;

    // Verify project exists
    let project_exists = sqlx::query_scalar::<_, bool>(
        r#"
        SELECT EXISTS(SELECT 1 FROM projects WHERE id = $1)
        "#,
    )
    .bind(payload.project_id)
    .fetch_one(&state.pool)
    .await?;

    if !project_exists {
        return Err(AppError::NotFound("Project not found".to_string()));
    }

    // Validate page number against project's page count
    let page_count = sqlx::query_scalar::<_, Option<i32>>(
        r#"
        SELECT page_count FROM projects WHERE id = $1
        "#,
    )
    .bind(payload.project_id)
    .fetch_one(&state.pool)
    .await?;

    if let Some(count) = page_count {
        if payload.page_number > count {
            return Err(AppError::BadRequest(format!(
                "Page number {} exceeds document page count {}",
                payload.page_number, count
            )));
        }
    }

    // Create annotation
    let annotation = sqlx::query_as::<_, Annotation>(
        r#"
        INSERT INTO annotations (project_id, user_id, page_number, annotation_type, content)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING *
        "#,
    )
    .bind(payload.project_id)
    .bind(user_id)
    .bind(payload.page_number)
    .bind(payload.annotation_type)
    .bind(payload.content)
    .fetch_one(&state.pool)
    .await?;

    tracing::info!(
        "Created {:?} annotation {} on project {} page {}",
        annotation.annotation_type,
        annotation.id,
        payload.project_id,
        payload.page_number
    );

    Ok((StatusCode::CREATED, Json(annotation)))
}

/// List annotations for a project (optionally filtered by page)
pub async fn list_annotations(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Query(query): Query<ListAnnotationsQuery>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Check view access to the project
    check_project_access(&state.pool, user_id, query.project_id, AccessLevel::View).await?;

    // Query annotations
    let annotations = if let Some(page) = query.page_number {
        // Filter by page number
        sqlx::query_as::<_, Annotation>(
            r#"
            SELECT * FROM annotations
            WHERE project_id = $1 AND page_number = $2
            ORDER BY created_at ASC
            "#,
        )
        .bind(query.project_id)
        .bind(page)
        .fetch_all(&state.pool)
        .await?
    } else {
        // Get all annotations for the project
        sqlx::query_as::<_, Annotation>(
            r#"
            SELECT * FROM annotations
            WHERE project_id = $1
            ORDER BY page_number ASC, created_at ASC
            "#,
        )
        .bind(query.project_id)
        .fetch_all(&state.pool)
        .await?
    };

    Ok(Json(annotations))
}

/// Get a specific annotation
pub async fn get_annotation(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Get annotation
    let annotation = sqlx::query_as::<_, Annotation>(
        r#"
        SELECT * FROM annotations WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_optional(&state.pool)
    .await?
    .ok_or_else(|| AppError::NotFound("Annotation not found".to_string()))?;

    // Check view access to the project
    check_project_access(
        &state.pool,
        user_id,
        annotation.project_id,
        AccessLevel::View,
    )
    .await?;

    Ok(Json(annotation))
}

/// Update an annotation
pub async fn update_annotation(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(payload): Json<UpdateAnnotationRequest>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Get annotation
    let annotation = sqlx::query_as::<_, Annotation>(
        r#"
        SELECT * FROM annotations WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_optional(&state.pool)
    .await?
    .ok_or_else(|| AppError::NotFound("Annotation not found".to_string()))?;

    // Verify user owns the annotation OR has edit access to the project
    let has_access = if annotation.user_id == user_id {
        true
    } else {
        check_project_access(
            &state.pool,
            user_id,
            annotation.project_id,
            AccessLevel::Edit,
        )
        .await
        .is_ok()
    };

    if !has_access {
        return Err(AppError::Forbidden(
            "Cannot edit another user's annotation without project edit access".to_string(),
        ));
    }

    // Validate content for the annotation type
    payload
        .validate(annotation.annotation_type)
        .map_err(AppError::BadRequest)?;

    // Update annotation
    let updated = sqlx::query_as::<_, Annotation>(
        r#"
        UPDATE annotations
        SET content = $1, updated_at = NOW()
        WHERE id = $2
        RETURNING *
        "#,
    )
    .bind(payload.content)
    .bind(id)
    .fetch_one(&state.pool)
    .await?;

    tracing::info!("Updated annotation {}", id);

    Ok(Json(updated))
}

/// Delete an annotation
pub async fn delete_annotation(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Get annotation
    let annotation = sqlx::query_as::<_, Annotation>(
        r#"
        SELECT * FROM annotations WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_optional(&state.pool)
    .await?
    .ok_or_else(|| AppError::NotFound("Annotation not found".to_string()))?;

    // Verify user owns the annotation OR has admin access to the project
    let has_access = if annotation.user_id == user_id {
        true
    } else {
        check_project_access(
            &state.pool,
            user_id,
            annotation.project_id,
            AccessLevel::Admin,
        )
        .await
        .is_ok()
    };

    if !has_access {
        return Err(AppError::Forbidden(
            "Cannot delete another user's annotation without project admin access".to_string(),
        ));
    }

    // Delete annotation
    sqlx::query(
        r#"
        DELETE FROM annotations WHERE id = $1
        "#,
    )
    .bind(id)
    .execute(&state.pool)
    .await?;

    tracing::info!("Deleted annotation {}", id);

    Ok(StatusCode::NO_CONTENT)
}
