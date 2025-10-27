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
        AccessLevel, AccessResponse, CreateProjectRequest,
        CreateProjectResponse, GrantAccessRequest, Project, ProjectAccess,
        ProjectAccessListResponse, ProjectAccessWithUser, ProjectWithAccess,
        UpdateProjectRequest, UserRole,
    },
    state::AppState,
};

/// Query parameters for listing projects
#[derive(Debug, Deserialize)]
pub struct ListProjectsQuery {
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

/// Confirm upload request - sent after client uploads to R2
#[derive(Debug, Deserialize)]
pub struct ConfirmUploadRequest {
    pub file_size_bytes: i64,
    pub page_count: Option<i32>,
}

/// Create a new project
/// NOTE: This endpoint only creates the database record.
/// The Cloudflare Worker layer intercepts this request and:
/// 1. Generates an R2 presigned upload URL
/// 2. Forwards the request to this endpoint
/// 3. Returns {project, upload_url} to the client
pub async fn create_project(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Json(payload): Json<CreateProjectRequest>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Validate title
    if payload.title.trim().is_empty() {
        return Err(AppError::BadRequest("Title cannot be empty".to_string()));
    }

    // Validate filename
    if !payload.filename.ends_with(".pdf") {
        return Err(AppError::BadRequest(
            "Only PDF files are supported".to_string(),
        ));
    }

    let project_id = Uuid::new_v4();

    // R2 key format: projects/{user_id}/{project_id}/{filename}
    let r2_key = format!("projects/{}/{}/{}", user_id, project_id, payload.filename);

    // Insert project into database
    sqlx::query!(
        r#"
        INSERT INTO projects (id, owner_id, title, description, r2_bucket, r2_key)
        VALUES ($1, $2, $3, $4, $5, $6)
        "#,
        project_id,
        user_id,
        payload.title.trim(),
        payload.description.as_deref().map(|d| d.trim()),
        "piano-pdfs", // Bucket name (Worker has the actual binding)
        r2_key
    )
    .execute(&state.pool)
    .await?;

    // Fetch the created project
    let project = sqlx::query_as!(
        Project,
        r#"
        SELECT id, owner_id, title, description, r2_bucket, r2_key,
               file_size_bytes, page_count, created_at, updated_at
        FROM projects
        WHERE id = $1
        "#,
        project_id
    )
    .fetch_one(&state.pool)
    .await?;

    // Return project (Worker will add upload_url)
    Ok(Json(CreateProjectResponse { project }))
}

/// Confirm upload after client has uploaded to R2
/// This endpoint updates the project metadata after the client
/// has successfully uploaded the PDF to R2 using the presigned URL.
pub async fn confirm_upload(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(project_id): Path<Uuid>,
    Json(payload): Json<ConfirmUploadRequest>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Check if project exists and user is owner
    let project = sqlx::query_as!(
        Project,
        r#"
        SELECT id, owner_id, title, description, r2_bucket, r2_key,
               file_size_bytes, page_count, created_at, updated_at
        FROM projects
        WHERE id = $1
        "#,
        project_id
    )
    .fetch_optional(&state.pool)
    .await?
    .ok_or(AppError::NotFound("Project not found".to_string()))?;

    // Verify ownership
    if project.owner_id != user_id {
        return Err(AppError::Forbidden(
            "Only the project owner can confirm uploads".to_string(),
        ));
    }

    // Update project with file metadata
    sqlx::query!(
        r#"
        UPDATE projects
        SET file_size_bytes = $1, page_count = $2, updated_at = NOW()
        WHERE id = $3
        "#,
        payload.file_size_bytes,
        payload.page_count,
        project_id
    )
    .execute(&state.pool)
    .await?;

    Ok(StatusCode::OK)
}

/// List projects accessible to the user
pub async fn list_projects(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Query(query): Query<ListProjectsQuery>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;
    let limit = query.limit.unwrap_or(50).min(100);
    let offset = query.offset.unwrap_or(0);

    let projects = sqlx::query_as!(
        ProjectWithAccess,
        r#"
        SELECT
            p.id, p.owner_id, p.title, p.description, p.r2_bucket, p.r2_key,
            p.file_size_bytes, p.page_count, p.created_at, p.updated_at,
            COALESCE(pa.access_level, 'admin') as "access_level!: AccessLevel"
        FROM projects p
        LEFT JOIN project_access pa ON p.id = pa.project_id AND pa.user_id = $1
        WHERE p.owner_id = $1 OR pa.user_id = $1
        ORDER BY p.updated_at DESC
        LIMIT $2 OFFSET $3
        "#,
        user_id,
        limit,
        offset
    )
    .fetch_all(&state.pool)
    .await?;

    Ok(Json(projects))
}

/// Get a specific project
/// NOTE: The Cloudflare Worker intercepts this response and adds a presigned download_url
pub async fn get_project(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(project_id): Path<Uuid>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Check access
    check_project_access(&state.pool, project_id, user_id, AccessLevel::View).await?;

    // Get project with access level
    let project = sqlx::query_as!(
        ProjectWithAccess,
        r#"
        SELECT
            p.id, p.owner_id, p.title, p.description, p.r2_bucket, p.r2_key,
            p.file_size_bytes, p.page_count, p.created_at, p.updated_at,
            CASE
                WHEN p.owner_id = $2 THEN 'admin'::access_level
                ELSE COALESCE(pa.access_level, 'view'::access_level)
            END as "access_level!: AccessLevel"
        FROM projects p
        LEFT JOIN project_access pa ON p.id = pa.project_id AND pa.user_id = $2
        WHERE p.id = $1
        "#,
        project_id,
        user_id
    )
    .fetch_optional(&state.pool)
    .await?
    .ok_or(AppError::NotFound("Project not found".to_string()))?;

    // Return project (Worker will add download_url)
    Ok(Json(project))
}

/// Update project metadata
pub async fn update_project(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(project_id): Path<Uuid>,
    Json(payload): Json<UpdateProjectRequest>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Check edit access
    check_project_access(&state.pool, project_id, user_id, AccessLevel::Edit).await?;

    // Update project
    let project = sqlx::query_as!(
        Project,
        r#"
        UPDATE projects
        SET title = COALESCE($2, title),
            description = COALESCE($3, description),
            updated_at = NOW()
        WHERE id = $1
        RETURNING id, owner_id, title, description, r2_bucket, r2_key,
                  file_size_bytes, page_count, created_at, updated_at
        "#,
        project_id,
        payload.title.as_deref(),
        payload.description
    )
    .fetch_one(&state.pool)
    .await?;

    Ok(Json(project))
}

/// Delete a project
/// NOTE: The Cloudflare Worker intercepts this request and:
/// 1. Forwards to this endpoint to delete from database
/// 2. Deletes the file from R2 using direct binding
pub async fn delete_project(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(project_id): Path<Uuid>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Get project to verify ownership and get R2 key for Worker cleanup
    let project = sqlx::query_as!(
        Project,
        r#"
        SELECT id, owner_id, title, description, r2_bucket, r2_key,
               file_size_bytes, page_count, created_at, updated_at
        FROM projects
        WHERE id = $1
        "#,
        project_id
    )
    .fetch_optional(&state.pool)
    .await?
    .ok_or(AppError::NotFound("Project not found".to_string()))?;

    // Only owner can delete
    if project.owner_id != user_id {
        return Err(AppError::Forbidden(
            "Only the owner can delete a project".to_string(),
        ));
    }

    // Delete from database (cascades to annotations and access)
    sqlx::query!(
        r#"
        DELETE FROM projects WHERE id = $1
        "#,
        project_id
    )
    .execute(&state.pool)
    .await?;

    // Worker handles R2 file deletion
    Ok(StatusCode::NO_CONTENT)
}

/// Grant access to a project
pub async fn grant_access(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(project_id): Path<Uuid>,
    Json(payload): Json<GrantAccessRequest>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Check if user has admin access or is owner
    check_project_access(&state.pool, project_id, user_id, AccessLevel::Admin).await?;

    // Verify target user exists
    let target_user = sqlx::query!(
        r#"SELECT id FROM users WHERE id = $1"#,
        payload.user_id
    )
    .fetch_optional(&state.pool)
    .await?
    .ok_or(AppError::NotFound("User not found".to_string()))?;

    // Grant access (upsert)
    let access = sqlx::query_as!(
        ProjectAccess,
        r#"
        INSERT INTO project_access (project_id, user_id, access_level, granted_by)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (project_id, user_id)
        DO UPDATE SET access_level = $3
        RETURNING project_id, user_id, access_level as "access_level: AccessLevel", created_at
        "#,
        project_id,
        payload.user_id,
        payload.access_level as AccessLevel,
        user_id
    )
    .fetch_one(&state.pool)
    .await?;

    Ok(Json(AccessResponse { access }))
}

/// List users with access to a project
pub async fn list_project_access(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(project_id): Path<Uuid>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Check if user has access to the project
    check_project_access(&state.pool, project_id, user_id, AccessLevel::View).await?;

    // Get all access records with user details
    let access_list = sqlx::query_as!(
        ProjectAccessWithUser,
        r#"
        SELECT
            pa.project_id,
            pa.user_id,
            pa.access_level as "access_level: AccessLevel",
            u.email,
            u.full_name,
            u.role as "role: UserRole"
        FROM project_access pa
        JOIN users u ON pa.user_id = u.id
        WHERE pa.project_id = $1
        ORDER BY pa.created_at DESC
        "#,
        project_id
    )
    .fetch_all(&state.pool)
    .await?;

    Ok(Json(ProjectAccessListResponse { access_list }))
}

/// Revoke access to a project
pub async fn revoke_access(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path((project_id, target_user_id)): Path<(Uuid, Uuid)>,
) -> Result<impl IntoResponse> {
    let user_id = Uuid::parse_str(&claims.sub)?;

    // Check if user has admin access
    check_project_access(&state.pool, project_id, user_id, AccessLevel::Admin).await?;

    // Cannot revoke owner's access
    let project = sqlx::query!("SELECT owner_id FROM projects WHERE id = $1", project_id)
        .fetch_optional(&state.pool)
        .await?
        .ok_or(AppError::NotFound("Project not found".to_string()))?;

    if project.owner_id == target_user_id {
        return Err(AppError::BadRequest(
            "Cannot revoke owner's access".to_string(),
        ));
    }

    // Revoke access
    let result = sqlx::query!(
        r#"
        DELETE FROM project_access
        WHERE project_id = $1 AND user_id = $2
        "#,
        project_id,
        target_user_id
    )
    .execute(&state.pool)
    .await?;

    if result.rows_affected() == 0 {
        return Err(AppError::NotFound("Access record not found".to_string()));
    }

    Ok(StatusCode::NO_CONTENT)
}
