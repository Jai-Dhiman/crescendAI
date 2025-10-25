use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    auth::{require_teacher, JwtClaims},
    errors::AppError,
    state::AppState,
};

/// Create a new teacher-student relationship
pub async fn create_relationship(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Json(payload): Json<CreateRelationshipRequest>,
) -> Result<impl IntoResponse, AppError> {
    // Only teachers can create relationships
    require_teacher(&claims).await?;

    let teacher_id = uuid::Uuid::parse_str(&claims.sub)
        .map_err(|_| AppError::Unauthorized("Invalid user ID in token".to_string()))?;

    // Prevent teacher from adding themselves as student
    if teacher_id == payload.student_id {
        return Err(AppError::BadRequest(
            "Cannot create relationship with yourself".to_string(),
        ));
    }

    // Verify student exists and has student role
    let student_role = sqlx::query_scalar::<_, String>(
        "SELECT role::text FROM users WHERE id = $1"
    )
    .bind(payload.student_id)
    .fetch_optional(&state.pool)
    .await
    .map_err(AppError::Database)?
    .ok_or_else(|| AppError::NotFound("Student not found".to_string()))?;

    if student_role.to_lowercase() != "student" {
        return Err(AppError::BadRequest(
            "Target user is not a student".to_string(),
        ));
    }

    // Create relationship
    let relationship = sqlx::query_as::<_, Relationship>(
        r#"
        INSERT INTO teacher_student_relationships (teacher_id, student_id)
        VALUES ($1, $2)
        ON CONFLICT (teacher_id, student_id) DO NOTHING
        RETURNING id, teacher_id, student_id, created_at
        "#,
    )
    .bind(teacher_id)
    .bind(payload.student_id)
    .fetch_one(&state.pool)
    .await
    .map_err(AppError::Database)?;

    Ok((StatusCode::CREATED, Json(relationship)))
}

/// List all relationships for the current user
pub async fn list_relationships(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    let user_id = uuid::Uuid::parse_str(&claims.sub)
        .map_err(|_| AppError::Unauthorized("Invalid user ID in token".to_string()))?;

    let relationships = match claims.role {
        crate::models::UserRole::Teacher => {
            // Teachers see their students
            sqlx::query_as::<_, RelationshipWithUser>(
                r#"
                SELECT
                    r.id,
                    r.teacher_id,
                    r.student_id,
                    r.created_at,
                    u.email as user_email,
                    u.full_name as user_full_name,
                    'student' as user_role
                FROM teacher_student_relationships r
                JOIN users u ON u.id = r.student_id
                WHERE r.teacher_id = $1
                ORDER BY r.created_at DESC
                "#,
            )
            .bind(user_id)
            .fetch_all(&state.pool)
            .await
            .map_err(AppError::Database)?
        }
        crate::models::UserRole::Student => {
            // Students see their teachers
            sqlx::query_as::<_, RelationshipWithUser>(
                r#"
                SELECT
                    r.id,
                    r.teacher_id,
                    r.student_id,
                    r.created_at,
                    u.email as user_email,
                    u.full_name as user_full_name,
                    'teacher' as user_role
                FROM teacher_student_relationships r
                JOIN users u ON u.id = r.teacher_id
                WHERE r.student_id = $1
                ORDER BY r.created_at DESC
                "#,
            )
            .bind(user_id)
            .fetch_all(&state.pool)
            .await
            .map_err(AppError::Database)?
        }
        crate::models::UserRole::Admin => {
            // Admins see all relationships
            sqlx::query_as::<_, RelationshipWithUser>(
                r#"
                SELECT
                    r.id,
                    r.teacher_id,
                    r.student_id,
                    r.created_at,
                    u.email as user_email,
                    u.full_name as user_full_name,
                    'student' as user_role
                FROM teacher_student_relationships r
                JOIN users u ON u.id = r.student_id
                ORDER BY r.created_at DESC
                "#,
            )
            .fetch_all(&state.pool)
            .await
            .map_err(AppError::Database)?
        }
    };

    Ok(Json(relationships))
}

/// Delete a relationship
pub async fn delete_relationship(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
    Path(relationship_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let user_id = uuid::Uuid::parse_str(&claims.sub)
        .map_err(|_| AppError::Unauthorized("Invalid user ID in token".to_string()))?;

    // Get relationship to verify permissions
    let relationship = sqlx::query_as::<_, Relationship>(
        "SELECT id, teacher_id, student_id, created_at FROM teacher_student_relationships WHERE id = $1"
    )
    .bind(relationship_id)
    .fetch_optional(&state.pool)
    .await
    .map_err(AppError::Database)?
    .ok_or_else(|| AppError::NotFound("Relationship not found".to_string()))?;

    // Only the teacher or admin can delete the relationship
    if relationship.teacher_id != user_id && claims.role != crate::models::UserRole::Admin {
        return Err(AppError::Forbidden(
            "Only the teacher can delete this relationship".to_string(),
        ));
    }

    // Delete the relationship
    sqlx::query("DELETE FROM teacher_student_relationships WHERE id = $1")
        .bind(relationship_id)
        .execute(&state.pool)
        .await
        .map_err(AppError::Database)?;

    Ok(StatusCode::NO_CONTENT)
}

// Request/Response types
#[derive(Debug, Deserialize)]
pub struct CreateRelationshipRequest {
    pub student_id: Uuid,
}

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct Relationship {
    pub id: Uuid,
    pub teacher_id: Uuid,
    pub student_id: Uuid,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct RelationshipWithUser {
    pub id: Uuid,
    pub teacher_id: Uuid,
    pub student_id: Uuid,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub user_email: String,
    pub user_full_name: String,
    pub user_role: String,
}
