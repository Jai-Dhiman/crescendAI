use axum::{
    async_trait,
    extract::{FromRequestParts, Path, State},
    http::{request::Parts, StatusCode},
    Extension, RequestPartsExt,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::{
    auth::JwtClaims,
    errors::AppError,
    models::{AccessLevel, UserRole},
    state::AppState,
};

/// Extension for requiring specific user roles
#[derive(Debug, Clone)]
pub struct RequireRole(pub UserRole);

#[async_trait]
impl<S> FromRequestParts<S> for RequireRole
where
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let Extension(claims): Extension<JwtClaims> = parts
            .extract()
            .await
            .map_err(|_| AppError::Unauthorized("Missing authentication".to_string()))?;

        Ok(RequireRole(claims.role))
    }
}

/// Check if user has a specific role
pub async fn require_role(
    claims: &JwtClaims,
    required_role: UserRole,
) -> Result<(), AppError> {
    if claims.role != required_role && claims.role != UserRole::Admin {
        return Err(AppError::Forbidden(format!(
            "Required role: {:?}, got: {:?}",
            required_role, claims.role
        )));
    }
    Ok(())
}

/// Check if user has admin role
pub async fn require_admin(claims: &JwtClaims) -> Result<(), AppError> {
    require_role(claims, UserRole::Admin).await
}

/// Check if user has teacher role
pub async fn require_teacher(claims: &JwtClaims) -> Result<(), AppError> {
    if claims.role != UserRole::Teacher && claims.role != UserRole::Admin {
        return Err(AppError::Forbidden(
            "Teacher role required".to_string(),
        ));
    }
    Ok(())
}

/// Check if user has access to a specific project
pub async fn check_project_access(
    pool: &PgPool,
    user_id: Uuid,
    project_id: Uuid,
    required_level: AccessLevel,
) -> Result<(), AppError> {
    // Check if user is the project owner
    let is_owner = sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS(SELECT 1 FROM projects WHERE id = $1 AND owner_id = $2)"
    )
    .bind(project_id)
    .bind(user_id)
    .fetch_one(pool)
    .await
    .map_err(AppError::Database)?;

    if is_owner {
        return Ok(());
    }

    // Check project_access table
    let access = sqlx::query_as::<_, (AccessLevel,)>(
        "SELECT access_level FROM project_access WHERE project_id = $1 AND user_id = $2"
    )
    .bind(project_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await
    .map_err(AppError::Database)?;

    match access {
        Some((level,)) => {
            // Check if user has sufficient access level
            if can_access(level, required_level) {
                Ok(())
            } else {
                Err(AppError::Forbidden(format!(
                    "Insufficient project access. Required: {:?}, have: {:?}",
                    required_level, level
                )))
            }
        }
        None => Err(AppError::Forbidden(
            "No access to this project".to_string(),
        )),
    }
}

/// Alias for backwards compatibility
pub use check_project_access as require_project_access;

/// Check if granted level satisfies required level
fn can_access(granted: AccessLevel, required: AccessLevel) -> bool {
    match required {
        AccessLevel::View => true, // Any level can view
        AccessLevel::Edit => matches!(granted, AccessLevel::Edit | AccessLevel::Admin),
        AccessLevel::Admin => matches!(granted, AccessLevel::Admin),
    }
}

/// Check if user has a teacher-student relationship
pub async fn require_teacher_student_relationship(
    pool: &PgPool,
    teacher_id: Uuid,
    student_id: Uuid,
) -> Result<(), AppError> {
    let exists = sqlx::query_scalar::<_, bool>(
        r#"
        SELECT EXISTS(
            SELECT 1 FROM teacher_student_relationships
            WHERE teacher_id = $1 AND student_id = $2
        )
        "#,
    )
    .bind(teacher_id)
    .bind(student_id)
    .fetch_one(pool)
    .await
    .map_err(AppError::Database)?;

    if exists {
        Ok(())
    } else {
        Err(AppError::Forbidden(
            "No teacher-student relationship exists".to_string(),
        ))
    }
}

/// Check if user is teacher of a specific student
pub async fn is_teacher_of_student(
    pool: &PgPool,
    teacher_id: Uuid,
    student_id: Uuid,
) -> Result<bool, AppError> {
    let exists = sqlx::query_scalar::<_, bool>(
        r#"
        SELECT EXISTS(
            SELECT 1 FROM teacher_student_relationships
            WHERE teacher_id = $1 AND student_id = $2
        )
        "#,
    )
    .bind(teacher_id)
    .bind(student_id)
    .fetch_one(pool)
    .await
    .map_err(AppError::Database)?;

    Ok(exists)
}

/// Check if user can access content based on ownership and relationships
pub async fn can_access_content(
    pool: &PgPool,
    user_id: Uuid,
    user_role: UserRole,
    content_owner_id: Option<Uuid>,
    is_public: bool,
) -> Result<bool, AppError> {
    // Public content is accessible to everyone
    if is_public {
        return Ok(true);
    }

    // No owner means base content (accessible to all)
    let Some(owner_id) = content_owner_id else {
        return Ok(true);
    };

    // Owner can always access their own content
    if user_id == owner_id {
        return Ok(true);
    }

    // Admins can access everything
    if user_role == UserRole::Admin {
        return Ok(true);
    }

    // Students can access their teacher's content
    if user_role == UserRole::Student {
        let has_relationship = is_teacher_of_student(pool, owner_id, user_id).await?;
        return Ok(has_relationship);
    }

    // Teachers can only access their own content (already checked above)
    Ok(false)
}
