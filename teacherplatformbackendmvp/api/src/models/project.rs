use serde::{Deserialize, Serialize};
use sqlx::types::chrono::{DateTime, Utc};
use uuid::Uuid;

use super::{AccessLevel, UserRole};

/// Project (PDF document with annotations)
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Project {
    pub id: Uuid,
    pub owner_id: Uuid,
    pub title: String,
    pub description: Option<String>,

    // R2 storage (metadata only - Worker handles actual R2 operations)
    pub r2_bucket: String,
    pub r2_key: String,

    // PDF metadata (populated after upload confirmation)
    pub file_size_bytes: Option<i64>,
    pub page_count: Option<i32>,

    // Timestamps
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Project access control
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ProjectAccess {
    pub project_id: Uuid,
    pub user_id: Uuid,
    pub access_level: AccessLevel,
    pub created_at: DateTime<Utc>,
}

/// Request to create a new project
#[derive(Debug, Deserialize)]
pub struct CreateProjectRequest {
    pub title: String,
    pub description: Option<String>,
    pub filename: String, // Original filename for generating R2 key
}

/// Response after creating a project
/// Includes presigned upload URL for client direct upload to R2
#[derive(Debug, Serialize)]
pub struct CreateProjectResponse {
    pub project: Project,
    pub upload_url: String,
}

/// Request to update project metadata
#[derive(Debug, Deserialize)]
pub struct UpdateProjectRequest {
    pub title: Option<String>,
    pub description: Option<String>,
}

/// Project with access level information
#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct ProjectWithAccess {
    // Project fields
    pub id: Uuid,
    pub owner_id: Uuid,
    pub title: String,
    pub description: Option<String>,
    pub r2_bucket: String,
    pub r2_key: String,
    pub file_size_bytes: Option<i64>,
    pub page_count: Option<i32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,

    // Access level for current user
    pub access_level: AccessLevel,
}

/// Project with access level and download URL
#[derive(Debug, Serialize)]
pub struct ProjectWithAccessAndDownload {
    #[serde(flatten)]
    pub project: ProjectWithAccess,
    pub download_url: String,
}

/// PDF metadata extracted from uploaded file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfMetadata {
    pub page_count: i32,
    pub file_size_bytes: i64,
    pub title: Option<String>,
}

/// Request to grant access to a project
#[derive(Debug, Deserialize)]
pub struct GrantAccessRequest {
    pub user_id: Uuid,
    pub access_level: AccessLevel,
}

/// Response with access information
#[derive(Debug, Serialize)]
pub struct AccessResponse {
    pub access: ProjectAccess,
}

/// Project access with user information
#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct ProjectAccessWithUser {
    pub project_id: Uuid,
    pub user_id: Uuid,
    pub access_level: AccessLevel,
    pub email: String,
    pub full_name: String,
    pub role: UserRole,
}

/// List of project access records
#[derive(Debug, Serialize)]
pub struct ProjectAccessListResponse {
    pub access_list: Vec<ProjectAccessWithUser>,
}
