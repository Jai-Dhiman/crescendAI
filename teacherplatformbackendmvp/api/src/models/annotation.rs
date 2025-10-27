use serde::{Deserialize, Serialize};
use sqlx::types::chrono::{DateTime, Utc};
use uuid::Uuid;

/// Annotation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "annotation_type", rename_all = "lowercase")]
pub enum AnnotationType {
    Highlight,
    Note,
    Drawing,
}

/// Annotation on a PDF page
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Annotation {
    pub id: Uuid,
    pub project_id: Uuid,
    pub user_id: Uuid,
    pub page_number: i32,
    pub annotation_type: AnnotationType,
    pub content: serde_json::Value, // JSONB content
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Request to create an annotation
#[derive(Debug, Deserialize)]
pub struct CreateAnnotationRequest {
    pub project_id: Uuid,
    pub page_number: i32,
    pub annotation_type: AnnotationType,
    pub content: serde_json::Value,
}

/// Request to update an annotation
#[derive(Debug, Deserialize)]
pub struct UpdateAnnotationRequest {
    pub content: serde_json::Value,
}

/// Highlight annotation content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighlightContent {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub color: String,
    pub text: Option<String>,
}

/// Note annotation content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteContent {
    pub x: f64,
    pub y: f64,
    pub text: String,
    pub color: String,
}

/// Drawing annotation content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawingContent {
    pub paths: Vec<Point>,
    pub color: String,
    pub stroke_width: f64,
}

/// Point in a drawing path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl CreateAnnotationRequest {
    /// Validate that the content matches the annotation type
    pub fn validate(&self) -> Result<(), String> {
        // Validate page number
        if self.page_number < 1 {
            return Err("Page number must be positive".to_string());
        }

        // Validate content structure based on type
        match self.annotation_type {
            AnnotationType::Highlight => {
                serde_json::from_value::<HighlightContent>(self.content.clone())
                    .map_err(|e| format!("Invalid highlight content: {}", e))?;
            }
            AnnotationType::Note => {
                serde_json::from_value::<NoteContent>(self.content.clone())
                    .map_err(|e| format!("Invalid note content: {}", e))?;
            }
            AnnotationType::Drawing => {
                serde_json::from_value::<DrawingContent>(self.content.clone())
                    .map_err(|e| format!("Invalid drawing content: {}", e))?;
            }
        }

        Ok(())
    }
}

impl UpdateAnnotationRequest {
    /// Validate content structure for a given annotation type
    pub fn validate(&self, annotation_type: AnnotationType) -> Result<(), String> {
        match annotation_type {
            AnnotationType::Highlight => {
                serde_json::from_value::<HighlightContent>(self.content.clone())
                    .map_err(|e| format!("Invalid highlight content: {}", e))?;
            }
            AnnotationType::Note => {
                serde_json::from_value::<NoteContent>(self.content.clone())
                    .map_err(|e| format!("Invalid note content: {}", e))?;
            }
            AnnotationType::Drawing => {
                serde_json::from_value::<DrawingContent>(self.content.clone())
                    .map_err(|e| format!("Invalid drawing content: {}", e))?;
            }
        }

        Ok(())
    }
}
