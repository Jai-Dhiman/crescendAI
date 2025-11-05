//! User Context Handler
//!
//! Manages user context (goals, constraints, repertoire) for personalized feedback.

use worker::*;
use serde::{Deserialize, Serialize};
use crate::db::context;

// ============================================================================
// Constants
// ============================================================================

const MAX_GOALS_LENGTH: usize = 500;
const MAX_CONSTRAINTS_LENGTH: usize = 500;
const MAX_REPERTOIRE_ITEMS: usize = 50;
const MAX_REPERTOIRE_ITEM_LENGTH: usize = 200;

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct UpdateContextRequest {
    pub goals: Option<Vec<String>>,
    pub constraints: Option<Vec<String>>,
    pub repertoire: Option<Vec<String>>,
    pub experience_level: Option<String>,
    pub preferred_feedback_style: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ContextResponse {
    pub user_id: String,
    pub goals: Option<Vec<String>>,
    pub constraints: Option<Vec<String>>,
    pub repertoire: Option<Vec<String>>,
    pub experience_level: Option<String>,
    pub preferred_feedback_style: Option<serde_json::Value>,
    pub created_at: i64,
    pub updated_at: i64,
}

// ============================================================================
// Handlers
// ============================================================================

/// PUT /api/v1/context - Update user context
pub async fn update_context_handler(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    console_log!("Update context request received");

    // Get user_id from header
    let user_id = req.headers()
        .get("X-User-Id")?
        .unwrap_or_else(|| "default_user".to_string());

    // Parse request body
    let body: UpdateContextRequest = match req.json().await {
        Ok(body) => body,
        Err(e) => {
            console_error!("Failed to parse request body: {:?}", e);
            return Response::error("Invalid request body", 400);
        }
    };

    // Validate context
    if let Err(msg) = validate_context(&body) {
        console_error!("Validation failed: {}", msg);
        return Response::error(&msg, 400);
    }

    // Upsert context in database
    let context = match context::upsert_context(
        &ctx.env,
        &user_id,
        body.goals.clone(),
        body.constraints.clone(),
        body.repertoire.clone(),
        body.experience_level.as_deref(),
        body.preferred_feedback_style.clone(),
    ).await {
        Ok(ctx) => ctx,
        Err(e) => {
            console_error!("Failed to upsert context: {:?}", e);
            return Response::error("Failed to save context", 500);
        }
    };

    console_log!("Context updated successfully for user_id={}", user_id);

    // Return response
    let response = ContextResponse {
        user_id: context.user_id,
        goals: context.goals,
        constraints: context.constraints,
        repertoire: context.repertoire,
        experience_level: context.experience_level,
        preferred_feedback_style: context.preferred_feedback_style,
        created_at: context.created_at,
        updated_at: context.updated_at,
    };

    Response::from_json(&response)
}

/// GET /api/v1/context - Get user context
pub async fn get_context_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    console_log!("Get context request received");

    // Get user_id from header
    let user_id = req.headers()
        .get("X-User-Id")?
        .unwrap_or_else(|| "default_user".to_string());

    // Get context from database (or default)
    let context = context::get_context_or_default(&ctx.env, &user_id).await;

    console_log!("Context retrieved for user_id={}", user_id);

    // Return response
    let response = ContextResponse {
        user_id: context.user_id,
        goals: context.goals,
        constraints: context.constraints,
        repertoire: context.repertoire,
        experience_level: context.experience_level,
        preferred_feedback_style: context.preferred_feedback_style,
        created_at: context.created_at,
        updated_at: context.updated_at,
    };

    Response::from_json(&response)
}

// ============================================================================
// Validation
// ============================================================================

fn validate_context(ctx: &UpdateContextRequest) -> std::result::Result<(), String> {
    // Validate goals
    if let Some(ref goals) = ctx.goals {
        for goal in goals {
            if goal.len() > MAX_GOALS_LENGTH {
                return Err(format!(
                    "Each goal must be at most {} characters",
                    MAX_GOALS_LENGTH
                ));
            }
        }
    }

    // Validate constraints
    if let Some(ref constraints) = ctx.constraints {
        for constraint in constraints {
            if constraint.len() > MAX_CONSTRAINTS_LENGTH {
                return Err(format!(
                    "Each constraint must be at most {} characters",
                    MAX_CONSTRAINTS_LENGTH
                ));
            }
        }
    }

    // Validate repertoire
    if let Some(ref repertoire) = ctx.repertoire {
        if repertoire.len() > MAX_REPERTOIRE_ITEMS {
            return Err(format!(
                "Repertoire cannot have more than {} items",
                MAX_REPERTOIRE_ITEMS
            ));
        }

        for piece in repertoire {
            if piece.len() > MAX_REPERTOIRE_ITEM_LENGTH {
                return Err(format!(
                    "Each repertoire item must be at most {} characters",
                    MAX_REPERTOIRE_ITEM_LENGTH
                ));
            }
        }
    }

    // Validate experience level
    if let Some(ref level) = ctx.experience_level {
        if !["beginner", "intermediate", "advanced", "professional"].contains(&level.as_str()) {
            return Err(
                "Experience level must be one of: beginner, intermediate, advanced, professional"
                    .to_string(),
            );
        }
    }

    Ok(())
}
