// CrescendAI Server - User Context Database Queries

use worker::*;
use wasm_bindgen::JsValue;
use serde::{Deserialize, Serialize};
use super::{DbError, DbResult, current_timestamp_ms, generate_id};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    pub id: String,
    pub user_id: String,
    pub goals: Option<Vec<String>>,
    pub constraints: Option<Vec<String>>,
    pub repertoire: Option<Vec<String>>,
    pub experience_level: Option<String>, // "beginner", "intermediate", "advanced", "professional"
    pub preferred_feedback_style: Option<serde_json::Value>,
    pub created_at: i64,
    pub updated_at: i64,
}

// Insert or update user context (upsert)
pub async fn upsert_context(
    env: &Env,
    user_id: &str,
    goals: Option<Vec<String>>,
    constraints: Option<Vec<String>>,
    repertoire: Option<Vec<String>>,
    experience_level: Option<&str>,
    preferred_feedback_style: Option<serde_json::Value>,
) -> DbResult<UserContext> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    // Check if context already exists
    let existing = get_context(env, user_id).await;

    let now = current_timestamp_ms();

    if let Ok(existing_context) = existing {
        // Update existing context
        let stmt = db.prepare("
            UPDATE user_contexts
            SET goals = ?1, constraints = ?2, repertoire = ?3, experience_level = ?4,
                preferred_feedback_style = ?5, updated_at = ?6
            WHERE user_id = ?7
        ");

        let goals_json = goals.as_ref()
            .map(|g| serde_json::to_string(g).unwrap_or_else(|_| "[]".to_string()))
            .unwrap_or_else(|| "null".to_string());

        let constraints_json = constraints.as_ref()
            .map(|c| serde_json::to_string(c).unwrap_or_else(|_| "[]".to_string()))
            .unwrap_or_else(|| "null".to_string());

        let repertoire_json = repertoire.as_ref()
            .map(|r| serde_json::to_string(r).unwrap_or_else(|_| "[]".to_string()))
            .unwrap_or_else(|| "null".to_string());

        let feedback_style_json = preferred_feedback_style.as_ref()
            .map(|f| serde_json::to_string(f).unwrap_or_else(|_| "null".to_string()))
            .unwrap_or_else(|| "null".to_string());

        let query = stmt
            .bind(&[
                JsValue::from_str(&goals_json),
                JsValue::from_str(&constraints_json),
                JsValue::from_str(&repertoire_json),
                experience_level.map(|e| JsValue::from_str(e)).unwrap_or(JsValue::NULL),
                JsValue::from_str(&feedback_style_json),
                JsValue::from_f64(now as f64),
                JsValue::from_str(user_id),
            ])
            .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

        query.run().await
            .map_err(|e| DbError::DatabaseError(format!("Failed to update context: {}", e)))?;

        Ok(UserContext {
            id: existing_context.id,
            user_id: user_id.to_string(),
            goals,
            constraints,
            repertoire,
            experience_level: experience_level.map(|s| s.to_string()),
            preferred_feedback_style,
            created_at: existing_context.created_at,
            updated_at: now,
        })
    } else {
        // Insert new context
        let context_id = generate_id();

        let stmt = db.prepare("
            INSERT INTO user_contexts (id, user_id, goals, constraints, repertoire,
                                      experience_level, preferred_feedback_style,
                                      created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        ");

        let goals_json = goals.as_ref()
            .map(|g| serde_json::to_string(g).unwrap_or_else(|_| "[]".to_string()))
            .unwrap_or_else(|| "null".to_string());

        let constraints_json = constraints.as_ref()
            .map(|c| serde_json::to_string(c).unwrap_or_else(|_| "[]".to_string()))
            .unwrap_or_else(|| "null".to_string());

        let repertoire_json = repertoire.as_ref()
            .map(|r| serde_json::to_string(r).unwrap_or_else(|_| "[]".to_string()))
            .unwrap_or_else(|| "null".to_string());

        let feedback_style_json = preferred_feedback_style.as_ref()
            .map(|f| serde_json::to_string(f).unwrap_or_else(|_| "null".to_string()))
            .unwrap_or_else(|| "null".to_string());

        let query = stmt
            .bind(&[
                JsValue::from_str(&context_id),
                JsValue::from_str(user_id),
                JsValue::from_str(&goals_json),
                JsValue::from_str(&constraints_json),
                JsValue::from_str(&repertoire_json),
                experience_level.map(|e| JsValue::from_str(e)).unwrap_or(JsValue::NULL),
                JsValue::from_str(&feedback_style_json),
                JsValue::from_f64(now as f64),
                JsValue::from_f64(now as f64),
            ])
            .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

        query.run().await
            .map_err(|e| DbError::DatabaseError(format!("Failed to insert context: {}", e)))?;

        Ok(UserContext {
            id: context_id,
            user_id: user_id.to_string(),
            goals,
            constraints,
            repertoire,
            experience_level: experience_level.map(|s| s.to_string()),
            preferred_feedback_style,
            created_at: now,
            updated_at: now,
        })
    }
}

// Get user context by user_id
pub async fn get_context(env: &Env, user_id: &str) -> DbResult<UserContext> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let stmt = db.prepare("
        SELECT id, user_id, goals, constraints, repertoire, experience_level,
               preferred_feedback_style, created_at, updated_at
        FROM user_contexts
        WHERE user_id = ?1
    ");

    let query = stmt
        .bind(&[JsValue::from_str(user_id)])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    #[derive(Deserialize)]
    struct ContextRow {
        id: String,
        user_id: String,
        goals: Option<String>,
        constraints: Option<String>,
        repertoire: Option<String>,
        experience_level: Option<String>,
        preferred_feedback_style: Option<String>,
        created_at: i64,
        updated_at: i64,
    }

    let result = query.first::<ContextRow>(None).await
        .map_err(|e| DbError::DatabaseError(format!("Failed to query context: {}", e)))?;

    let row = result.ok_or_else(|| DbError::NotFound(format!("Context not found for user: {}", user_id)))?;

    // Parse JSON fields
    let goals = row.goals.and_then(|g| serde_json::from_str(&g).ok());
    let constraints = row.constraints.and_then(|c| serde_json::from_str(&c).ok());
    let repertoire = row.repertoire.and_then(|r| serde_json::from_str(&r).ok());
    let preferred_feedback_style = row.preferred_feedback_style.and_then(|f| serde_json::from_str(&f).ok());

    Ok(UserContext {
        id: row.id,
        user_id: row.user_id,
        goals,
        constraints,
        repertoire,
        experience_level: row.experience_level,
        preferred_feedback_style,
        created_at: row.created_at,
        updated_at: row.updated_at,
    })
}

// Get context or return default empty context
pub async fn get_context_or_default(env: &Env, user_id: &str) -> UserContext {
    match get_context(env, user_id).await {
        Ok(context) => context,
        Err(_) => UserContext {
            id: "default".to_string(),
            user_id: user_id.to_string(),
            goals: None,
            constraints: None,
            repertoire: None,
            experience_level: None,
            preferred_feedback_style: None,
            created_at: current_timestamp_ms(),
            updated_at: current_timestamp_ms(),
        },
    }
}
