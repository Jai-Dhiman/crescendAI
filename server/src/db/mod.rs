// CrescendAI Server - D1 Database Query Helpers
// Common database utilities and traits

use worker::*;
use wasm_bindgen::JsCast;

pub mod sessions;
pub mod messages;
pub mod recordings;
pub mod knowledge;
pub mod analysis;
pub mod context;

// Re-export commonly used types
pub use sessions::{ChatSession, create_session, get_session, list_sessions_by_user, delete_session};
pub use messages::{ChatMessage, insert_message, get_messages_by_session, get_messages_paginated};
pub use recordings::{Recording, insert_recording, get_recording, list_recordings_by_user, update_recording_status};
pub use knowledge::{KnowledgeDocument, KnowledgeChunk, insert_document, insert_chunk, get_chunks_by_ids, search_chunks_fulltext};
pub use analysis::{AnalysisResultRecord, insert_analysis_result, get_analysis_result, delete_analysis_result};
pub use context::{UserContext, upsert_context, get_context, get_context_or_default};

// Common error type for database operations
#[derive(Debug)]
pub enum DbError {
    NotFound(String),
    InvalidInput(String),
    DatabaseError(String),
    SerializationError(String),
}

impl std::fmt::Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DbError::NotFound(msg) => write!(f, "Not found: {}", msg),
            DbError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            DbError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            DbError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for DbError {}

pub type DbResult<T> = std::result::Result<T, DbError>;

// Helper function to get current timestamp in milliseconds
pub fn current_timestamp_ms() -> i64 {
    js_sys::Date::now() as i64
}

// Helper function to generate UUID v4
pub fn generate_id() -> String {
    use worker::wasm_bindgen::JsValue;

    // Use crypto.randomUUID() if available
    let window = js_sys::global();
    let crypto = js_sys::Reflect::get(&window, &JsValue::from_str("crypto")).ok();

    if let Some(crypto) = crypto {
        if let Ok(uuid) = js_sys::Reflect::get(&crypto, &JsValue::from_str("randomUUID")) {
            if let Ok(uuid_fn) = uuid.dyn_into::<js_sys::Function>() {
                if let Ok(uuid_result) = uuid_fn.call0(&crypto) {
                    if let Some(uuid_str) = uuid_result.as_string() {
                        return uuid_str;
                    }
                }
            }
        }
    }

    // Fallback to simple UUID generation
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        (js_sys::Math::random() * 0xFFFFFFFFu32 as f64) as u32,
        (js_sys::Math::random() * 0xFFFFu16 as f64) as u16,
        (js_sys::Math::random() * 0xFFFFu16 as f64) as u16,
        (js_sys::Math::random() * 0xFFFFu16 as f64) as u16,
        (js_sys::Math::random() * 0xFFFFFFFFFFFFu64 as f64) as u64,
    )
}
