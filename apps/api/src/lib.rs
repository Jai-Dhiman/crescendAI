pub mod auth;
pub mod error;
pub mod practice;
pub mod server;
pub mod services;
pub mod types;

pub use error::{ApiError, Result};
pub use types::{ConversationId, PieceId, SessionId, StudentId};

/// Truncate a string at a UTF-8 safe boundary, returning at most `max_bytes` bytes.
pub fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
