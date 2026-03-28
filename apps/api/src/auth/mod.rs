pub mod extractor;
pub mod handlers;
pub mod jwt;

pub use extractor::AuthUser;
pub use handlers::{AppleAuthRequest, AuthResponse, GoogleAuthRequest};

// Legacy re-exports for server.rs and other modules that still use old signatures.
// These will be removed when the router is rewritten in Task 6.
pub use handlers::handle_apple_auth_legacy as handle_apple_auth;
pub use handlers::handle_auth_me_legacy as handle_auth_me;
pub use handlers::handle_debug_auth_legacy as handle_debug_auth;
pub use handlers::handle_google_auth_legacy as handle_google_auth;
pub use handlers::handle_signout_legacy as handle_signout;
pub use handlers::verify_auth;
pub use handlers::verify_auth_header;
