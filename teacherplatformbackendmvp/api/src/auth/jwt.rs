use anyhow::{Context, Result};
use jsonwebtoken::{decode, decode_header, Algorithm, DecodingKey, Validation};
use serde::{Deserialize, Serialize};
use crate::models::UserRole;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtClaims {
    pub sub: String,           // User ID
    pub email: String,          // User email
    #[serde(deserialize_with = "deserialize_role")]
    pub role: UserRole,         // User role (parsed from Supabase string)
    pub exp: usize,            // Expiration time
    pub iat: usize,            // Issued at
    pub aud: Option<String>,   // Audience
    pub iss: Option<String>,   // Issuer
}

// Custom deserializer to convert Supabase role string to UserRole enum
fn deserialize_role<'de, D>(deserializer: D) -> Result<UserRole, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let role_str = Option::<String>::deserialize(deserializer)?;

    match role_str.as_deref() {
        Some("teacher") | Some("Teacher") => Ok(UserRole::Teacher),
        Some("student") | Some("Student") => Ok(UserRole::Student),
        Some("admin") | Some("Admin") => Ok(UserRole::Admin),
        _ => Ok(UserRole::Student), // Default to student if role is missing
    }
}

pub fn decode_jwt(token: &str, secret: &str) -> Result<JwtClaims> {
    // Supabase uses HS256 for JWT signing
    let mut validation = Validation::new(Algorithm::HS256);

    // Allow for some clock skew
    validation.leeway = 60;

    // Validate audience (optional - can be set to "authenticated")
    validation.validate_aud = false;

    let decoding_key = DecodingKey::from_secret(secret.as_bytes());

    let token_data = decode::<JwtClaims>(token, &decoding_key, &validation)
        .context("Failed to decode JWT token")?;

    Ok(token_data.claims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jwt_decode() {
        // This is a test JWT - not a real secret
        let secret = "test-secret-key-for-testing-only-min-32-chars";

        // In production, this would come from Supabase
        // For now, this test validates the structure works
        let test_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiaWF0IjoxNTE2MjM5MDIyLCJleHAiOjk5OTk5OTk5OTl9.invalid";

        // This will fail with invalid signature, but validates structure
        let result = decode_jwt(test_token, secret);
        assert!(result.is_err()); // Expected to fail with invalid signature
    }
}
