use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::error::{ApiError, Result};
use crate::types::StudentId;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Claims {
    pub sub: String,
    pub iat: u64,
    pub exp: u64,
}

pub fn sign(claims: &Claims, secret: &[u8]) -> Result<String> {
    let header = r#"{"alg":"HS256","typ":"JWT"}"#;
    let header_b64 = URL_SAFE_NO_PAD.encode(header.as_bytes());

    let claims_json = serde_json::to_string(claims)
        .map_err(|e| ApiError::Internal(format!("JWT claims serialize: {e}")))?;
    let claims_b64 = URL_SAFE_NO_PAD.encode(claims_json.as_bytes());

    let signing_input = format!("{header_b64}.{claims_b64}");

    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|e| ApiError::Internal(format!("HMAC key: {e}")))?;
    mac.update(signing_input.as_bytes());
    let signature = mac.finalize().into_bytes();
    let sig_b64 = URL_SAFE_NO_PAD.encode(signature);

    Ok(format!("{signing_input}.{sig_b64}"))
}

pub fn verify(token: &str, secret: &[u8]) -> Result<Claims> {
    let parts: Vec<&str> = token.split('.').collect();
    let [header, payload, signature] = parts.as_slice() else {
        return Err(ApiError::Unauthorized);
    };

    let signing_input = format!("{header}.{payload}");

    let mut mac = HmacSha256::new_from_slice(secret).map_err(|_| ApiError::Unauthorized)?;
    mac.update(signing_input.as_bytes());

    let expected_sig = URL_SAFE_NO_PAD
        .decode(signature)
        .map_err(|_| ApiError::Unauthorized)?;

    mac.verify_slice(&expected_sig)
        .map_err(|_| ApiError::Unauthorized)?;

    let claims_json = URL_SAFE_NO_PAD
        .decode(payload)
        .map_err(|_| ApiError::Unauthorized)?;

    let claims: Claims =
        serde_json::from_slice(&claims_json).map_err(|_| ApiError::Unauthorized)?;

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let now = (js_sys::Date::now() / 1000.0) as u64;
    if claims.exp < now {
        return Err(ApiError::Unauthorized);
    }

    Ok(claims)
}

/// Sign a JWT for a student with 30-day expiry.
pub fn sign_for_student(student_id: &StudentId, secret: &[u8]) -> Result<String> {
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let now = (js_sys::Date::now() / 1000.0) as u64;
    let claims = Claims {
        sub: student_id.as_str().to_string(),
        iat: now,
        exp: now + 30 * 24 * 60 * 60, // 30 days
    };
    sign(&claims, secret)
}
