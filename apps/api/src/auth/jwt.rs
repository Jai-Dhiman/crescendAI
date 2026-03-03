use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Claims {
    pub sub: String,
    pub iat: u64,
    pub exp: u64,
}

pub fn sign(claims: &Claims, secret: &[u8]) -> Result<String, String> {
    let header = r#"{"alg":"HS256","typ":"JWT"}"#;
    let header_b64 = URL_SAFE_NO_PAD.encode(header.as_bytes());

    let claims_json =
        serde_json::to_string(claims).map_err(|e| format!("Failed to serialize claims: {}", e))?;
    let claims_b64 = URL_SAFE_NO_PAD.encode(claims_json.as_bytes());

    let signing_input = format!("{}.{}", header_b64, claims_b64);

    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|e| format!("Invalid HMAC key: {}", e))?;
    mac.update(signing_input.as_bytes());
    let signature = mac.finalize().into_bytes();
    let sig_b64 = URL_SAFE_NO_PAD.encode(&signature);

    Ok(format!("{}.{}", signing_input, sig_b64))
}

pub fn verify(token: &str, secret: &[u8]) -> Result<Claims, String> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err("Invalid JWT format".to_string());
    }

    let signing_input = format!("{}.{}", parts[0], parts[1]);

    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|e| format!("Invalid HMAC key: {}", e))?;
    mac.update(signing_input.as_bytes());

    let expected_sig = URL_SAFE_NO_PAD
        .decode(parts[2])
        .map_err(|e| format!("Invalid signature encoding: {}", e))?;

    mac.verify_slice(&expected_sig)
        .map_err(|_| "Invalid signature".to_string())?;

    let claims_json = URL_SAFE_NO_PAD
        .decode(parts[1])
        .map_err(|e| format!("Invalid claims encoding: {}", e))?;

    let claims: Claims = serde_json::from_slice(&claims_json)
        .map_err(|e| format!("Invalid claims JSON: {}", e))?;

    let now = js_sys::Date::now() as u64 / 1000;
    if claims.exp < now {
        return Err("Token expired".to_string());
    }

    Ok(claims)
}
