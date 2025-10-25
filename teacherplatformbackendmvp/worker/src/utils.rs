use sha2::{Digest, Sha256};

pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

pub fn cache_key_embedding(query: &str) -> String {
    format!("embed:v1:{}", hash(query))
}

pub fn cache_key_search(query: &str, filters: &str) -> String {
    format!("search:v1:{}", hash(&format!("{}{}", query, filters)))
}

pub fn cache_key_llm(query: &str, context: &str) -> String {
    format!("llm:v1:{}", hash(&format!("{}{}", query, context)))
}

fn hash(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hex(bytes: &[u8]) -> String {
    hex_encode(bytes)
}
