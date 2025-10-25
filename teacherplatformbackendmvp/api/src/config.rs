use anyhow::{Context, Result};
use serde::Deserialize;
use std::env;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub jwt: JwtConfig,
    pub cloudflare: CloudflareConfig,
    pub performance: PerformanceConfig,
    pub cache: CacheConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JwtConfig {
    pub secret: String,
    pub access_token_expiry: i64,  // seconds
    pub refresh_token_expiry: i64, // seconds
}

#[derive(Debug, Clone, Deserialize)]
pub struct PerformanceConfig {
    pub request_timeout_seconds: u64,
    pub rag_request_timeout_seconds: u64,
    pub max_upload_size_mb: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CloudflareConfig {
    pub account_id: Option<String>,
    pub r2_endpoint: Option<String>,
    pub r2_access_key_id: Option<String>,
    pub r2_secret_access_key: Option<String>,
    pub r2_bucket_pdfs: String,
    pub r2_bucket_knowledge: String,
    pub workers_ai_api_token: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CacheConfig {
    pub embedding_cache_ttl_hours: u64,
    pub search_cache_ttl_hours: u64,
    pub llm_cache_ttl_hours: u64,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        // Load .env file if it exists (for local development)
        dotenvy::dotenv().ok();

        let config = Config {
            server: ServerConfig {
                host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: env::var("PORT")
                    .unwrap_or_else(|_| "8080".to_string())
                    .parse()
                    .context("Failed to parse PORT")?,
            },
            database: DatabaseConfig {
                url: env::var("DATABASE_URL")
                    .context("DATABASE_URL must be set")?,
                max_connections: env::var("DATABASE_MAX_CONNECTIONS")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .context("Failed to parse DATABASE_MAX_CONNECTIONS")?,
            },
            jwt: JwtConfig {
                secret: env::var("SUPABASE_JWT_SECRET")
                    .context("SUPABASE_JWT_SECRET must be set")?,
                access_token_expiry: env::var("JWT_ACCESS_TOKEN_EXPIRY")
                    .unwrap_or_else(|_| "3600".to_string())
                    .parse()
                    .context("Failed to parse JWT_ACCESS_TOKEN_EXPIRY")?,
                refresh_token_expiry: env::var("JWT_REFRESH_TOKEN_EXPIRY")
                    .unwrap_or_else(|_| "604800".to_string())
                    .parse()
                    .context("Failed to parse JWT_REFRESH_TOKEN_EXPIRY")?,
            },
            cloudflare: CloudflareConfig {
                account_id: env::var("CLOUDFLARE_ACCOUNT_ID").ok(),
                r2_endpoint: env::var("CLOUDFLARE_R2_ENDPOINT").ok(),
                r2_access_key_id: env::var("CLOUDFLARE_R2_ACCESS_KEY_ID").ok(),
                r2_secret_access_key: env::var("CLOUDFLARE_R2_SECRET_ACCESS_KEY").ok(),
                r2_bucket_pdfs: env::var("CLOUDFLARE_R2_BUCKET_PDFS")
                    .unwrap_or_else(|_| "piano-pdfs".to_string()),
                r2_bucket_knowledge: env::var("CLOUDFLARE_R2_BUCKET_KNOWLEDGE")
                    .unwrap_or_else(|_| "piano-knowledge".to_string()),
                workers_ai_api_token: env::var("CLOUDFLARE_WORKERS_AI_API_TOKEN").ok(),
            },
            performance: PerformanceConfig {
                request_timeout_seconds: env::var("REQUEST_TIMEOUT_SECONDS")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()
                    .context("Failed to parse REQUEST_TIMEOUT_SECONDS")?,
                rag_request_timeout_seconds: env::var("RAG_REQUEST_TIMEOUT_SECONDS")
                    .unwrap_or_else(|_| "30".to_string())
                    .parse()
                    .context("Failed to parse RAG_REQUEST_TIMEOUT_SECONDS")?,
                max_upload_size_mb: env::var("MAX_UPLOAD_SIZE_MB")
                    .unwrap_or_else(|_| "50".to_string())
                    .parse()
                    .context("Failed to parse MAX_UPLOAD_SIZE_MB")?,
            },
            cache: CacheConfig {
                embedding_cache_ttl_hours: env::var("EMBEDDING_CACHE_TTL_HOURS")
                    .unwrap_or_else(|_| "24".to_string())
                    .parse()
                    .context("Failed to parse EMBEDDING_CACHE_TTL_HOURS")?,
                search_cache_ttl_hours: env::var("SEARCH_CACHE_TTL_HOURS")
                    .unwrap_or_else(|_| "1".to_string())
                    .parse()
                    .context("Failed to parse SEARCH_CACHE_TTL_HOURS")?,
                llm_cache_ttl_hours: env::var("LLM_CACHE_TTL_HOURS")
                    .unwrap_or_else(|_| "24".to_string())
                    .parse()
                    .context("Failed to parse LLM_CACHE_TTL_HOURS")?,
            },
        };

        // Validate JWT secret length (minimum 32 characters for security)
        if config.jwt.secret.len() < 32 {
            anyhow::bail!("JWT_SECRET must be at least 32 characters long");
        }

        Ok(config)
    }

    pub fn database_url(&self) -> &str {
        &self.database.url
    }

    pub fn server_address(&self) -> String {
        format!("{}:{}", self.server.host, self.server.port)
    }
}
