//! CrescendAI Server - Piano Education Platform Backend
//!
//! A Cloudflare Workers-based backend combining AI-powered piano performance
//! analysis with RAG (Retrieval-Augmented Generation) for pedagogy support.
//!
//! # Architecture
//!
//! - **Runtime:** Rust on Cloudflare Workers (WASM)
//! - **AI:** Dedalus API (GPT-5-nano) via Python worker service binding
//! - **Database:** D1 (distributed SQLite)
//! - **Vector Search:** Cloudflare Vectorize
//! - **Storage:** R2 (audio files, PDFs), KV (caching)
//! - **Analysis:** MERT-330M (mocked, Modal integration planned)

use worker::*;

// ============================================================================
// Core Modules
// ============================================================================

/// Database query helpers and models
pub mod db;

/// API request/response models and Dedalus types
pub mod models;

/// HTTP handlers for API endpoints
mod handlers;

// ============================================================================
// AI & RAG System
// ============================================================================

/// Dedalus service binding client
mod dedalus_client;

/// RAG tool definitions for Dedalus
mod rag_tools;

/// Tool execution handlers
mod tool_executor;

/// Hybrid RAG search (Vectorize + D1 FTS)
mod knowledge_base;

/// Document ingestion pipeline
mod ingestion;

// ============================================================================
// Analysis & Feedback
// ============================================================================

/// Mock MERT-330M model for development
mod ast_mock;

// ============================================================================
// Infrastructure
// ============================================================================

/// Security, auth, rate limiting
mod security;

/// Health checks and monitoring
mod monitoring;

// ============================================================================
// Router & Main Handler
// ============================================================================

use handlers::{chat, feedback};
use security::{get_client_ip, RateLimiter, secure_error_response};
use monitoring::{HealthChecker, SystemInfo};

#[event(fetch)]
async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    // Initialize panic hook for better error messages
    console_error_panic_hook::set_once();

    // Get CORS headers before env is moved
    let cors_headers = get_cors_headers(&env);

    // CORS preflight
    if req.method() == Method::Options {
        return Response::empty()
            .map(|res| {
                res.with_cors(&cors_headers)
                    .expect("CORS headers")
            });
    }

    // Route the request
    let router = Router::new();

    router
        // Health & Status
        .get_async("/health", health_handler)
        .get_async("/api/health", health_handler)
        .get_async("/api/status", status_handler)

        // Chat System
        .post_async("/api/v1/chat", secure_chat_handler)
        .post_async("/api/v1/chat/sessions", secure_create_session_handler)
        .get_async("/api/v1/chat/sessions/:id", secure_get_session_handler)
        .get_async("/api/v1/chat/sessions", secure_list_sessions_handler)
        .delete_async("/api/v1/chat/sessions/:id", secure_delete_session_handler)

        // Feedback System
        .post_async("/api/v1/feedback/:id", secure_feedback_handler)

        // Catch-all
        .or_else_any_method_async("/*", not_found_handler)
        .run(req, env)
        .await
        .map(|res| res.with_cors(&cors_headers).expect("CORS"))
}

// ============================================================================
// Handler Wrappers with Security
// ============================================================================

async fn secure_chat_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Rate limiting
    let client_ip = get_client_ip(&req);
    let rate_limiter = RateLimiter::new(&ctx.env);

    if !rate_limiter.check_rate_limit(&client_ip).await? {
        return secure_error_response(
            "Rate limit exceeded. Please try again later.",
            429,
        );
    }

    chat::stream_chat(req, ctx).await
}

async fn secure_create_session_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Rate limiting
    let client_ip = get_client_ip(&req);
    let rate_limiter = RateLimiter::new(&ctx.env);

    if !rate_limiter.check_rate_limit(&client_ip).await? {
        return secure_error_response(
            "Rate limit exceeded. Please try again later.",
            429,
        );
    }

    chat::create_session(req, ctx).await
}

async fn secure_get_session_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Rate limiting
    let client_ip = get_client_ip(&req);
    let rate_limiter = RateLimiter::new(&ctx.env);

    if !rate_limiter.check_rate_limit(&client_ip).await? {
        return secure_error_response(
            "Rate limit exceeded. Please try again later.",
            429,
        );
    }

    chat::get_session(req, ctx).await
}

async fn secure_list_sessions_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Rate limiting
    let client_ip = get_client_ip(&req);
    let rate_limiter = RateLimiter::new(&ctx.env);

    if !rate_limiter.check_rate_limit(&client_ip).await? {
        return secure_error_response(
            "Rate limit exceeded. Please try again later.",
            429,
        );
    }

    chat::list_sessions(req, ctx).await
}

async fn secure_delete_session_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Rate limiting
    let client_ip = get_client_ip(&req);
    let rate_limiter = RateLimiter::new(&ctx.env);

    if !rate_limiter.check_rate_limit(&client_ip).await? {
        return secure_error_response(
            "Rate limit exceeded. Please try again later.",
            429,
        );
    }

    chat::delete_session(req, ctx).await
}

async fn secure_feedback_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Rate limiting
    let client_ip = get_client_ip(&req);
    let rate_limiter = RateLimiter::new(&ctx.env);

    if !rate_limiter.check_rate_limit(&client_ip).await? {
        return secure_error_response(
            "Rate limit exceeded. Please try again later.",
            429,
        );
    }

    feedback::generate_feedback_handler(req, ctx).await
}

// ============================================================================
// System Handlers
// ============================================================================

async fn health_handler(_req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let checker = HealthChecker::new(&ctx.env);
    let health = checker.check_health().await?;
    Response::from_json(&health)
}

async fn status_handler(_req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let info = SystemInfo::gather(&ctx.env).await?;
    Response::from_json(&info)
}

async fn not_found_handler(_req: Request, _ctx: RouteContext<()>) -> Result<Response> {
    Response::error("Not Found", 404)
}

// ============================================================================
// CORS Configuration
// ============================================================================

fn get_cors_headers(env: &Env) -> worker::Cors {
    let allowed_origins = env
        .var("ALLOWED_ORIGINS")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "http://localhost:3000,http://localhost:5173".to_string());

    let origins: Vec<&str> = allowed_origins.split(',').map(|s| s.trim()).collect();

    worker::Cors::new()
        .with_origins(origins)
        .with_methods(vec![
            Method::Get,
            Method::Post,
            Method::Put,
            Method::Delete,
            Method::Options,
        ])
        .with_allowed_headers(vec![
            "Content-Type",
            "Authorization",
            "X-API-Key",
        ])
        .with_max_age(86400)
}
