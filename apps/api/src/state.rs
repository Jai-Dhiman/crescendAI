//! Application state and service layer.
//!
//! Each service wraps `SendWrapper<Env>` and exposes typed methods.
//! Handlers access services via `State<AppState>` extractor.

use crate::error::{ApiError, Result};
use worker::send::SendWrapper;
use worker::Env;

/// Shared application state passed to all Axum handlers via `State<AppState>`.
#[derive(Clone)]
pub struct AppState {
    pub auth: AuthService,
    pub db: DbService,
    pub inference: InferenceService,
    pub practice: PracticeService,
}

impl AppState {
    pub fn from_env(env: Env) -> Self {
        let env = SendWrapper(env);
        Self {
            auth: AuthService::new(env.clone()),
            db: DbService::new(env.clone()),
            inference: InferenceService::new(env.clone()),
            practice: PracticeService::new(env),
        }
    }
}

// ---------------------------------------------------------------------------
// AuthService
// ---------------------------------------------------------------------------

/// Handles JWT operations and user management.
#[derive(Clone)]
pub struct AuthService {
    env: SendWrapper<Env>,
}

impl AuthService {
    pub fn new(env: SendWrapper<Env>) -> Self {
        Self { env }
    }

    /// Access the raw Env (for legacy code that still needs it during migration).
    pub fn env(&self) -> &Env {
        &self.env
    }

    pub fn jwt_secret(&self) -> Result<Vec<u8>> {
        self.env
            .secret("JWT_SECRET")
            .map(|s| s.to_string().into_bytes())
            .map_err(|e| ApiError::Internal(format!("JWT_SECRET: {e}")))
    }

    pub fn verify_jwt(&self, token: &str) -> Result<crate::auth::jwt::Claims> {
        let secret = self.jwt_secret()?;
        crate::auth::jwt::verify(token, &secret)
    }

    pub fn sign_jwt(&self, student_id: &str) -> Result<String> {
        let secret = self.jwt_secret()?;
        crate::auth::jwt::sign_for_student(student_id, &secret)
    }
}

// ---------------------------------------------------------------------------
// DbService
// ---------------------------------------------------------------------------

/// Wraps D1 database access.
#[derive(Clone)]
pub struct DbService {
    env: SendWrapper<Env>,
}

impl DbService {
    pub fn new(env: SendWrapper<Env>) -> Self {
        Self { env }
    }

    pub fn env(&self) -> &Env {
        &self.env
    }

    pub fn d1(&self) -> Result<worker::D1Database> {
        self.env
            .d1("DB")
            .map_err(|e| ApiError::Internal(format!("D1 binding: {e}")))
    }
}

// ---------------------------------------------------------------------------
// InferenceService
// ---------------------------------------------------------------------------

/// Wraps LLM and inference endpoint access.
#[derive(Clone)]
pub struct InferenceService {
    env: SendWrapper<Env>,
}

impl InferenceService {
    pub fn new(env: SendWrapper<Env>) -> Self {
        Self { env }
    }

    pub fn env(&self) -> &Env {
        &self.env
    }
}

// ---------------------------------------------------------------------------
// PracticeService
// ---------------------------------------------------------------------------

/// Wraps Durable Object namespace and R2 bucket access.
#[derive(Clone)]
pub struct PracticeService {
    env: SendWrapper<Env>,
}

impl PracticeService {
    pub fn new(env: SendWrapper<Env>) -> Self {
        Self { env }
    }

    pub fn env(&self) -> &Env {
        &self.env
    }

    pub fn do_namespace(&self) -> Result<worker::ObjectNamespace> {
        self.env
            .durable_object("PRACTICE_SESSION")
            .map_err(|e| ApiError::Internal(format!("DO namespace: {e}")))
    }

    pub fn chunks_bucket(&self) -> Result<worker::Bucket> {
        self.env
            .bucket("CHUNKS")
            .map_err(|e| ApiError::Internal(format!("R2 CHUNKS: {e}")))
    }

    pub fn scores_bucket(&self) -> Result<worker::Bucket> {
        self.env
            .bucket("SCORES")
            .map_err(|e| ApiError::Internal(format!("R2 SCORES: {e}")))
    }
}
