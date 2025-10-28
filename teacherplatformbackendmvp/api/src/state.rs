use crate::{ai::workers_ai::WorkersAIClient, cache::CacheService, config::Config, db::DbPool, storage::R2Client};

/// Application state shared across all handlers
/// This API server handles database operations, business logic, and R2 presigned URL generation.
#[derive(Clone)]
pub struct AppState {
    pub pool: DbPool,
    pub config: Config,
    pub workers_ai: Option<WorkersAIClient>,
    pub cache: CacheService,
    pub r2: R2Client,
}

impl AppState {
    pub fn new(
        pool: DbPool,
        config: Config,
        workers_ai: Option<WorkersAIClient>,
        cache: CacheService,
        r2: R2Client,
    ) -> Self {
        Self {
            pool,
            config,
            workers_ai,
            cache,
            r2,
        }
    }
}
