use crate::{ai::workers_ai::WorkersAIClient, cache::CacheService, config::Config, db::DbPool};

/// Application state shared across all handlers
/// This API server focuses on database operations and business logic.
#[derive(Clone)]
pub struct AppState {
    pub pool: DbPool,
    pub config: Config,
    pub workers_ai: Option<WorkersAIClient>,
    pub cache: CacheService,
}

impl AppState {
    pub fn new(
        pool: DbPool,
        config: Config,
        workers_ai: Option<WorkersAIClient>,
        cache: CacheService,
    ) -> Self {
        Self {
            pool,
            config,
            workers_ai,
            cache,
        }
    }
}
