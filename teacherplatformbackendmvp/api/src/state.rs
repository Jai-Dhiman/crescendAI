use crate::{config::Config, db::DbPool};

#[derive(Clone)]
pub struct AppState {
    pub pool: DbPool,
    pub config: Config,
}

impl AppState {
    pub fn new(pool: DbPool, config: Config) -> Self {
        Self { pool, config }
    }
}
