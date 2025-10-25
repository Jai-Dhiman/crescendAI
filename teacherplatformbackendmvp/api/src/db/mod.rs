use sqlx::postgres::{PgPool, PgPoolOptions};
use std::time::Duration;

pub type DbPool = PgPool;

/// Creates a PostgreSQL connection pool with optimized settings
pub async fn create_pool(database_url: &str, max_connections: u32) -> Result<DbPool, sqlx::Error> {
    tracing::info!(
        "Connecting to database with max_connections={}",
        max_connections
    );

    let pool = PgPoolOptions::new()
        .max_connections(max_connections)
        .min_connections(2)
        .acquire_timeout(Duration::from_secs(5))
        .idle_timeout(Duration::from_secs(600))
        .max_lifetime(Duration::from_secs(1800))
        .connect(database_url)
        .await?;

    // Test the connection
    sqlx::query("SELECT 1")
        .execute(&pool)
        .await?;

    tracing::info!("Database connection pool created successfully");

    Ok(pool)
}

/// Runs database health check
pub async fn health_check(pool: &DbPool) -> Result<(), sqlx::Error> {
    sqlx::query("SELECT 1")
        .execute(pool)
        .await?;
    Ok(())
}
