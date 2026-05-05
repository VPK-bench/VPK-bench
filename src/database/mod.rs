//! Database layer for VPK
//!
//! VPK uses TWO databases:
//! 1. Bob's Database (PostgreSQL with pgvector) - Plaintext data, keys, mappings
//! 2. Eve's Vector DB (FAISS) - Encrypted vectors (handled in vectordb module)

pub mod queries;

use crate::error::{VPKError, VPKResult};
use deadpool_postgres::{Manager, Pool};
use tokio_postgres::NoTls;

/// Bob's database configuration (PostgreSQL)
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub connection_string: String,
    pub max_connections: u32,
    pub min_connections: u32,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            connection_string: "postgresql://localhost/vpk".to_string(),
            max_connections: 10,
            min_connections: 2,
        }
    }
}

/// Bob's Database - Stores plaintext documents, keys, and mappings
///
/// This is the TRUSTED database that Bob controls. It stores:
/// - Plaintext documents
/// - Encryption keys (dimensional scrambling parameters)
/// - Index mappings (doc_id ↔ scrambled_id)
/// - Embedding cache (optional)
/// - Query logs and audit trails
pub struct Database {
    pool: Pool,
}

impl Database {
    /// Create a new database connection to Bob's PostgreSQL
    pub async fn new(config: &DatabaseConfig) -> VPKResult<Self> {
        let pg_config: tokio_postgres::Config = config
            .connection_string
            .parse()
            .map_err(|e| VPKError::DatabaseError(format!("Invalid connection string: {}", e)))?;

        let manager = Manager::new(pg_config, NoTls);
        let pool = Pool::builder(manager)
            .max_size(config.max_connections as usize)
            .build()
            .map_err(|e| {
                VPKError::DatabaseError(format!("Failed to create connection pool: {}", e))
            })?;

        Ok(Self { pool })
    }

    /// Run database migrations
    ///
    /// Executes schema SQL files using IF NOT EXISTS guards — safe to re-run on startup.
    pub async fn migrate(&self) -> VPKResult<()> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Failed to get connection for migration: {}", e))
        })?;

        // Migration 001: initial schema (documents, keys, index, audit log)
        let sql1 = include_str!("../../migrations/20260129_001_initial_schema.sql");
        client
            .batch_execute(sql1)
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Migration 001 failed: {}", e)))?;

        // Migration 002: shard configuration table
        let sql2 = include_str!("../../migrations/20260221_002_shard_config.sql");
        client
            .batch_execute(sql2)
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Migration 002 failed: {}", e)))?;

        // Migration 003: server config key-value store
        let sql3 = include_str!("../../migrations/20260224_003_server_config.sql");
        client
            .batch_execute(sql3)
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Migration 003 failed: {}", e)))?;

        // Migration 004: multi-tenancy (tenants table, tenant_id columns, default tenant)
        let sql4 = include_str!("../../migrations/20260404_004_multi_tenancy.sql");
        client
            .batch_execute(sql4)
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Migration 004 failed: {}", e)))?;

        Ok(())
    }

    /// Get a reference to the connection pool
    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    /// Health check - verify database is accessible
    pub async fn health_check(&self) -> VPKResult<bool> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Health check pool error: {}", e))
        })?;
        client
            .query_one("SELECT 1", &[])
            .await
            .map(|_| true)
            .map_err(|e| VPKError::DatabaseError(format!("Health check failed: {}", e)))
    }
}
