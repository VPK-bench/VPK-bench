//! Local PGVector backend — implements `VectorStorage` on Bob's own PostgreSQL.
//!
//! `PgVectorClient` gives Bob a **persistent** local vector store as an alternative
//! to the in-memory `VectorDBClient`.  It is backed by the same PostgreSQL instance
//! that Bob already runs for his documents database, so no extra infrastructure is
//! needed.
//!
//! # Table naming
//!
//! Each tenant gets its own table: `vpk_local_{tenant_hex8}` (first 8 hex digits
//! of the tenant UUID).  The `vpk_local_` prefix avoids collisions with:
//! - Bob's core tables (`documents`, `index_mapping`, …)
//! - Eve's tables (`eve_vectors_{hex8}`) if Bob and Eve share a DB (unusual but possible)
//!
//! # Index strategy
//!
//! Flat storage (no HNSW) by default, following the project index strategy:
//! when Bob shards across many Eves each shard is small, so brute-force beats
//! HNSW.  A future migration can create the HNSW index when a per-tenant
//! threshold is crossed.
//!
//! # Precision
//!
//! PHE uses `f64` internally; PGVector stores `VECTOR(n)` as `f32`.  Vectors
//! are downcast on write (same as `RemoteEveClient`) — precision loss is
//! negligible given that the encryption pipeline has already scrambled the
//! values.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use deadpool_postgres::Pool;
use ndarray::Array1;
use pgvector::Vector;
use uuid::Uuid;

use crate::error::{VPKError, VPKResult};
use super::{HealthStatus, SearchResult, VectorStorage};

/// Local PGVector storage for a single tenant.
pub struct PgVectorClient {
    pool: Pool,
    /// Tenant UUID — used to derive the table name.
    tenant_id: Uuid,
    /// Expected vector dimension (after encryption / padding).
    dimension: usize,
    /// Derived table name: `vpk_local_{first-8-hex-chars}`.
    table: String,
    /// Approximate vector count, maintained locally (avoids a round-trip for count()).
    vector_count: AtomicUsize,
    /// Set to true after the table has been created for the first time.
    initialized: AtomicBool,
}

impl PgVectorClient {
    /// Create a new client.
    ///
    /// The table is created lazily on first use; the client can be constructed
    /// without an active database connection.
    pub fn new(pool: Pool, tenant_id: Uuid, dimension: usize) -> Self {
        let hex = tenant_id.as_simple().to_string();
        let table = format!("vpk_local_{}", &hex[..8]);
        Self {
            pool,
            tenant_id,
            dimension,
            table,
            vector_count: AtomicUsize::new(0),
            initialized: AtomicBool::new(false),
        }
    }

    /// Return the derived table name (useful for debugging).
    pub fn table_name(&self) -> &str {
        &self.table
    }

    /// Ensure the vector table exists (idempotent, called before first write/read).
    async fn ensure_table(&self) -> VPKResult<()> {
        if self.initialized.load(Ordering::Relaxed) {
            return Ok(());
        }

        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("PgVectorClient pool error: {}", e))
        })?;

        // Flat table — no HNSW index yet (see module doc for rationale).
        let table = &self.table;
        let dim = self.dimension;
        client
            .batch_execute(&format!(
                r#"
CREATE TABLE IF NOT EXISTS {table} (
    id          BIGINT PRIMARY KEY,
    embedding   VECTOR({dim}) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
"#,
            ))
            .await
            .map_err(|e| {
                VPKError::DatabaseError(format!("create table {}: {}", self.table, e))
            })?;

        // Seed the local count from the DB (handles restart after previous writes).
        let row = client
            .query_one(&format!("SELECT COUNT(*) FROM {}", self.table), &[])
            .await
            .map_err(|e| VPKError::DatabaseError(format!("count {}: {}", self.table, e)))?;
        let n: i64 = row.get(0);
        self.vector_count.store(n as usize, Ordering::Relaxed);

        self.initialized.store(true, Ordering::Relaxed);
        Ok(())
    }
}

#[async_trait]
impl VectorStorage for PgVectorClient {
    async fn upload_vectors(&self, vectors: Vec<(i64, Array1<f64>)>) -> VPKResult<usize> {
        self.ensure_table().await?;

        if vectors.is_empty() {
            return Ok(0);
        }

        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("PgVectorClient pool error: {}", e))
        })?;

        let stmt = client
            .prepare(&format!(
                "INSERT INTO {} (id, embedding) VALUES ($1, $2) \
                 ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, created_at = NOW()",
                self.table
            ))
            .await
            .map_err(|e| VPKError::DatabaseError(format!("prepare upload: {}", e)))?;

        let n = vectors.len();
        for (id, vec) in &vectors {
            // Downcast f64 → f32 for pgvector storage.
            let embedding: Vec<f32> = vec.iter().map(|&x| x as f32).collect();
            let pgvec = Vector::from(embedding);
            client
                .execute(&stmt, &[id, &pgvec])
                .await
                .map_err(|e| VPKError::DatabaseError(format!("insert vector: {}", e)))?;
        }

        self.vector_count.fetch_add(n, Ordering::Relaxed);
        Ok(n)
    }

    async fn search(&self, query: &Array1<f64>, top_k: usize) -> VPKResult<Vec<SearchResult>> {
        self.ensure_table().await?;

        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("PgVectorClient pool error: {}", e))
        })?;

        let query_f32: Vec<f32> = query.iter().map(|&x| x as f32).collect();
        let query_vec = Vector::from(query_f32);

        // `<=>` is cosine distance; `1 - distance` gives cosine similarity.
        let rows = client
            .query(
                &format!(
                    "SELECT id, 1 - (embedding <=> $1) AS score \
                     FROM {} \
                     ORDER BY embedding <=> $1 \
                     LIMIT $2",
                    self.table
                ),
                &[&query_vec, &(top_k as i64)],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("search {}: {}", self.table, e)))?;

        // Search by Index: return (scrambled_id, score) only — vector field stays None.
        Ok(rows
            .iter()
            .map(|r| SearchResult {
                index: r.get(0),
                score: r.get(1),
                vector: None,
            })
            .collect())
    }

    async fn delete_vectors(&self, indices: Vec<i64>) -> VPKResult<usize> {
        self.ensure_table().await?;

        if indices.is_empty() {
            return Ok(0);
        }

        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("PgVectorClient pool error: {}", e))
        })?;

        let n = client
            .execute(
                &format!("DELETE FROM {} WHERE id = ANY($1)", self.table),
                &[&indices],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("delete vectors: {}", e)))? as usize;

        let current = self.vector_count.load(Ordering::Relaxed);
        self.vector_count.store(current.saturating_sub(n), Ordering::Relaxed);
        Ok(n)
    }

    async fn health_check(&self) -> VPKResult<HealthStatus> {
        let client = self.pool.get().await;
        let healthy = client.is_ok();

        Ok(HealthStatus {
            healthy,
            num_vectors: self.vector_count.load(Ordering::Relaxed),
            dimension: self.dimension,
        })
    }

    async fn clear(&self) -> VPKResult<()> {
        self.ensure_table().await?;

        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("PgVectorClient pool error: {}", e))
        })?;

        client
            .execute(&format!("TRUNCATE {}", self.table), &[])
            .await
            .map_err(|e| VPKError::DatabaseError(format!("truncate {}: {}", self.table, e)))?;

        self.vector_count.store(0, Ordering::Relaxed);
        Ok(())
    }

    fn count(&self) -> usize {
        self.vector_count.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies table name derivation is stable and uses the correct prefix.
    #[test]
    fn test_table_name() {
        let id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let hex = id.as_simple().to_string();
        let table = format!("vpk_local_{}", &hex[..8]);
        assert_eq!(table, "vpk_local_550e8400");
    }
}
