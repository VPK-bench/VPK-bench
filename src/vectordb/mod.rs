//! Vector database backends for encrypted vector storage and search.
//!
//! Two implementations of the `VectorStorage` trait:
//! - `VectorDBClient` — in-memory brute-force (MVP / testing)
//! - `RemoteEveClient` — HTTP client for a remote Eve node running the storage service

use async_trait::async_trait;
use ndarray::Array1;

use crate::error::VPKResult;

pub mod client;
pub mod ckks_store;
pub mod pgvector;
pub mod remote;

pub use client::{HealthStatus, SearchResult, VectorDBClient};
pub use ckks_store::CkksVectorStorage;
pub use pgvector::PgVectorClient;
pub use remote::RemoteEveClient;

/// Abstraction over a vector storage backend (local in-memory or remote Eve HTTP).
///
/// Implementations must enforce the **Search by Index** protocol:
/// - `search()` returns only `(scrambled_id, score)` pairs — **never** the vectors themselves.
/// - `vector` on `SearchResult` must always be `None`.
#[async_trait]
pub trait VectorStorage: Send + Sync {
    /// Upload a batch of `(scrambled_id, encrypted_vector)` pairs.
    async fn upload_vectors(&self, vectors: Vec<(i64, Array1<f64>)>) -> VPKResult<usize>;

    /// Cosine-similarity search; returns at most `top_k` `(scrambled_id, score)` pairs.
    async fn search(&self, query: &Array1<f64>, top_k: usize) -> VPKResult<Vec<SearchResult>>;

    /// Delete vectors by scrambled ID; returns the number actually removed.
    async fn delete_vectors(&self, indices: Vec<i64>) -> VPKResult<usize>;

    /// Liveness check.
    async fn health_check(&self) -> VPKResult<HealthStatus>;

    /// Remove all vectors (used during key rotation / re-encryption).
    async fn clear(&self) -> VPKResult<()>;

    /// Approximate vector count (may be locally tracked for remote backends).
    fn count(&self) -> usize;
}
