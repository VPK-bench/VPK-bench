//! Multi-shard fan-out manager for Eve's vector databases
//!
//! Implements horizontal partitioning: each Eve shard holds a disjoint
//! subset of encrypted document vectors. Queries fan out to all shards
//! in parallel; top-K is merged at VPK.
//!
//! # Security benefit
//!
//! Each Eve shard sees only 1/N of the corpus and 1/N of query frequency.
//! This limits the effectiveness of frequency-analysis attacks: Eve_i can
//! only correlate queries against its own shard, not the full knowledge base.
//!
//! # Degraded mode
//!
//! If a shard fails during search, its results are silently skipped.
//! The remaining N-1 shards still serve results. Health status reflects
//! which shards are reachable.

use crate::config::ShardEndpoint;
use crate::crypto::Ckks;
use crate::error::{VPKError, VPKResult};
use crate::vectordb::{CkksVectorStorage, RemoteEveClient, VectorDBClient, VectorStorage};
use ndarray::Array1;
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Health status of a single Eve shard
#[derive(Debug, Clone, Serialize)]
pub struct ShardHealthStatus {
    pub shard_id: i32,
    pub name: String,
    pub healthy: bool,
    pub num_vectors: usize,
    pub error: Option<String>,
}

/// Search result tagged with the originating shard
#[derive(Debug, Clone)]
pub struct ShardedSearchResult {
    /// Scrambled ID (Eve's opaque identifier — not the real doc ID)
    pub scrambled_id: i64,
    /// Similarity score computed on encrypted vectors (homomorphic property)
    pub score: f64,
    /// Which shard returned this result
    pub shard_id: i32,
}

struct Shard {
    shard_id: i32,
    name: String,
    /// Arc so we can clone the handle into spawned tasks.
    /// Either an in-memory `VectorDBClient` (local) or a `RemoteEveClient` (remote).
    client: Arc<dyn VectorStorage>,
}

/// Manages N Eve shards with parallel fan-out search and round-robin assignment
///
/// For MVP: N in-memory `VectorDBClient` instances simulate N Qdrant/FAISS nodes.
/// For production: each `VectorDBClient` would be replaced by a network client
/// pointing at a distinct Eve host.
pub struct ShardManager {
    shards: Vec<Shard>,
    /// Atomic counter for round-robin assignment
    next_shard: AtomicUsize,
}

impl ShardManager {
    /// Create a new ShardManager.
    ///
    /// For each `ShardEndpoint`:
    /// - `endpoint = None`  → local in-memory `VectorDBClient` (MVP / testing)
    /// - `endpoint = Some(url)` → remote `RemoteEveClient` talking to an Eve node
    ///
    /// When `ckks` is `Some`, local shards use `CkksVectorStorage` instead of
    /// `VectorDBClient`, enabling real CKKS homomorphic search.
    ///
    /// # Arguments
    /// * `endpoints` - Ordered slice of shard endpoint descriptors from config
    /// * `peer_id`   - Bob's peer ID sent to Eve during tenant registration
    /// * `dimension` - Vector dimension for every shard
    /// * `ckks`      - Optional CKKS context for homomorphic search
    pub fn new(
        endpoints: &[ShardEndpoint],
        peer_id: &str,
        dimension: usize,
        ckks: Option<Arc<Ckks>>,
        use_dot_product: bool,
    ) -> Self {
        let shards = endpoints
            .iter()
            .map(|ep| {
                let client: Arc<dyn VectorStorage> = match &ep.endpoint {
                    Some(url) => Arc::new(RemoteEveClient::new(
                        url.clone(),
                        peer_id.to_string(),
                        ep.shard_id,
                        ep.tenant_id.clone(),
                        1024,
                        dimension,
                    )),
                    None => {
                        if let Some(ref ckks_ctx) = ckks {
                            Arc::new(CkksVectorStorage::new(Arc::clone(ckks_ctx), dimension))
                        } else {
                            Arc::new(VectorDBClient::new(dimension, use_dot_product))
                        }
                    }
                };
                Shard {
                    shard_id: ep.shard_id,
                    name: ep.name.clone(),
                    client,
                }
            })
            .collect();

        Self {
            shards,
            next_shard: AtomicUsize::new(0),
        }
    }

    /// Number of configured shards
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Assign the next shard in round-robin order
    ///
    /// Returns the `shard_id` that the caller should use for the next upload.
    pub fn assign_shard(&self) -> i32 {
        if self.shards.is_empty() {
            return 0;
        }
        let idx = self.next_shard.fetch_add(1, Ordering::Relaxed) % self.shards.len();
        self.shards[idx].shard_id
    }

    /// Upload encrypted vectors to a specific shard
    ///
    /// # Arguments
    /// * `shard_id` - Destination shard (must exist in this manager)
    /// * `vectors`  - (scrambled_id, encrypted_vector) pairs
    pub async fn upload_to_shard(
        &self,
        shard_id: i32,
        vectors: Vec<(i64, Array1<f64>)>,
    ) -> VPKResult<usize> {
        let shard = self
            .shards
            .iter()
            .find(|s| s.shard_id == shard_id)
            .ok_or_else(|| {
                VPKError::ShardConnectionError(format!("Shard {} not found", shard_id))
            })?;
        shard.client.upload_vectors(vectors).await
    }

    /// Fan-out: search all shards in parallel, merge, return global top-K
    ///
    /// Each shard is queried for `top_k * num_shards` candidates so that
    /// noise-displaced documents near the K boundary are not truncated at the
    /// per-shard level before the global merge.  Without this over-fetch, a
    /// relevant document pushed from local rank 8 → 12 by noise injection
    /// would be missed when top_k=10, producing a spurious K=20 recall bump.
    ///
    /// After collecting all candidates they are sorted by score and truncated
    /// to the global `top_k`.  The extra per-shard candidates have negligible
    /// cost because each shard already does O(n) brute-force search.
    ///
    /// Shards that fail are silently skipped (degraded mode).
    pub async fn search_all(
        &self,
        query: &Array1<f64>,
        top_k: usize,
    ) -> VPKResult<Vec<ShardedSearchResult>> {
        if self.shards.is_empty() {
            return Ok(Vec::new());
        }

        // Ask each shard for more candidates than strictly needed so that
        // noise-displaced documents are not lost at shard boundaries.
        let n_shards = self.shards.len().max(1);
        let per_shard_k = top_k * n_shards;

        // Spawn one task per shard
        let mut handles = Vec::new();
        for shard in &self.shards {
            let client = Arc::clone(&shard.client);
            let query_clone = query.clone();
            let shard_id = shard.shard_id;
            handles.push(tokio::spawn(async move {
                let result = client.search(&query_clone, per_shard_k).await;
                (shard_id, result)
            }));
        }

        // Collect and merge
        let mut all_results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok((shard_id, Ok(results))) => {
                    for r in results {
                        all_results.push(ShardedSearchResult {
                            scrambled_id: r.index,
                            score: r.score,
                            shard_id,
                        });
                    }
                }
                Ok((shard_id, Err(e))) => {
                    tracing::warn!("Shard {} search failed (degraded mode): {}", shard_id, e);
                }
                Err(e) => {
                    tracing::error!("Shard task panicked: {}", e);
                }
            }
        }

        // Sort by score descending; take global top-K
        all_results
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(top_k);

        Ok(all_results)
    }

    /// Health-check all shards in parallel
    pub async fn health_check_all(&self) -> Vec<ShardHealthStatus> {
        let mut handles = Vec::new();
        for shard in &self.shards {
            let client = Arc::clone(&shard.client);
            let shard_id = shard.shard_id;
            let name = shard.name.clone();
            handles.push(tokio::spawn(async move {
                let result = client.health_check().await;
                (shard_id, name, result)
            }));
        }

        let mut statuses = Vec::new();
        for handle in handles {
            match handle.await {
                Ok((shard_id, name, Ok(h))) => {
                    statuses.push(ShardHealthStatus {
                        shard_id,
                        name,
                        healthy: h.healthy,
                        num_vectors: h.num_vectors,
                        error: None,
                    });
                }
                Ok((shard_id, name, Err(e))) => {
                    statuses.push(ShardHealthStatus {
                        shard_id,
                        name,
                        healthy: false,
                        num_vectors: 0,
                        error: Some(e.to_string()),
                    });
                }
                Err(e) => {
                    statuses.push(ShardHealthStatus {
                        shard_id: -1,
                        name: "unknown".to_string(),
                        healthy: false,
                        num_vectors: 0,
                        error: Some(format!("Task panicked: {}", e)),
                    });
                }
            }
        }

        statuses.sort_by_key(|s| s.shard_id);
        statuses
    }

    /// Total vector count across all shards
    pub fn total_count(&self) -> usize {
        self.shards.iter().map(|s| s.client.count()).sum()
    }

    /// Number of configured Eve shards
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Delete vectors from a specific shard
    pub async fn delete_from_shard(&self, shard_id: i32, indices: Vec<i64>) -> VPKResult<usize> {
        let shard = self
            .shards
            .iter()
            .find(|s| s.shard_id == shard_id)
            .ok_or_else(|| {
                VPKError::ShardConnectionError(format!("Shard {} not found", shard_id))
            })?;
        shard.client.delete_vectors(indices).await
    }

    /// Clear all vectors from all shards (used during re-encryption)
    pub async fn clear_all(&self) -> VPKResult<()> {
        for shard in &self.shards {
            shard.client.clear().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager(n: usize, dim: usize) -> ShardManager {
        let endpoints: Vec<crate::config::ShardEndpoint> = (1..=n)
            .map(|i| crate::config::ShardEndpoint {
                shard_id: i as i32,
                name: format!("eve-{}", i),
                endpoint: None,
                tenant_id: None,
            })
            .collect();
        ShardManager::new(&endpoints, "", dim, None)
    }

    #[test]
    fn test_round_robin_assignment() {
        let m = make_manager(3, 4);
        // Three consecutive assignments should cycle through all shards
        let ids: Vec<i32> = (0..6).map(|_| m.assign_shard()).collect();
        assert_eq!(ids[0], ids[3]); // Same shard every 3 calls
        assert_eq!(ids[1], ids[4]);
        assert_eq!(ids[2], ids[5]);
        // All three shards should appear in first 3 assignments
        let unique: std::collections::HashSet<_> = ids[..3].iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[tokio::test]
    async fn test_upload_and_search() {
        let m = make_manager(3, 4);

        // Upload a vector to shard 1
        let v1 = Array1::from_vec(vec![1.0_f64, 0.0, 0.0, 0.0]);
        m.upload_to_shard(1, vec![(100, v1)]).await.unwrap();

        // Upload a vector to shard 2
        let v2 = Array1::from_vec(vec![0.0_f64, 1.0, 0.0, 0.0]);
        m.upload_to_shard(2, vec![(200, v2)]).await.unwrap();

        // Query should return results from both shards
        let query = Array1::from_vec(vec![1.0_f64, 0.0, 0.0, 0.0]);
        let results = m.search_all(&query, 5).await.unwrap();

        assert_eq!(results.len(), 2);
        // Top result should be shard 1's vector (exact match)
        assert_eq!(results[0].scrambled_id, 100);
        assert_eq!(results[0].shard_id, 1);
    }

    #[tokio::test]
    async fn test_health_check_all() {
        let m = make_manager(3, 4);
        let statuses = m.health_check_all().await;
        assert_eq!(statuses.len(), 3);
        for s in &statuses {
            assert!(s.healthy);
            assert_eq!(s.num_vectors, 0);
            assert!(s.error.is_none());
        }
    }

    #[tokio::test]
    async fn test_total_count() {
        let m = make_manager(3, 4);
        assert_eq!(m.total_count(), 0);

        let v = Array1::from_vec(vec![1.0_f64, 0.0, 0.0, 0.0]);
        m.upload_to_shard(1, vec![(1, v.clone())]).await.unwrap();
        m.upload_to_shard(2, vec![(2, v.clone())]).await.unwrap();
        m.upload_to_shard(3, vec![(3, v)]).await.unwrap();

        assert_eq!(m.total_count(), 3);
    }

    #[tokio::test]
    async fn test_clear_all() {
        let m = make_manager(2, 4);
        let v = Array1::from_vec(vec![1.0_f64, 0.0, 0.0, 0.0]);
        m.upload_to_shard(1, vec![(1, v.clone())]).await.unwrap();
        m.upload_to_shard(2, vec![(2, v)]).await.unwrap();
        assert_eq!(m.total_count(), 2);
        m.clear_all().await.unwrap();
        assert_eq!(m.total_count(), 0);
    }
}
