//! Remote Eve HTTP client — implements `VectorStorage` over the remote Eve node REST API.
//!
//! Bob uses this to fan out encrypted vectors to a remote Eve node.
//!
//! # Tenant registration
//!
//! Eve requires Bob to register as a tenant before uploading.  Registration is
//! **lazy**: the first call to any mutating method triggers `POST /api/tenants`.
//! If `existing_tenant_id` is supplied (e.g. persisted from a previous run),
//! registration is skipped and that tenant UUID is used directly.
//!
//! # Search by Index
//!
//! `search()` returns **only** `(scrambled_id, score)` pairs — the `vector`
//! field on every `SearchResult` is always `None`.  Eve never sends vectors
//! back across the network boundary.

use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use ndarray::Array1;
use tokio::sync::RwLock;

use crate::error::{VPKError, VPKResult};
use super::{HealthStatus, SearchResult, VectorStorage};

/// HTTP client for a single remote Eve node (one tenant slot).
pub struct RemoteEveClient {
    /// Base URL of the Eve node (e.g. `http://192.168.1.10:7432`).
    eve_url: String,
    /// Bob's peer ID sent to Eve during tenant registration.
    peer_id: String,
    /// Logical shard identifier for logging.
    shard_id: i32,
    /// Eve-assigned tenant UUID, lazily populated on first use.
    tenant_id: RwLock<Option<String>>,
    /// Capacity offered to Eve (MB).
    capacity_limit_mb: i64,
    /// Expected vector dimension (after encryption / padding).
    dimension: usize,
    /// Shared HTTP client (reqwest handles connection pooling internally).
    http: reqwest::Client,
    /// Local approximate vector count (incremented on upload, decremented on delete).
    vector_count: AtomicUsize,
}

impl RemoteEveClient {
    /// Create a new client.
    ///
    /// # Arguments
    /// * `eve_url`             — base URL of the remote Eve node
    /// * `peer_id`             — Bob's peer ID (libp2p base58 string or similar)
    /// * `shard_id`            — logical shard number (for logging only)
    /// * `existing_tenant_id`  — UUID from a previous registration (skips re-registration)
    /// * `capacity_limit_mb`   — how much capacity to request from Eve
    /// * `dimension`           — vector dimension (used when registering the tenant)
    pub fn new(
        eve_url: String,
        peer_id: String,
        shard_id: i32,
        existing_tenant_id: Option<String>,
        capacity_limit_mb: i64,
        dimension: usize,
    ) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("failed to build reqwest client");

        Self {
            eve_url,
            peer_id,
            shard_id,
            tenant_id: RwLock::new(existing_tenant_id),
            capacity_limit_mb,
            dimension,
            http,
            vector_count: AtomicUsize::new(0),
        }
    }

    /// Return the tenant UUID, registering with Eve if not yet done.
    async fn ensure_tenant(&self) -> VPKResult<String> {
        // Fast path — already have a tenant UUID.
        {
            let guard = self.tenant_id.read().await;
            if let Some(ref id) = *guard {
                return Ok(id.clone());
            }
        }

        // Slow path — take write lock, double-check, then register.
        let mut guard = self.tenant_id.write().await;
        if let Some(ref id) = *guard {
            return Ok(id.clone());
        }

        let url = format!("{}/api/tenants", self.eve_url);
        let body = serde_json::json!({
            "peer_id": self.peer_id,
            "capacity_limit_mb": self.capacity_limit_mb,
            "vector_dimension": self.dimension,
        });

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                VPKError::HttpError(format!(
                    "shard {}: tenant registration failed: {}",
                    self.shard_id, e
                ))
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(VPKError::HttpError(format!(
                "shard {}: create tenant {} — {}",
                self.shard_id, status, text
            )));
        }

        let info: serde_json::Value = resp.json().await.map_err(|e| {
            VPKError::HttpError(format!("shard {}: parse tenant response: {}", self.shard_id, e))
        })?;

        let tenant_id = info["tenant_id"]
            .as_str()
            .ok_or_else(|| {
                VPKError::HttpError(format!(
                    "shard {}: tenant_id missing from Eve response",
                    self.shard_id
                ))
            })?
            .to_string();

        tracing::info!(
            shard_id = self.shard_id,
            eve = %self.eve_url,
            tenant_id = %tenant_id,
            "registered as Eve tenant"
        );

        *guard = Some(tenant_id.clone());
        Ok(tenant_id)
    }
}

#[async_trait]
impl VectorStorage for RemoteEveClient {
    async fn upload_vectors(&self, vectors: Vec<(i64, Array1<f64>)>) -> VPKResult<usize> {
        let tenant_id = self.ensure_tenant().await?;
        let url = format!("{}/api/tenants/{}/vectors", self.eve_url, tenant_id);

        let n = vectors.len();
        // Eve API uses f32; PHE uses f64 internally — downcast here.
        let entries: Vec<serde_json::Value> = vectors
            .into_iter()
            .map(|(id, vec)| {
                let embedding: Vec<f32> = vec.iter().map(|&x| x as f32).collect();
                serde_json::json!({ "id": id, "embedding": embedding })
            })
            .collect();

        let body = serde_json::json!({ "vectors": entries });

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                VPKError::HttpError(format!("shard {}: upload_vectors: {}", self.shard_id, e))
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(VPKError::HttpError(format!(
                "shard {}: upload_vectors {} — {}",
                self.shard_id, status, text
            )));
        }

        let result: serde_json::Value = resp.json().await.map_err(|e| {
            VPKError::HttpError(format!("shard {}: parse upload response: {}", self.shard_id, e))
        })?;

        let uploaded = result["uploaded"].as_u64().unwrap_or(n as u64) as usize;
        self.vector_count.fetch_add(uploaded, Ordering::Relaxed);
        Ok(uploaded)
    }

    async fn search(&self, query: &Array1<f64>, top_k: usize) -> VPKResult<Vec<SearchResult>> {
        let tenant_id = self.ensure_tenant().await?;
        let url = format!("{}/api/tenants/{}/search", self.eve_url, tenant_id);

        // Eve API uses f32.
        let query_f32: Vec<f32> = query.iter().map(|&x| x as f32).collect();
        let body = serde_json::json!({ "query": query_f32, "top_k": top_k });

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                VPKError::HttpError(format!("shard {}: search: {}", self.shard_id, e))
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(VPKError::HttpError(format!(
                "shard {}: search {} — {}",
                self.shard_id, status, text
            )));
        }

        let response: serde_json::Value = resp.json().await.map_err(|e| {
            VPKError::HttpError(format!("shard {}: parse search response: {}", self.shard_id, e))
        })?;

        // Search by Index protocol: Eve returns (id, score) only — NEVER vectors.
        let empty = vec![];
        let results = response["results"]
            .as_array()
            .unwrap_or(&empty)
            .iter()
            .map(|r| SearchResult {
                index: r["id"].as_i64().unwrap_or(0),
                score: r["score"].as_f64().unwrap_or(0.0),
                vector: None, // NEVER populate — Search by Index protocol
            })
            .collect();

        Ok(results)
    }

    async fn delete_vectors(&self, indices: Vec<i64>) -> VPKResult<usize> {
        let tenant_id = self.ensure_tenant().await?;
        let url = format!("{}/api/tenants/{}/vectors", self.eve_url, tenant_id);

        let n = indices.len();
        let body = serde_json::json!({ "ids": indices });

        let resp = self
            .http
            .delete(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                VPKError::HttpError(format!("shard {}: delete_vectors: {}", self.shard_id, e))
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(VPKError::HttpError(format!(
                "shard {}: delete_vectors {} — {}",
                self.shard_id, status, text
            )));
        }

        let result: serde_json::Value = resp.json().await.map_err(|e| {
            VPKError::HttpError(format!("shard {}: parse delete response: {}", self.shard_id, e))
        })?;

        let deleted = result["deleted"].as_u64().unwrap_or(n as u64) as usize;
        let current = self.vector_count.load(Ordering::Relaxed);
        self.vector_count
            .store(current.saturating_sub(deleted), Ordering::Relaxed);
        Ok(deleted)
    }

    async fn health_check(&self) -> VPKResult<HealthStatus> {
        let url = format!("{}/api/health", self.eve_url);

        let reachable = self
            .http
            .get(&url)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);

        Ok(HealthStatus {
            healthy: reachable,
            num_vectors: self.vector_count.load(Ordering::Relaxed),
            dimension: self.dimension,
        })
    }

    async fn clear(&self) -> VPKResult<()> {
        // Remote bulk-clear is not yet exposed by the Eve API.
        // Re-encryption on remote shards requires deleting + re-uploading via the caller.
        tracing::warn!(
            shard_id = self.shard_id,
            eve = %self.eve_url,
            "RemoteEveClient::clear() is a no-op — re-encrypt by calling delete_vectors then upload_vectors"
        );
        Ok(())
    }

    fn count(&self) -> usize {
        self.vector_count.load(Ordering::Relaxed)
    }
}
