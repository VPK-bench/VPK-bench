//! CKKS ciphertext vector storage
//!
//! Implements `VectorStorage` for real CKKS ciphertexts. Instead of storing
//! `Array1<f64>` and computing cosine similarity, this backend stores
//! serialized SEAL `Ciphertext` bytes and computes similarity via
//! homomorphic dot-product + decryption.
//!
//! This is the Eve-side equivalent for the CKKS algorithm: Eve holds
//! only opaque ciphertext blobs and cannot recover plaintext embeddings.

use async_trait::async_trait;
use ndarray::Array1;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::crypto::Ckks;
use crate::error::VPKResult;
use super::{HealthStatus, SearchResult, VectorStorage};

/// In-memory store of CKKS ciphertexts with homomorphic search.
pub struct CkksVectorStorage {
    /// scrambled_id → serialized SEAL Ciphertext bytes
    ciphertexts: Arc<RwLock<HashMap<i64, Vec<u8>>>>,
    /// Shared reference to the Ckks context (holds evaluator, keys, etc.)
    ckks: Arc<Ckks>,
    /// Nominal dimension (for health check reporting; not used for storage)
    dimension: usize,
}

impl CkksVectorStorage {
    pub fn new(ckks: Arc<Ckks>, dimension: usize) -> Self {
        Self {
            ciphertexts: Arc::new(RwLock::new(HashMap::new())),
            ckks,
            dimension,
        }
    }
}

#[async_trait]
impl VectorStorage for CkksVectorStorage {
    /// Upload vectors. The `Array1<f64>` values are treated as **plaintext
    /// embeddings** that get CKKS-encrypted before storage. The caller
    /// (VPK core) passes raw embeddings when the algorithm is CKKS.
    async fn upload_vectors(&self, vectors: Vec<(i64, Array1<f64>)>) -> VPKResult<usize> {
        let mut store = self.ciphertexts.write().unwrap();
        let mut uploaded = 0;

        for (id, vector) in vectors {
            let ct_bytes = self.ckks.encrypt(&vector)?;
            store.insert(id, ct_bytes);
            uploaded += 1;
        }

        Ok(uploaded)
    }

    /// Search by homomorphic dot-product.
    ///
    /// If VPK core pre-encrypted the query (cached via `Ckks::cache_query_ct`),
    /// we reuse those bytes; otherwise we encrypt here as a fallback.
    async fn search(&self, query: &Array1<f64>, top_k: usize) -> VPKResult<Vec<SearchResult>> {
        let ct_query = match self.ckks.cached_query_ct() {
            Some(ct) => ct,
            None => self.ckks.encrypt(query)?,
        };

        let store = self.ciphertexts.read().unwrap();
        if store.is_empty() {
            return Ok(Vec::new());
        }

        let mut scores: Vec<(i64, f64)> = Vec::with_capacity(store.len());
        for (id, ct_doc_bytes) in store.iter() {
            match self.ckks.dot_product_decrypt(&ct_query, ct_doc_bytes) {
                Ok(score) => scores.push((*id, score)),
                Err(e) => {
                    tracing::warn!("CKKS dot product failed for id {}: {}", id, e);
                }
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scores
            .into_iter()
            .take(top_k)
            .map(|(index, score)| SearchResult {
                index,
                score,
                vector: None,
            })
            .collect())
    }

    async fn delete_vectors(&self, indices: Vec<i64>) -> VPKResult<usize> {
        let mut store = self.ciphertexts.write().unwrap();
        let mut deleted = 0;
        for id in indices {
            if store.remove(&id).is_some() {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    async fn health_check(&self) -> VPKResult<HealthStatus> {
        let store = self.ciphertexts.read().unwrap();
        Ok(HealthStatus {
            healthy: true,
            num_vectors: store.len(),
            dimension: self.dimension,
        })
    }

    async fn clear(&self) -> VPKResult<()> {
        let mut store = self.ciphertexts.write().unwrap();
        store.clear();
        Ok(())
    }

    fn count(&self) -> usize {
        self.ciphertexts.read().unwrap().len()
    }
}
