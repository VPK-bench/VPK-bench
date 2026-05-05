//! Vector database client for Eve's encrypted vector storage
//!
//! This module implements **Eve's side of the VPK protocol** - the untrusted
//! service operator who stores encrypted vectors and performs similarity search.
//!
//! # Security Model
//!
//! Eve (this module) only sees:
//! - Encrypted vectors: E(d₁), E(d₂), ... E(dₙ)
//! - Scrambled IDs: Not the real document IDs
//! - Encrypted queries: E(q)
//!
//! Eve NEVER sees:
//! - Plaintext documents
//! - Real document IDs
//! - Plaintext queries
//! - Who submitted queries (Alice's identity)
//!
//! # "Search by Index" Protocol
//!
//! Critical security feature: Eve returns ONLY:
//! - `index`: Scrambled ID (Bob must reverse-map to real ID)
//! - `score`: Similarity score computed on encrypted vectors
//!
//! Eve NEVER returns:
//! - Document content (doesn't have it)
//! - Encrypted vectors (stays in database, reduces attack surface)
//! - Real document IDs (only knows scrambled IDs)
//!
//! This minimizes data crossing the security boundary (100-1000x bandwidth reduction)
//! and prevents vector leakage attacks.
//!
//! # Implementation
//!
//! For MVP: In-memory FAISS-like implementation using brute-force search
//!
//! For Production: Can be extended to support various vector databases.
//!
//! ## Critical Security Configuration
//!
//! **"Search by Index" is NATIVE only in FAISS**. Other databases require
//! application-level enforcement:
//!
//! - **FAISS**: Returns indices + scores by default ✅ (native behavior)
//! - **Qdrant**: Must set `with_vector: false` ⚠️ (returns vectors by default)
//! - **Milvus**: Must exclude vectors from `output_fields` ⚠️
//! - **Pinecone**: Must set `include_values=False` ⚠️
//! - **Weaviate**: Must not request vector field in GraphQL ⚠️
//!
//! **SECURITY REQUIREMENT**: When implementing production vector DB clients,
//! you MUST configure them to NOT return vectors, or the "Search by Index"
//! security property is violated and encrypted vectors leak across the
//! security boundary.
//!
//! See README section 2.4.4 for example configurations.

use async_trait::async_trait;
use crate::error::{VPKError, VPKResult};
use ndarray::Array1;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use super::VectorStorage;

/// Search result from vector database (Eve's response)
///
/// SECURITY: This struct implements "Search by Index" - a critical security feature
/// that minimizes data exposure. Eve returns ONLY scrambled indices and similarity scores,
/// never the actual encrypted vectors or document content.
///
/// This provides:
/// - Bandwidth reduction (100-1000x less data)
/// - No vector leakage (encrypted vectors never leave Eve's database)
/// - ID unlinkability (scrambled IDs prevent correlation to real documents)
/// - Metadata privacy (document size/structure hidden)
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Scrambled ID (NOT the real document ID! Mapping known only to Bob's VPK)
    pub index: i64,

    /// Similarity score computed on encrypted vectors
    /// This is the homomorphic property: cos(E(q), E(d)) ≈ cos(q, d)
    pub score: f64,

    /// The encrypted vector (ALWAYS None for security - see line 130)
    /// Never returned to minimize attack surface and bandwidth
    pub vector: Option<Array1<f64>>,
}

/// Health status of vector database
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub healthy: bool,
    pub num_vectors: usize,
    pub dimension: usize,
}

/// Vector database client
///
/// For MVP: In-memory FAISS-like implementation using brute-force search
/// For Production: Can be extended to use actual FAISS or other vector DBs
pub struct VectorDBClient {
    /// Storage for encrypted vectors: scrambled_id -> vector
    vectors: Arc<RwLock<HashMap<i64, Vec<f64>>>>,
    /// Expected vector dimension
    dimension: usize,
    /// Use raw dot product instead of cosine similarity.
    /// Required for DIEHARD: it preserves inner products exactly but not vector
    /// norms, so cosine similarity distorts rankings. For unit-normalised inputs
    /// (all-MiniLM-L6-v2) dot product ranking == cosine ranking on the originals.
    use_dot_product: bool,
}

impl VectorDBClient {
    /// Create a new in-memory vector database client
    ///
    /// # Arguments
    /// * `dimension` - Expected dimension of vectors
    /// * `use_dot_product` - Use dot product similarity instead of cosine
    pub fn new(dimension: usize, use_dot_product: bool) -> Self {
        Self {
            vectors: Arc::new(RwLock::new(HashMap::new())),
            dimension,
            use_dot_product,
        }
    }

    /// Upload vectors to the database (batch operation)
    ///
    /// # Arguments
    /// * `vectors` - List of (scrambled_id, encrypted_vector) pairs
    ///
    /// # Returns
    /// Number of vectors successfully uploaded
    pub async fn upload_vectors(&self, vectors: Vec<(i64, Array1<f64>)>) -> VPKResult<usize> {
        let mut store = self.vectors.write().unwrap();

        let mut uploaded = 0;
        for (id, vector) in vectors {
            // Validate dimension
            if vector.len() != self.dimension {
                return Err(VPKError::InvalidDimension {
                    expected: self.dimension,
                    actual: vector.len(),
                });
            }

            // Convert to Vec for storage
            let vec_data: Vec<f64> = vector.to_vec();
            store.insert(id, vec_data);
            uploaded += 1;
        }

        Ok(uploaded)
    }

    /// Search for similar vectors using cosine similarity
    ///
    /// # Arguments
    /// * `query` - Query vector (encrypted)
    /// * `top_k` - Number of top results to return
    ///
    /// # Returns
    /// List of search results sorted by similarity (highest first)
    pub async fn search(&self, query: &Array1<f64>, top_k: usize) -> VPKResult<Vec<SearchResult>> {
        // Validate query dimension
        if query.len() != self.dimension {
            return Err(VPKError::InvalidDimension {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let store = self.vectors.read().unwrap();

        if store.is_empty() {
            return Ok(Vec::new());
        }

        // Compute similarity with all vectors (brute force).
        // DIEHARD uses dot product: it preserves ⟨Aq,Bd⟩=⟨q,d⟩ exactly but
        // changes vector norms, so cosine similarity distorts rankings.
        let metric = if self.use_dot_product { dot_product } else { cosine_similarity };
        let mut similarities: Vec<(i64, f64)> = store
            .iter()
            .map(|(id, vec)| {
                let score = metric(query, vec);
                (*id, score)
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k results
        // SECURITY FEATURE: "Search by Index"
        // Eve returns ONLY scrambled indices + similarity scores
        // This minimizes data crossing the security boundary and prevents vector leakage
        let results: Vec<SearchResult> = similarities
            .into_iter()
            .take(top_k)
            .map(|(index, score)| SearchResult {
                index,  // Scrambled ID only (not real document ID)
                score,  // Similarity computed on encrypted vectors
                vector: None,  // NEVER return vectors - security & bandwidth optimization
            })
            .collect();

        Ok(results)
    }

    /// Delete vectors by indices
    ///
    /// # Arguments
    /// * `indices` - List of scrambled IDs to delete
    ///
    /// # Returns
    /// Number of vectors successfully deleted
    pub async fn delete_vectors(&self, indices: Vec<i64>) -> VPKResult<usize> {
        let mut store = self.vectors.write().unwrap();

        let mut deleted = 0;
        for id in indices {
            if store.remove(&id).is_some() {
                deleted += 1;
            }
        }

        Ok(deleted)
    }

    /// Get a specific vector by ID
    ///
    /// # Arguments
    /// * `id` - Scrambled ID
    ///
    /// # Returns
    /// The encrypted vector if found
    pub async fn get_vector(&self, id: i64) -> VPKResult<Option<Array1<f64>>> {
        let store = self.vectors.read().unwrap();

        Ok(store.get(&id).map(|vec| Array1::from_vec(vec.clone())))
    }

    /// Health check - verify database is accessible and get stats
    pub async fn health_check(&self) -> VPKResult<HealthStatus> {
        let store = self.vectors.read().unwrap();

        Ok(HealthStatus {
            healthy: true,
            num_vectors: store.len(),
            dimension: self.dimension,
        })
    }

    /// Clear all vectors (useful for testing)
    pub async fn clear(&self) -> VPKResult<()> {
        let mut store = self.vectors.write().unwrap();
        store.clear();
        Ok(())
    }

    /// Get number of vectors stored
    pub fn count(&self) -> usize {
        let store = self.vectors.read().unwrap();
        store.len()
    }
}

// ---------------------------------------------------------------------------
// VectorStorage trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl VectorStorage for VectorDBClient {
    async fn upload_vectors(&self, vectors: Vec<(i64, Array1<f64>)>) -> VPKResult<usize> {
        VectorDBClient::upload_vectors(self, vectors).await
    }

    async fn search(&self, query: &Array1<f64>, top_k: usize) -> VPKResult<Vec<super::SearchResult>> {
        VectorDBClient::search(self, query, top_k).await
    }

    async fn delete_vectors(&self, indices: Vec<i64>) -> VPKResult<usize> {
        VectorDBClient::delete_vectors(self, indices).await
    }

    async fn health_check(&self) -> VPKResult<super::HealthStatus> {
        VectorDBClient::health_check(self).await
    }

    async fn clear(&self) -> VPKResult<()> {
        VectorDBClient::clear(self).await
    }

    fn count(&self) -> usize {
        VectorDBClient::count(self)
    }
}

fn cosine_similarity(v1: &Array1<f64>, v2: &[f64]) -> f64 {
    let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm1 < 1e-10 || norm2 < 1e-10 { return 0.0; }
    dot / (norm1 * norm2)
}

/// Raw dot product — correct similarity metric for DIEHARD, which preserves
/// inner products exactly but not vector norms.
fn dot_product(v1: &Array1<f64>, v2: &[f64]) -> f64 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_upload_and_search() {
        let client = VectorDBClient::new(3);

        // Upload test vectors
        let v1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let v2 = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let v3 = Array1::from_vec(vec![0.9, 0.1, 0.0]); // Similar to v1

        let vectors = vec![(1, v1.clone()), (2, v2.clone()), (3, v3.clone())];

        let uploaded = client.upload_vectors(vectors).await.unwrap();
        assert_eq!(uploaded, 3);

        // Search with query similar to v1
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let results = client.search(&query, 2).await.unwrap();

        assert_eq!(results.len(), 2);
        // v1 should be most similar
        assert_eq!(results[0].index, 1);
        // v3 should be second (more similar to v1 than v2)
        assert_eq!(results[1].index, 3);
    }

    #[tokio::test]
    async fn test_delete_vectors() {
        let client = VectorDBClient::new(3);

        // Upload vectors
        let vectors = vec![
            (1, Array1::from_vec(vec![1.0, 0.0, 0.0])),
            (2, Array1::from_vec(vec![0.0, 1.0, 0.0])),
        ];
        client.upload_vectors(vectors).await.unwrap();

        assert_eq!(client.count(), 2);

        // Delete one vector
        let deleted = client.delete_vectors(vec![1]).await.unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(client.count(), 1);

        // Search should only return remaining vector
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let results = client.search(&query, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 2);
    }

    #[tokio::test]
    async fn test_health_check() {
        let client = VectorDBClient::new(3);

        let health = client.health_check().await.unwrap();
        assert!(health.healthy);
        assert_eq!(health.num_vectors, 0);
        assert_eq!(health.dimension, 3);

        // Upload a vector
        let vectors = vec![(1, Array1::from_vec(vec![1.0, 0.0, 0.0]))];
        client.upload_vectors(vectors).await.unwrap();

        let health = client.health_check().await.unwrap();
        assert_eq!(health.num_vectors, 1);
    }

    #[tokio::test]
    async fn test_invalid_dimension() {
        let client = VectorDBClient::new(3);

        // Try to upload vector with wrong dimension
        let wrong_vec = Array1::from_vec(vec![1.0, 0.0]); // 2D instead of 3D
        let result = client.upload_vectors(vec![(1, wrong_vec)]).await;

        assert!(result.is_err());
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let v1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v1, &v2);
        assert!((sim - 1.0).abs() < 1e-6);

        // Orthogonal vectors
        let v1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let v2 = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&v1, &v2);
        assert!(sim.abs() < 1e-6);

        // Opposite vectors
        let v1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&v1, &v2);
        assert!((sim + 1.0).abs() < 1e-6);
    }
}

