//! Embedding model trait and implementations

use crate::error::{VPKError, VPKResult};
use crate::utils::vector_ops::normalize;
use blake3::Hasher;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Trait for embedding models
#[async_trait::async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Embed a single text
    async fn embed(&self, text: &str) -> VPKResult<Array1<f64>>;

    /// Embed multiple texts
    async fn embed_batch(&self, texts: &[String]) -> VPKResult<Vec<Array1<f64>>>;

    /// Get embedding dimension
    fn dimension(&self) -> usize;
}

/// Dummy embedding model for MVP (deterministic hash-based)
///
/// This model generates deterministic embeddings from text using BLAKE3 hashing.
/// It's suitable for testing and MVP purposes, but should be replaced with a
/// real embedding model (e.g., sentence-transformers) for production use.
///
/// Properties:
/// - Deterministic: Same text always produces same embedding
/// - Normalized: All embeddings have L2 norm = 1.0 (required for dimensional scrambling)
/// - Fast: No neural network inference
/// - Not semantic: Similar texts don't have similar embeddings
pub struct DummyEmbeddingModel {
    dimension: usize,
}

impl DummyEmbeddingModel {
    /// Create a new dummy embedding model
    ///
    /// # Arguments
    /// * `dimension` - Embedding dimension (e.g., 384, 768, 1536)
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Generate deterministic embedding from text using BLAKE3 hash
    fn hash_to_embedding(&self, text: &str) -> VPKResult<Array1<f64>> {
        // Hash the text to get deterministic bytes
        let hash = blake3::hash(text.as_bytes());
        let hash_bytes = hash.as_bytes();

        // Generate vector by repeatedly hashing to get enough bytes
        let mut values = Vec::with_capacity(self.dimension);

        for i in 0..self.dimension {
            // Hash (original_hash || index) to get more bytes
            let mut hasher = Hasher::new();
            hasher.update(hash_bytes);
            hasher.update(&(i as u64).to_le_bytes());
            let chunk_hash = hasher.finalize();
            let chunk_bytes = chunk_hash.as_bytes();

            // Convert first 8 bytes to f64
            let raw_value = u64::from_le_bytes([
                chunk_bytes[0], chunk_bytes[1], chunk_bytes[2], chunk_bytes[3],
                chunk_bytes[4], chunk_bytes[5], chunk_bytes[6], chunk_bytes[7],
            ]);

            // Map to [-1.0, 1.0] range
            let normalized_value = (raw_value as f64 / u64::MAX as f64) * 2.0 - 1.0;
            values.push(normalized_value);
        }

        let vector = Array1::from_vec(values);

        // Normalize to unit vector (L2 norm = 1.0)
        // This is CRITICAL for dimensional scrambling to preserve rankings
        normalize(&vector)
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for DummyEmbeddingModel {
    async fn embed(&self, text: &str) -> VPKResult<Array1<f64>> {
        self.hash_to_embedding(text)
    }

    async fn embed_batch(&self, texts: &[String]) -> VPKResult<Vec<Array1<f64>>> {
        texts.iter().map(|text| self.hash_to_embedding(text)).collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// HTTP embedding model using sentence-transformers via REST API
///
/// This model calls a local HTTP embedding server (embedding_server.py)
/// that uses sentence-transformers for real semantic embeddings.
///
/// Properties:
/// - Semantic: Similar texts have similar embeddings
/// - Normalized: All embeddings have L2 norm = 1.0
/// - Fast: ~50-100ms per embedding with all-MiniLM-L6-v2
/// - Requires: embedding_server.py running on http://localhost:5001
pub struct HttpEmbeddingModel {
    endpoint: String,
    dimension: usize,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct EmbedRequest {
    text: String,
}

#[derive(Serialize)]
struct EmbedBatchRequest {
    texts: Vec<String>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f64>,
    dimension: usize,
}

#[derive(Deserialize)]
struct EmbedBatchResponse {
    embeddings: Vec<Vec<f64>>,
    count: usize,
    dimension: usize,
}

impl HttpEmbeddingModel {
    /// Create a new HTTP embedding model
    ///
    /// # Arguments
    /// * `endpoint` - Base URL of the embedding server (e.g., "http://localhost:5001")
    /// * `dimension` - Expected embedding dimension (default: 384)
    pub fn new(endpoint: &str, dimension: usize) -> Self {
        // Create async client with timeout
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            endpoint: endpoint.to_string(),
            dimension,
            client,
        }
    }

    /// Create with default settings (localhost:5001, 384 dimensions)
    pub fn default() -> Self {
        Self::new("http://localhost:5001", 384)
    }

    /// Check if the embedding server is healthy
    pub async fn health_check(&self) -> VPKResult<bool> {
        let url = format!("{}/health", self.endpoint);
        match self.client.get(&url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for HttpEmbeddingModel {
    async fn embed(&self, text: &str) -> VPKResult<Array1<f64>> {
        let url = format!("{}/embed", self.endpoint);
        let request = EmbedRequest {
            text: text.to_string(),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| VPKError::EmbeddingError(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(VPKError::EmbeddingError(format!(
                "Embedding server returned error: {}",
                response.status()
            )));
        }

        let embed_response: EmbedResponse = response
            .json()
            .await
            .map_err(|e| VPKError::EmbeddingError(format!("Failed to parse response: {}", e)))?;

        if embed_response.dimension != self.dimension {
            return Err(VPKError::EmbeddingError(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension, embed_response.dimension
            )));
        }

        Ok(Array1::from_vec(embed_response.embedding))
    }

    async fn embed_batch(&self, texts: &[String]) -> VPKResult<Vec<Array1<f64>>> {
        let url = format!("{}/embed_batch", self.endpoint);
        let request = EmbedBatchRequest {
            texts: texts.to_vec(),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| VPKError::EmbeddingError(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(VPKError::EmbeddingError(format!(
                "Embedding server returned error: {}",
                response.status()
            )));
        }

        let embed_response: EmbedBatchResponse = response
            .json()
            .await
            .map_err(|e| VPKError::EmbeddingError(format!("Failed to parse response: {}", e)))?;

        if embed_response.dimension != self.dimension {
            return Err(VPKError::EmbeddingError(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension, embed_response.dimension
            )));
        }

        Ok(embed_response
            .embeddings
            .into_iter()
            .map(Array1::from_vec)
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dummy_embedding_deterministic() {
        let model = DummyEmbeddingModel::new(384);

        let text = "patient with lupus";
        let emb1 = model.embed(text).await.unwrap();
        let emb2 = model.embed(text).await.unwrap();

        // Same text should produce identical embeddings
        assert_eq!(emb1, emb2);
    }

    #[tokio::test]
    async fn test_dummy_embedding_dimension() {
        let model = DummyEmbeddingModel::new(384);
        let embedding = model.embed("test").await.unwrap();

        assert_eq!(embedding.len(), 384);
        assert_eq!(model.dimension(), 384);
    }

    #[tokio::test]
    async fn test_dummy_embedding_normalized() {
        let model = DummyEmbeddingModel::new(384);
        let embedding = model.embed("test").await.unwrap();

        // Verify L2 norm = 1.0 (unit vector)
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "Norm = {}, expected 1.0", norm);
    }

    #[tokio::test]
    async fn test_dummy_embedding_different_texts() {
        let model = DummyEmbeddingModel::new(384);

        let emb1 = model.embed("patient with lupus").await.unwrap();
        let emb2 = model.embed("diabetes treatment").await.unwrap();

        // Different texts should produce different embeddings
        assert_ne!(emb1, emb2);
    }

    #[tokio::test]
    async fn test_batch_embedding() {
        let model = DummyEmbeddingModel::new(384);

        let texts = vec![
            "patient with lupus".to_string(),
            "diabetes treatment".to_string(),
        ];

        let embeddings = model.embed_batch(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);

        // Verify normalized
        for emb in embeddings {
            let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }

    #[tokio::test]
    async fn test_embedding_consistency() {
        let model = DummyEmbeddingModel::new(384);

        // Batch and single should produce same results
        let text = "test document".to_string();
        let single_emb = model.embed(&text).await.unwrap();
        let batch_emb = model.embed_batch(&[text]).await.unwrap();

        assert_eq!(single_emb, batch_emb[0]);
    }
}
