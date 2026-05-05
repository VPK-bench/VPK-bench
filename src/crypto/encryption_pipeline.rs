//! Flexible Encryption Pipeline
//!
//! Allows dynamic selection and composition of encryption algorithms.
//!
//! # Design
//!
//! Each encryption algorithm implements the `EncryptionLayer` trait, and multiple
//! layers can be chained together in a pipeline to create composite encryption schemes.
//!
//! # Example Usage
//!
//! ```ignore
//! // Create pipeline with just scrambling
//! let pipeline = EncryptionPipeline::builder()
//!     .add_scrambling(384)
//!     .build()?;
//!
//! // Create pipeline with scrambling + noise
//! let pipeline = EncryptionPipeline::builder()
//!     .add_scrambling(384)
//!     .add_noise(0.8, 1.2)
//!     .build()?;
//!
//! // Encrypt documents and queries
//! let enc_doc = pipeline.encrypt_document(&doc)?;
//! let enc_query = pipeline.encrypt_query(&query)?;
//! ```

use crate::crypto::{Diehard, DimensionalScrambling, NoiseInjection, Rome};
use crate::error::{VPKError, VPKResult};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Trait for encryption layers that can be composed in a pipeline
pub trait EncryptionLayer: Send + Sync {
    /// Apply encryption to a document vector
    fn encrypt_document(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>>;

    /// Apply encryption to a query vector (may differ from document encryption)
    fn encrypt_query(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>>;

    /// Decrypt a vector (reverse the encryption)
    fn decrypt(&self, vector: &Array1<f64>, was_document: bool) -> VPKResult<Array1<f64>>;

    /// Get the vector dimension this layer operates on
    fn vector_dim(&self) -> usize;

    /// Get a unique identifier for this layer type
    fn layer_type(&self) -> &'static str;
}

/// Dimensional Scrambling layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScramblingLayer {
    scrambling: DimensionalScrambling,
}

impl ScramblingLayer {
    pub fn new(vector_dim: usize) -> VPKResult<Self> {
        Ok(Self {
            scrambling: DimensionalScrambling::new(vector_dim, 0)?,
        })
    }
}

impl EncryptionLayer for ScramblingLayer {
    fn encrypt_document(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        self.scrambling.encrypt(vector)
    }

    fn encrypt_query(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        self.scrambling.encrypt(vector)
    }

    fn decrypt(&self, vector: &Array1<f64>, _was_document: bool) -> VPKResult<Array1<f64>> {
        self.scrambling.decrypt(vector)
    }

    fn vector_dim(&self) -> usize {
        self.scrambling.vector_dim()
    }

    fn layer_type(&self) -> &'static str {
        "scrambling"
    }
}

/// Noise Injection layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseLayer {
    noise: NoiseInjection,
}

impl NoiseLayer {
    pub fn new(vector_dim: usize, noise_range: (f64, f64)) -> VPKResult<Self> {
        Ok(Self {
            noise: NoiseInjection::new(vector_dim, noise_range)?,
        })
    }

    pub fn new_default(vector_dim: usize) -> VPKResult<Self> {
        Self::new(vector_dim, (0.8, 1.2))
    }
}

impl EncryptionLayer for NoiseLayer {
    fn encrypt_document(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        // Documents get noise injected
        self.noise.inject_noise(vector)
    }

    fn encrypt_query(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        // Queries get denoised (reciprocal operation)
        self.noise.denoise_query(vector)
    }

    fn decrypt(&self, vector: &Array1<f64>, was_document: bool) -> VPKResult<Array1<f64>> {
        if was_document {
            // Reverse noise injection
            self.noise.denoise_query(vector)
        } else {
            // Reverse denoising (re-inject noise)
            self.noise.inject_noise(vector)
        }
    }

    fn vector_dim(&self) -> usize {
        self.noise.vector_dim()
    }

    fn layer_type(&self) -> &'static str {
        "noise"
    }
}

/// ROME (Random Orthogonal Matrix Encryption) layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RomeLayer {
    rome: Rome,
}

impl RomeLayer {
    /// Create a new ROME layer with specified input and padded dimensions
    ///
    /// # Parameters
    ///
    /// - `input_dim`: Original vector dimension
    /// - `padded_dim`: Padded dimension (must be > input_dim)
    ///
    /// # Recommended padding ratios
    ///
    /// - Light: `padded_dim = input_dim * 1.25` (25% overhead)
    /// - Medium: `padded_dim = input_dim * 1.5` (50% overhead)
    /// - Heavy: `padded_dim = input_dim * 2` (100% overhead)
    pub fn new(input_dim: usize, padded_dim: usize) -> VPKResult<Self> {
        Ok(Self {
            rome: Rome::new(input_dim, padded_dim)?,
        })
    }

    /// Create a new ROME layer with default 50% padding
    pub fn new_default(input_dim: usize) -> VPKResult<Self> {
        let padded_dim = input_dim + input_dim / 2;
        Self::new(input_dim, padded_dim)
    }

    /// Get the padded dimension (output dimension after ROME encryption)
    pub fn padded_dim(&self) -> usize {
        self.rome.padded_dim()
    }
}

impl EncryptionLayer for RomeLayer {
    fn encrypt_document(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        self.rome.encrypt(vector)
    }

    fn encrypt_query(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        // ROME uses same encryption for documents and queries (symmetric)
        self.rome.encrypt(vector)
    }

    fn decrypt(&self, vector: &Array1<f64>, _was_document: bool) -> VPKResult<Array1<f64>> {
        // ROME decryption is symmetric (doesn't depend on document vs query)
        self.rome.decrypt(vector)
    }

    fn vector_dim(&self) -> usize {
        // Note: ROME changes dimension from input_dim to padded_dim
        // This returns the OUTPUT dimension after encryption
        self.rome.padded_dim()
    }

    fn layer_type(&self) -> &'static str {
        "rome"
    }
}

/// ROMM (Random Orthogonal Matrix Multiplication) layer — ROME without zero-padding.
///
/// Uses a square orthogonal matrix Q ∈ ℝⁿˣⁿ applied directly to the input.
/// No dimension increase. Preserves inner products exactly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RommLayer {
    rome: Rome,
}

impl RommLayer {
    pub fn new(dim: usize) -> VPKResult<Self> {
        Ok(Self {
            rome: Rome::new_square(dim)?,
        })
    }
}

impl EncryptionLayer for RommLayer {
    fn encrypt_document(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        self.rome.encrypt(vector)
    }

    fn encrypt_query(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        self.rome.encrypt(vector)
    }

    fn decrypt(&self, vector: &Array1<f64>, _was_document: bool) -> VPKResult<Array1<f64>> {
        self.rome.decrypt(vector)
    }

    fn vector_dim(&self) -> usize {
        self.rome.input_dim()
    }

    fn layer_type(&self) -> &'static str {
        "romm"
    }
}

/// DIEHARD (Dual Independent Encryption) layer
///
/// Uses two different matrices for documents and queries (asymmetric encryption)
/// while preserving inner products exactly. Dimension increases from input_dim to output_dim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiehardLayer {
    diehard: Diehard,
}

impl DiehardLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> VPKResult<Self> {
        Ok(Self {
            diehard: Diehard::new(input_dim, output_dim)?,
        })
    }

    pub fn new_default(input_dim: usize) -> VPKResult<Self> {
        Ok(Self {
            diehard: Diehard::new_default(input_dim)?,
        })
    }

    pub fn output_dim(&self) -> usize {
        self.diehard.output_dim()
    }
}

impl EncryptionLayer for DiehardLayer {
    fn encrypt_document(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        self.diehard.encrypt_document(vector)
    }

    fn encrypt_query(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        self.diehard.encrypt_query(vector)
    }

    fn decrypt(&self, vector: &Array1<f64>, was_document: bool) -> VPKResult<Array1<f64>> {
        if was_document {
            self.diehard.decrypt_document(vector)
        } else {
            self.diehard.decrypt_query(vector)
        }
    }

    fn vector_dim(&self) -> usize {
        self.diehard.output_dim()
    }

    fn layer_type(&self) -> &'static str {
        "diehard"
    }
}

/// CKKS passthrough layer
///
/// With real CKKS (Microsoft SEAL), encryption produces opaque ciphertexts
/// that cannot be stored as f64 vectors in a standard vector DB. The actual
/// CKKS encrypt/decrypt happens in `CkksVectorStorage`.
///
/// This layer acts as an identity transform so the pipeline can be
/// constructed uniformly. The raw embedding passes through unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CkksLayer {
    input_dim: usize,
}

impl CkksLayer {
    pub fn new(input_dim: usize, _padded_dim: usize) -> VPKResult<Self> {
        Ok(Self { input_dim })
    }
}

impl EncryptionLayer for CkksLayer {
    fn encrypt_document(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        Ok(vector.clone())
    }

    fn encrypt_query(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        Ok(vector.clone())
    }

    fn decrypt(&self, vector: &Array1<f64>, _was_document: bool) -> VPKResult<Array1<f64>> {
        Ok(vector.clone())
    }

    fn vector_dim(&self) -> usize {
        self.input_dim
    }

    fn layer_type(&self) -> &'static str {
        "ckks"
    }
}

/// Serializable layer enum for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerConfig {
    Scrambling(ScramblingLayer),
    Noise(NoiseLayer),
    Rome(RomeLayer),
    Romm(RommLayer),
    Diehard(DiehardLayer),
    Ckks(CkksLayer),
}

impl LayerConfig {
    fn as_layer(&self) -> &dyn EncryptionLayer {
        match self {
            LayerConfig::Scrambling(l) => l,
            LayerConfig::Noise(l) => l,
            LayerConfig::Rome(l) => l,
            LayerConfig::Romm(l) => l,
            LayerConfig::Diehard(l) => l,
            LayerConfig::Ckks(l) => l,
        }
    }
}

/// Encryption pipeline that chains multiple encryption layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionPipeline {
    layers: Vec<LayerConfig>,
    vector_dim: usize,
}

impl EncryptionPipeline {
    /// Create a new builder for constructing a pipeline
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    /// Encrypt a document vector through the entire pipeline
    pub fn encrypt_document(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        let mut result = vector.clone();

        // Apply each layer in forward order
        for layer in &self.layers {
            result = layer.as_layer().encrypt_document(&result)?;
        }

        Ok(result)
    }

    /// Encrypt a query vector through the entire pipeline
    pub fn encrypt_query(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        let mut result = vector.clone();

        // Apply each layer in forward order
        for layer in &self.layers {
            result = layer.as_layer().encrypt_query(&result)?;
        }

        Ok(result)
    }

    /// Decrypt a vector by reversing the pipeline
    pub fn decrypt(&self, vector: &Array1<f64>, was_document: bool) -> VPKResult<Array1<f64>> {
        let mut result = vector.clone();

        // Apply each layer in reverse order
        for layer in self.layers.iter().rev() {
            result = layer.as_layer().decrypt(&result, was_document)?;
        }

        Ok(result)
    }

    /// Get the vector dimension
    pub fn vector_dim(&self) -> usize {
        self.vector_dim
    }

    /// Get the number of layers in the pipeline
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the layer types in order
    pub fn layer_types(&self) -> Vec<&'static str> {
        self.layers.iter().map(|l| l.as_layer().layer_type()).collect()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> VPKResult<Vec<u8>> {
        bincode::serialize(&self)
            .map_err(|e| VPKError::SerializationError(e.to_string()))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> VPKResult<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| VPKError::SerializationError(e.to_string()))
    }

    /// Replace only the noise layer with a freshly-seeded noise vector at
    /// a new range, leaving all other layers (scrambling, ROME, CKKS)
    /// completely unchanged.
    ///
    /// Used by Experiment G: freeze the DS matrix across a noise sweep so
    /// that noise and rotation effects can be measured independently.
    ///
    /// Returns `Err` if the pipeline contains no noise layer.
    pub fn reinit_noise_only(&mut self, noise_range: (f64, f64)) -> VPKResult<()> {
        for layer in &mut self.layers {
            if let LayerConfig::Noise(ref mut nl) = layer {
                let dim = <NoiseLayer as EncryptionLayer>::vector_dim(nl);
                *nl = NoiseLayer::new(dim, noise_range)?;
                return Ok(());
            }
        }
        Err(VPKError::KeyGenerationError(
            "Pipeline has no noise layer; cannot reinit noise without full rebuild".to_string(),
        ))
    }
}

/// Builder for constructing encryption pipelines
pub struct PipelineBuilder {
    layers: Vec<LayerConfig>,
    vector_dim: Option<usize>,
}

impl PipelineBuilder {
    /// Create a new empty builder
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            vector_dim: None,
        }
    }

    /// Add dimensional scrambling layer
    pub fn add_scrambling(mut self, vector_dim: usize) -> VPKResult<Self> {
        if let Some(dim) = self.vector_dim {
            if dim != vector_dim {
                return Err(VPKError::InvalidDimension {
                    expected: dim,
                    actual: vector_dim,
                });
            }
        } else {
            self.vector_dim = Some(vector_dim);
        }

        self.layers.push(LayerConfig::Scrambling(
            ScramblingLayer::new(vector_dim)?
        ));
        Ok(self)
    }

    /// Add noise injection layer with specified range
    pub fn add_noise(mut self, min: f64, max: f64) -> VPKResult<Self> {
        let vector_dim = self.vector_dim.ok_or_else(|| {
            VPKError::KeyGenerationError(
                "Must set vector dimension before adding noise layer".to_string()
            )
        })?;

        self.layers.push(LayerConfig::Noise(
            NoiseLayer::new(vector_dim, (min, max))?
        ));
        Ok(self)
    }

    /// Add noise injection layer with default range (0.8, 1.2)
    pub fn add_noise_default(mut self) -> VPKResult<Self> {
        let vector_dim = self.vector_dim.ok_or_else(|| {
            VPKError::KeyGenerationError(
                "Must set vector dimension before adding noise layer".to_string()
            )
        })?;

        self.layers.push(LayerConfig::Noise(
            NoiseLayer::new_default(vector_dim)?
        ));
        Ok(self)
    }

    /// Add ROME layer with specified padding dimension
    ///
    /// # Parameters
    ///
    /// - `input_dim`: Input vector dimension (must match previous layers)
    /// - `padded_dim`: Padded output dimension (must be > input_dim)
    ///
    /// # Note
    ///
    /// ROME changes the vector dimension. Subsequent layers should use `padded_dim`.
    pub fn add_rome(mut self, input_dim: usize, padded_dim: usize) -> VPKResult<Self> {
        if let Some(dim) = self.vector_dim {
            if dim != input_dim {
                return Err(VPKError::InvalidDimension {
                    expected: dim,
                    actual: input_dim,
                });
            }
        } else {
            self.vector_dim = Some(input_dim);
        }

        self.layers.push(LayerConfig::Rome(
            RomeLayer::new(input_dim, padded_dim)?
        ));

        // Update vector_dim to padded_dim for subsequent layers
        self.vector_dim = Some(padded_dim);

        Ok(self)
    }

    /// Add ROMM layer (ROME without zero-padding — square orthogonal matrix).
    ///
    /// Dimension does NOT change (Q ∈ ℝⁿˣⁿ applied in-place).
    pub fn add_romm(mut self, dim: usize) -> VPKResult<Self> {
        if let Some(d) = self.vector_dim {
            if d != dim {
                return Err(VPKError::InvalidDimension {
                    expected: d,
                    actual: dim,
                });
            }
        } else {
            self.vector_dim = Some(dim);
        }

        self.layers.push(LayerConfig::Romm(
            RommLayer::new(dim)?
        ));

        Ok(self)
    }

    /// Add CKKS passthrough layer.
    ///
    /// Real CKKS encryption happens in `CkksVectorStorage`, not here.
    /// This layer is an identity transform so the pipeline can still be
    /// constructed and introspected uniformly. The dimension does NOT
    /// change because the raw embedding passes through to the CKKS shard.
    pub fn add_ckks(mut self, input_dim: usize, _padded_dim: usize) -> VPKResult<Self> {
        if let Some(dim) = self.vector_dim {
            if dim != input_dim {
                return Err(VPKError::InvalidDimension {
                    expected: dim,
                    actual: input_dim,
                });
            }
        } else {
            self.vector_dim = Some(input_dim);
        }

        self.layers.push(LayerConfig::Ckks(
            CkksLayer::new(input_dim, 0)?
        ));

        Ok(self)
    }

    /// Add DIEHARD layer with specified output dimension
    ///
    /// # Parameters
    ///
    /// - `input_dim`: Input vector dimension (must match previous layers)
    /// - `output_dim`: Output dimension (must be ≥ input_dim)
    ///
    /// DIEHARD uses asymmetric encryption: different matrices for documents and queries.
    pub fn add_diehard(mut self, input_dim: usize, output_dim: usize) -> VPKResult<Self> {
        if let Some(dim) = self.vector_dim {
            if dim != input_dim {
                return Err(VPKError::InvalidDimension {
                    expected: dim,
                    actual: input_dim,
                });
            }
        } else {
            self.vector_dim = Some(input_dim);
        }

        self.layers.push(LayerConfig::Diehard(
            DiehardLayer::new(input_dim, output_dim)?,
        ));

        self.vector_dim = Some(output_dim);

        Ok(self)
    }

    /// Add DIEHARD layer with default ⌈1.5×⌉ output dimension
    pub fn add_diehard_default(self) -> VPKResult<Self> {
        let input_dim = self.vector_dim.ok_or_else(|| {
            VPKError::KeyGenerationError(
                "Must set vector dimension before adding DIEHARD layer".to_string(),
            )
        })?;

        let output_dim = (input_dim as f64 * 1.5).ceil() as usize;
        self.add_diehard(input_dim, output_dim)
    }

    /// Add ROME layer with default 50% padding
    pub fn add_rome_default(self) -> VPKResult<Self> {
        let input_dim = self.vector_dim.ok_or_else(|| {
            VPKError::KeyGenerationError(
                "Must set vector dimension before adding ROME layer".to_string()
            )
        })?;

        let padded_dim = input_dim + input_dim / 2;
        self.add_rome(input_dim, padded_dim)
    }

    /// Build the pipeline
    pub fn build(self) -> VPKResult<EncryptionPipeline> {
        let vector_dim = self.vector_dim.ok_or_else(|| {
            VPKError::KeyGenerationError(
                "Must add at least one layer to the pipeline".to_string()
            )
        })?;

        if self.layers.is_empty() {
            return Err(VPKError::KeyGenerationError(
                "Pipeline must have at least one layer".to_string()
            ));
        }

        Ok(EncryptionPipeline {
            layers: self.layers,
            vector_dim,
        })
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn normalize(v: &Array1<f64>) -> Array1<f64> {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v / norm
    }

    fn cosine_similarity(v1: &Array1<f64>, v2: &Array1<f64>) -> f64 {
        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
        dot / (norm1 * norm2)
    }

    #[test]
    fn test_scrambling_only_pipeline() {
        println!("\n=== Pipeline: Scrambling Only ===\n");

        let pipeline = EncryptionPipeline::builder()
            .add_scrambling(50).unwrap()
            .build().unwrap();

        assert_eq!(pipeline.num_layers(), 1);
        assert_eq!(pipeline.layer_types(), vec!["scrambling"]);

        let doc = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));
        let query = normalize(&Array1::from_vec((0..50).map(|i| i as f64).collect()));

        let enc_doc = pipeline.encrypt_document(&doc).unwrap();
        let enc_query = pipeline.encrypt_query(&query).unwrap();

        // Should preserve unit length
        let norm = enc_doc.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-9);

        // Should preserve exact similarity
        let plain_sim = cosine_similarity(&query, &doc);
        let enc_sim = cosine_similarity(&enc_query, &enc_doc);

        println!("Plaintext similarity: {:.6}", plain_sim);
        println!("Encrypted similarity: {:.6}", enc_sim);
        assert_abs_diff_eq!(plain_sim, enc_sim, epsilon = 1e-10);

        println!("✓ Scrambling-only pipeline preserves exact similarity");
    }

    #[test]
    fn test_noise_only_pipeline() {
        println!("\n=== Pipeline: Noise Only ===\n");

        let mut builder = EncryptionPipeline::builder();
        builder.vector_dim = Some(50);  // Set dimension first
        let pipeline = builder
            .add_noise(0.9, 1.1).unwrap()
            .build().unwrap();

        assert_eq!(pipeline.num_layers(), 1);
        assert_eq!(pipeline.layer_types(), vec!["noise"]);

        let doc = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));
        let query = normalize(&Array1::from_vec((0..50).map(|i| i as f64).collect()));

        let enc_doc = pipeline.encrypt_document(&doc).unwrap();
        let enc_query = pipeline.encrypt_query(&query).unwrap();

        let plain_sim = cosine_similarity(&query, &doc);
        let enc_sim = cosine_similarity(&enc_query, &enc_doc);

        println!("Plaintext similarity: {:.6}", plain_sim);
        println!("Encrypted similarity: {:.6}", enc_sim);
        println!("Difference: {:.6}", (plain_sim - enc_sim).abs());

        // With low noise, should be close
        assert!((plain_sim - enc_sim).abs() < 0.1, "Should be approximately similar");

        println!("✓ Noise-only pipeline preserves approximate similarity");
    }

    #[test]
    fn test_combined_pipeline_scrambling_then_noise() {
        println!("\n=== Pipeline: Scrambling → Noise ===\n");

        let pipeline = EncryptionPipeline::builder()
            .add_scrambling(50).unwrap()
            .add_noise(0.9, 1.1).unwrap()
            .build().unwrap();

        assert_eq!(pipeline.num_layers(), 2);
        assert_eq!(pipeline.layer_types(), vec!["scrambling", "noise"]);

        let doc = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));
        let query = normalize(&Array1::from_vec((0..50).map(|i| i as f64).collect()));

        let enc_doc = pipeline.encrypt_document(&doc).unwrap();
        let enc_query = pipeline.encrypt_query(&query).unwrap();

        let plain_sim = cosine_similarity(&query, &doc);
        let enc_sim = cosine_similarity(&enc_query, &enc_doc);

        println!("Plaintext similarity: {:.6}", plain_sim);
        println!("Encrypted similarity: {:.6}", enc_sim);
        println!("Difference: {:.6}", (plain_sim - enc_sim).abs());

        println!("✓ Combined pipeline works");
    }

    #[test]
    fn test_pipeline_ranking_preservation() {
        println!("\n=== Testing Ranking Preservation ===\n");

        // Test different pipeline configurations
        let configs = vec![
            ("Scrambling only", vec!["scrambling"]),
            ("Noise only", vec!["noise"]),
            ("Scrambling + Noise", vec!["scrambling", "noise"]),
        ];

        for (name, layers) in configs {
            let mut builder = EncryptionPipeline::builder();
            builder.vector_dim = Some(50);

            for layer in layers {
                match layer {
                    "scrambling" => builder = builder.add_scrambling(50).unwrap(),
                    "noise" => builder = builder.add_noise(0.9, 1.1).unwrap(),
                    _ => panic!("Unknown layer"),
                }
            }

            let pipeline = builder.build().unwrap();

            let query = normalize(&Array1::from_vec((0..50).map(|i| i as f64).collect()));
            let doc1 = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));
            let doc2 = normalize(&Array1::from_vec((0..50).map(|i| (49 - i) as f64).collect()));

            let plain_sim1 = cosine_similarity(&query, &doc1);
            let plain_sim2 = cosine_similarity(&query, &doc2);

            let enc_query = pipeline.encrypt_query(&query).unwrap();
            let enc_doc1 = pipeline.encrypt_document(&doc1).unwrap();
            let enc_doc2 = pipeline.encrypt_document(&doc2).unwrap();

            let enc_sim1 = cosine_similarity(&enc_query, &enc_doc1);
            let enc_sim2 = cosine_similarity(&enc_query, &enc_doc2);

            let plain_order = plain_sim1 > plain_sim2;
            let enc_order = enc_sim1 > enc_sim2;
            let preserved = plain_order == enc_order;

            println!("{}: ranking {}", name, if preserved { "preserved ✓" } else { "NOT preserved ✗" });
        }
    }

    #[test]
    fn test_decrypt() {
        let pipeline = EncryptionPipeline::builder()
            .add_scrambling(50).unwrap()
            .add_noise(0.95, 1.05).unwrap() // Very low noise
            .build().unwrap();

        let doc = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));

        let encrypted = pipeline.encrypt_document(&doc).unwrap();
        let decrypted = pipeline.decrypt(&encrypted, true).unwrap();

        let similarity = cosine_similarity(&doc, &decrypted);
        println!("Decrypt similarity: {:.6}", similarity);

        assert!(similarity > 0.99, "Should recover with high similarity");
    }

    #[test]
    fn test_serialization() {
        let pipeline = EncryptionPipeline::builder()
            .add_scrambling(100).unwrap()
            .add_noise_default().unwrap()
            .build().unwrap();

        let bytes = pipeline.to_bytes().unwrap();
        let restored = EncryptionPipeline::from_bytes(&bytes).unwrap();

        assert_eq!(pipeline.num_layers(), restored.num_layers());
        assert_eq!(pipeline.vector_dim(), restored.vector_dim());

        // Test that encryption works the same
        let doc = normalize(&Array1::from_vec((0..100).map(|i| (i + 1) as f64).collect()));

        let enc1 = pipeline.encrypt_document(&doc).unwrap();
        let enc2 = restored.encrypt_document(&doc).unwrap();

        for i in 0..100 {
            assert_abs_diff_eq!(enc1[i], enc2[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_empty_pipeline_error() {
        let result = EncryptionPipeline::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let result = EncryptionPipeline::builder()
            .add_scrambling(50).unwrap()
            .add_scrambling(100); // Different dimension

        assert!(result.is_err());
    }

    #[test]
    fn test_rome_only_pipeline() {
        println!("\n=== Pipeline: ROME Only ===\n");

        let pipeline = EncryptionPipeline::builder()
            .add_rome(50, 75).unwrap()
            .build().unwrap();

        assert_eq!(pipeline.num_layers(), 1);
        assert_eq!(pipeline.layer_types(), vec!["rome"]);

        let doc = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));
        let query = normalize(&Array1::from_vec((0..50).map(|i| i as f64).collect()));

        let enc_doc = pipeline.encrypt_document(&doc).unwrap();
        let enc_query = pipeline.encrypt_query(&query).unwrap();

        // ROME preserves exact similarity
        let plain_sim = cosine_similarity(&query, &doc);
        let enc_sim = cosine_similarity(&enc_query, &enc_doc);

        println!("Plaintext similarity: {:.6}", plain_sim);
        println!("Encrypted similarity: {:.6}", enc_sim);
        assert_abs_diff_eq!(plain_sim, enc_sim, epsilon = 1e-10);

        // ROME changes dimension
        assert_eq!(enc_doc.len(), 75);
        assert_eq!(enc_query.len(), 75);

        println!("✓ ROME-only pipeline preserves exact similarity");
    }

    #[test]
    fn test_rome_with_default_padding() {
        println!("\n=== Pipeline: ROME with Default Padding ===\n");

        let mut builder = EncryptionPipeline::builder();
        builder.vector_dim = Some(100);
        let pipeline = builder
            .add_rome_default().unwrap()
            .build().unwrap();

        let doc = normalize(&Array1::from_vec((0..100).map(|i| (i + 1) as f64).collect()));

        let enc_doc = pipeline.encrypt_document(&doc).unwrap();

        // Default padding is 50%
        assert_eq!(enc_doc.len(), 150);

        println!("✓ ROME default padding (50%) works correctly");
    }

    #[test]
    fn test_combined_pipeline_rome_then_noise() {
        println!("\n=== Pipeline: ROME → Noise ===\n");

        let pipeline = EncryptionPipeline::builder()
            .add_rome(50, 75).unwrap()
            .add_noise(0.9, 1.1).unwrap()  // Noise operates on 75-dim output
            .build().unwrap();

        assert_eq!(pipeline.num_layers(), 2);
        assert_eq!(pipeline.layer_types(), vec!["rome", "noise"]);

        let doc = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));
        let query = normalize(&Array1::from_vec((0..50).map(|i| i as f64).collect()));

        let enc_doc = pipeline.encrypt_document(&doc).unwrap();
        let enc_query = pipeline.encrypt_query(&query).unwrap();

        let plain_sim = cosine_similarity(&query, &doc);
        let enc_sim = cosine_similarity(&enc_query, &enc_doc);

        println!("Plaintext similarity: {:.6}", plain_sim);
        println!("Encrypted similarity: {:.6}", enc_sim);
        println!("Difference: {:.6}", (plain_sim - enc_sim).abs());

        // Combined pipeline: exact from ROME, slight variation from noise
        assert!((plain_sim - enc_sim).abs() < 0.1, "Should be approximately similar");

        println!("✓ ROME + Noise combined pipeline works");
    }

    #[test]
    fn test_combined_pipeline_scrambling_rome_noise() {
        println!("\n=== Pipeline: Scrambling → ROME → Noise (Defense-in-Depth) ===\n");

        let pipeline = EncryptionPipeline::builder()
            .add_scrambling(50).unwrap()        // Permute dimensions
            .add_rome(50, 75).unwrap()          // Apply ROME (dimension change)
            .add_noise(0.95, 1.05).unwrap()     // Add subtle noise on 75-dim
            .build().unwrap();

        assert_eq!(pipeline.num_layers(), 3);
        assert_eq!(pipeline.layer_types(), vec!["scrambling", "rome", "noise"]);

        let doc = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));
        let query = normalize(&Array1::from_vec((0..50).map(|i| i as f64).collect()));

        let enc_doc = pipeline.encrypt_document(&doc).unwrap();
        let enc_query = pipeline.encrypt_query(&query).unwrap();

        let plain_sim = cosine_similarity(&query, &doc);
        let enc_sim = cosine_similarity(&enc_query, &enc_doc);

        println!("Plaintext similarity: {:.6}", plain_sim);
        println!("Encrypted similarity: {:.6}", enc_sim);
        println!("Difference: {:.6}", (plain_sim - enc_sim).abs());

        // Output is 75-dimensional
        assert_eq!(enc_doc.len(), 75);

        println!("✓ Triple-layer defense-in-depth pipeline works");
    }

    #[test]
    fn test_rome_ranking_preservation() {
        println!("\n=== ROME Pipeline: Ranking Preservation ===\n");

        let pipeline = EncryptionPipeline::builder()
            .add_rome(50, 75).unwrap()
            .build().unwrap();

        let query = normalize(&Array1::from_vec((0..50).map(|i| i as f64).collect()));
        let doc1 = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));
        let doc2 = normalize(&Array1::from_vec((0..50).map(|i| (49 - i) as f64).collect()));

        let plain_sim1 = cosine_similarity(&query, &doc1);
        let plain_sim2 = cosine_similarity(&query, &doc2);

        let enc_query = pipeline.encrypt_query(&query).unwrap();
        let enc_doc1 = pipeline.encrypt_document(&doc1).unwrap();
        let enc_doc2 = pipeline.encrypt_document(&doc2).unwrap();

        let enc_sim1 = cosine_similarity(&enc_query, &enc_doc1);
        let enc_sim2 = cosine_similarity(&enc_query, &enc_doc2);

        let plain_order = plain_sim1 > plain_sim2;
        let enc_order = enc_sim1 > enc_sim2;

        println!("Plain: doc1={:.6} > doc2={:.6} : {}", plain_sim1, plain_sim2, plain_order);
        println!("Enc:   doc1={:.6} > doc2={:.6} : {}", enc_sim1, enc_sim2, enc_order);

        assert_eq!(plain_order, enc_order, "ROME must preserve ranking exactly");

        println!("✓ ROME preserves ranking with 100% accuracy");
    }

    #[test]
    fn test_rome_decrypt() {
        let pipeline = EncryptionPipeline::builder()
            .add_rome(50, 75).unwrap()
            .build().unwrap();

        let doc = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));

        let encrypted = pipeline.encrypt_document(&doc).unwrap();
        let decrypted = pipeline.decrypt(&encrypted, true).unwrap();

        // Should recover original vector exactly
        for i in 0..50 {
            assert_abs_diff_eq!(doc[i], decrypted[i], epsilon = 1e-9);
        }
    }
}
