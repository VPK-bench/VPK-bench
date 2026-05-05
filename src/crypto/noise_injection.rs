//! Noise Injection for Additional Security
//!
//! Multiplies vectors by a secret noise vector, with a corresponding denoise operation
//! applied to queries. This approach, rooted in classical signal processing, can further
//! obscure data while maintaining search accuracy.
//!
//! # Algorithm
//!
//! 1. **Noise Vector Generation**: Create a secret noise vector n with elements that
//!    preserve (approximately) the unit length of input vectors
//!
//! 2. **Noise Injection**: For document vectors d:
//!    ```text
//!    E(d) = normalize(d ⊙ n)
//!    ```
//!    where ⊙ is element-wise multiplication
//!
//! 3. **Denoising**: For query vectors q:
//!    ```text
//!    D(q) = normalize(q ⊙ (1/n))
//!    ```
//!    where 1/n is the element-wise reciprocal
//!
//! # Properties
//!
//! - Preserves unit length (||E(d)|| = 1)
//! - Maintains cosine similarity rankings (approximately)
//! - Adds security through element-wise obfuscation
//! - Limited by floating point precision
//!
//! # Security
//!
//! The noise vector n acts as a secret key. Without knowing n, an adversary cannot:
//! - Recover the original vector from the noisy vector
//! - Denoise queries to perform accurate searches
//!
//! # Reference
//!
//! See the accompanying workshop paper for full derivation.

use crate::error::{VPKError, VPKResult};
use ndarray::Array1;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Noise Injection engine for vector obfuscation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseInjection {
    /// Secret noise vector (element-wise multiplier)
    noise_vector: Array1<f64>,

    /// Reciprocal noise vector (for denoising)
    denoise_vector: Array1<f64>,

    /// Vector dimension
    vector_dim: usize,
}

impl NoiseInjection {
    /// Create a new NoiseInjection instance with a random noise vector
    ///
    /// The noise vector is generated with elements distributed around 1.0
    /// to approximately preserve unit length after element-wise multiplication.
    ///
    /// # Arguments
    ///
    /// * `vector_dim` - Dimension of vectors to be processed
    /// * `noise_range` - Controls noise strength: (min, max) range for noise elements.
    ///                   Default recommendation: (0.7, 1.3) for moderate noise
    ///                   Smaller range (e.g., 0.9, 1.1): less noise, better accuracy
    ///                   Larger range (e.g., 0.5, 1.5): more noise, more obfuscation
    ///
    /// # Returns
    ///
    /// A new NoiseInjection instance ready for encryption/decryption
    ///
    /// # Constraints
    ///
    /// - Noise elements must be non-zero (to allow reciprocals)
    /// - Range should be positive to avoid sign changes
    /// - Floating point precision limits the practical range
    pub fn new(vector_dim: usize, noise_range: (f64, f64)) -> VPKResult<Self> {
        if vector_dim == 0 {
            return Err(VPKError::KeyGenerationError(
                "vector_dim must be greater than 0".to_string(),
            ));
        }

        let (min_noise, max_noise) = noise_range;
        if min_noise <= 0.0 || max_noise < min_noise {
            return Err(VPKError::KeyGenerationError(
                format!("Invalid noise range ({}, {}): must be positive and min <= max",
                        min_noise, max_noise),
            ));
        }

        let mut rng = rand::thread_rng();

        // Generate random noise vector with elements in the specified range.
        // When min == max (degenerate / zero-noise case) fill with that constant.
        let noise_vector = Array1::from_vec(
            (0..vector_dim)
                .map(|_| {
                    if (max_noise - min_noise).abs() < f64::EPSILON {
                        min_noise
                    } else {
                        rng.gen_range(min_noise..max_noise)
                    }
                })
                .collect(),
        );

        // Compute reciprocals for denoising
        let denoise_vector = noise_vector.mapv(|n| 1.0 / n);

        Ok(Self {
            noise_vector,
            denoise_vector,
            vector_dim,
        })
    }

    /// Create a NoiseInjection instance with default noise range (0.7, 1.3)
    ///
    /// This provides a good balance between security and accuracy.
    pub fn new_default(vector_dim: usize) -> VPKResult<Self> {
        Self::new(vector_dim, (0.7, 1.3))
    }

    /// Apply noise injection to a vector (for document vectors)
    ///
    /// # Algorithm
    ///
    /// 1. Element-wise multiply: v' = v ⊙ noise_vector
    /// 2. Normalize to unit length: E(v) = v' / ||v'||
    ///
    /// # Arguments
    ///
    /// * `vector` - The plaintext vector to inject noise into (must be unit length)
    ///
    /// # Returns
    ///
    /// Noisy vector of the same dimension (still unit length)
    pub fn inject_noise(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        if vector.len() != self.vector_dim {
            return Err(VPKError::InvalidDimension {
                expected: self.vector_dim,
                actual: vector.len(),
            });
        }

        // Element-wise multiply with noise vector
        let noisy = vector * &self.noise_vector;

        // Normalize to maintain unit length
        let noisy_norm: f64 = noisy.iter().map(|x| x * x).sum::<f64>().sqrt();
        let normalized = &noisy / noisy_norm;

        Ok(normalized)
    }

    /// Apply denoising to a query vector (for query vectors)
    ///
    /// # Algorithm
    ///
    /// 1. Element-wise multiply by reciprocals: q' = q ⊙ denoise_vector
    /// 2. Normalize to unit length: D(q) = q' / ||q'||
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector to denoise (must be unit length)
    ///
    /// # Returns
    ///
    /// Denoised query vector (still unit length)
    pub fn denoise_query(&self, query: &Array1<f64>) -> VPKResult<Array1<f64>> {
        if query.len() != self.vector_dim {
            return Err(VPKError::InvalidDimension {
                expected: self.vector_dim,
                actual: query.len(),
            });
        }

        // Element-wise multiply with denoise vector (reciprocals)
        let denoised = query * &self.denoise_vector;

        // Normalize to maintain unit length
        let denoised_norm: f64 = denoised.iter().map(|x| x * x).sum::<f64>().sqrt();
        let normalized = &denoised / denoised_norm;

        Ok(normalized)
    }

    /// Get the vector dimension
    pub fn vector_dim(&self) -> usize {
        self.vector_dim
    }

    /// Get the noise range statistics (min, max, mean)
    pub fn noise_stats(&self) -> (f64, f64, f64) {
        let min = self.noise_vector.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.noise_vector.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = self.noise_vector.mean().unwrap_or(0.0);
        (min, max, mean)
    }

    /// Serialize the noise parameters to bytes
    pub fn to_bytes(&self) -> VPKResult<Vec<u8>> {
        bincode::serialize(&self)
            .map_err(|e| VPKError::SerializationError(e.to_string()))
    }

    /// Deserialize noise parameters from bytes
    pub fn from_bytes(bytes: &[u8]) -> VPKResult<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| VPKError::SerializationError(e.to_string()))
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
    fn test_noise_injection_preserves_unit_length() {
        let noise = NoiseInjection::new_default(100).unwrap();

        let v = normalize(&Array1::from_vec((0..100).map(|i| (i + 1) as f64).collect()));
        let noisy = noise.inject_noise(&v).unwrap();

        let norm = noisy.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_denoise_preserves_unit_length() {
        let noise = NoiseInjection::new_default(100).unwrap();

        let q = normalize(&Array1::from_vec((0..100).map(|i| (i + 1) as f64).collect()));
        let denoised = noise.denoise_query(&q).unwrap();

        let norm = denoised.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_noise_then_denoise_approximate_recovery() {
        let noise = NoiseInjection::new(50, (0.9, 1.1)).unwrap(); // Small noise

        let v = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));

        // Apply noise
        let noisy = noise.inject_noise(&v).unwrap();

        // Apply denoise (simulating query denoising)
        let denoised = noise.denoise_query(&noisy).unwrap();

        // Should be close to original (but not exact due to normalization)
        let similarity = cosine_similarity(&v, &denoised);
        println!("Similarity after noise+denoise: {:.6}", similarity);

        // With small noise (0.9, 1.1), should be very close
        assert!(similarity > 0.99, "Similarity should be >0.99, got {}", similarity);
    }

    #[test]
    fn test_search_with_noise_injection() {
        let noise = NoiseInjection::new_default(50).unwrap();

        // Create query and documents
        let q = normalize(&Array1::from_vec((0..50).map(|i| i as f64).collect()));
        let d1 = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));
        let d2 = normalize(&Array1::from_vec((0..50).map(|i| (49 - i) as f64).collect()));

        // Plaintext similarities
        let sim_q_d1_plain = cosine_similarity(&q, &d1);
        let sim_q_d2_plain = cosine_similarity(&q, &d2);

        println!("Plaintext similarities:");
        println!("  cos(q, d1) = {:.6}", sim_q_d1_plain);
        println!("  cos(q, d2) = {:.6}", sim_q_d2_plain);

        // Apply noise to documents
        let d1_noisy = noise.inject_noise(&d1).unwrap();
        let d2_noisy = noise.inject_noise(&d2).unwrap();

        // Denoise query
        let q_denoised = noise.denoise_query(&q).unwrap();

        // Similarities with noise
        let sim_q_d1_noisy = cosine_similarity(&q_denoised, &d1_noisy);
        let sim_q_d2_noisy = cosine_similarity(&q_denoised, &d2_noisy);

        println!("\nWith noise injection:");
        println!("  cos(D(q), N(d1)) = {:.6}", sim_q_d1_noisy);
        println!("  cos(D(q), N(d2)) = {:.6}", sim_q_d2_noisy);

        // Ranking should be preserved
        let plain_order = sim_q_d1_plain > sim_q_d2_plain;
        let noisy_order = sim_q_d1_noisy > sim_q_d2_noisy;

        assert_eq!(plain_order, noisy_order, "Ranking should be preserved");
        println!("\n✓ Ranking preserved with noise injection");
    }

    #[test]
    fn test_noise_range_effects() {
        // Test different noise ranges
        let ranges = vec![
            (0.95, 1.05, "Very low noise"),
            (0.9, 1.1, "Low noise"),
            (0.8, 1.2, "Medium noise"),
            (0.7, 1.3, "Default noise"),
            (0.5, 1.5, "High noise"),
        ];

        let v = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));

        for (min, max, label) in ranges {
            let noise = NoiseInjection::new(50, (min, max)).unwrap();
            let noisy = noise.inject_noise(&v).unwrap();
            let denoised = noise.denoise_query(&noisy).unwrap();

            let similarity = cosine_similarity(&v, &denoised);
            println!("{}: noise=({}, {}), recovery={:.4}", label, min, max, similarity);
        }
    }

    #[test]
    fn test_invalid_noise_range() {
        // Zero in range
        assert!(NoiseInjection::new(10, (0.0, 1.0)).is_err());

        // Negative values
        assert!(NoiseInjection::new(10, (-0.5, 0.5)).is_err());

        // Min > Max
        assert!(NoiseInjection::new(10, (1.5, 0.5)).is_err());
    }

    #[test]
    fn test_invalid_dimension() {
        let noise = NoiseInjection::new_default(10).unwrap();

        // Wrong dimension
        let wrong_vector = normalize(&Array1::from_vec(vec![1.0, 2.0, 3.0]));
        assert!(noise.inject_noise(&wrong_vector).is_err());
        assert!(noise.denoise_query(&wrong_vector).is_err());
    }

    #[test]
    fn test_serialization() {
        let noise = NoiseInjection::new_default(100).unwrap();

        let bytes = noise.to_bytes().unwrap();
        let restored = NoiseInjection::from_bytes(&bytes).unwrap();

        // Test that noise/denoise works the same
        let v = normalize(&Array1::from_vec((0..100).map(|i| (i + 1) as f64).collect()));

        let noisy1 = noise.inject_noise(&v).unwrap();
        let noisy2 = restored.inject_noise(&v).unwrap();

        // Should produce identical results
        for i in 0..100 {
            assert_abs_diff_eq!(noisy1[i], noisy2[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_noise_stats() {
        let noise = NoiseInjection::new(100, (0.7, 1.3)).unwrap();
        let (min, max, mean) = noise.noise_stats();

        println!("Noise statistics: min={:.3}, max={:.3}, mean={:.3}", min, max, mean);

        assert!(min >= 0.7 && min < max);
        assert!(max <= 1.3 && max > min);
        assert!(mean > 0.7 && mean < 1.3);
    }
}
