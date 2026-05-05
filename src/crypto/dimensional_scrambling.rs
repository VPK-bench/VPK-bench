//! Dimensional Scrambling for Partial Homomorphic Encryption
//!
//! Pure dimensional scrambling that preserves cosine similarity EXACTLY through:
//! 1. Input vectors MUST be normalized to unit length (L2 norm = 1)
//! 2. Random permutation (remapping) of vector elements
//!
//! **CRITICAL REQUIREMENT**: All vectors must be normalized to unit vectors before encryption.
//! This ensures that cosine similarity is preserved EXACTLY:
//!   cos(E(v1), E(v2)) = cos(v1, v2)
//!   rank(cos(v1, q)) = rank(cos(E(v1), E(q)))  [EXACT preservation]
//!
//! The encryption scheme:
//!   E(v) = permute(v, π)  where v is a unit vector and π is a random permutation
//!
//! For search, compute cosine similarity on encrypted vectors:
//!   similarity(E(v1), E(v2)) = dot(E(v1), E(v2)) / (||E(v1)|| × ||E(v2)||)
//!                            = dot(v1, v2) / (||v1|| × ||v2||)
//!                            = similarity(v1, v2)  [EXACT]

use crate::error::{VPKError, VPKResult};
use ndarray::Array1;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Dimensional Scrambling encryption engine
#[derive(Clone, Serialize, Deserialize)]
pub struct DimensionalScrambling {
    /// Permutation mapping: encrypted[i] = plaintext[permutation[i]]
    permutation: Vec<usize>,

    /// Inverse permutation for decryption: plaintext[i] = encrypted[inverse_permutation[i]]
    inverse_permutation: Vec<usize>,

    /// Vector dimension (unchanged by encryption)
    vector_dim: usize,
}

impl fmt::Debug for DimensionalScrambling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DimensionalScrambling")
            .field("vector_dim", &self.vector_dim)
            .field("permutation", &"<redacted>")
            .field("inverse_permutation", &"<redacted>")
            .finish()
    }
}

impl DimensionalScrambling {
    /// Create a new Dimensional Scrambling instance with random permutation
    ///
    /// # Arguments
    /// * `vector_dim` - Vector dimension (unchanged by encryption)
    /// * `_padding_dim` - Deprecated parameter, kept for API compatibility (ignored)
    ///
    /// # Returns
    /// A new DimensionalScrambling instance ready for encryption/decryption
    pub fn new(vector_dim: usize, _padding_dim: usize) -> VPKResult<Self> {
        if vector_dim == 0 {
            return Err(VPKError::KeyGenerationError(
                "vector_dim must be greater than 0".to_string(),
            ));
        }

        let mut rng = rand::thread_rng();

        // Generate random permutation
        let mut permutation: Vec<usize> = (0..vector_dim).collect();
        permutation.shuffle(&mut rng);

        // Compute inverse permutation for decryption
        let mut inverse_permutation = vec![0; vector_dim];
        for (i, &p) in permutation.iter().enumerate() {
            inverse_permutation[p] = i;
        }

        Ok(Self {
            permutation,
            inverse_permutation,
            vector_dim,
        })
    }

    /// Encrypt a vector using dimensional scrambling
    ///
    /// Algorithm: E(v) = permute(v, π)
    /// where π is a random permutation
    ///
    /// # Arguments
    /// * `vector` - The plaintext vector to encrypt (must be `vector_dim` dimensions)
    ///             **MUST be normalized to unit length (L2 norm ≈ 1.0)**
    ///
    /// # Returns
    /// Encrypted vector of `vector_dim` dimensions (same dimension as input)
    ///
    /// # Important
    /// This function assumes the input vector is already normalized.
    /// Call `normalize()` before encryption if needed.
    ///
    /// # Homomorphism Property
    /// For unit vectors: cos(E(v1), E(v2)) = cos(v1, v2) EXACTLY
    pub fn encrypt(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        if vector.len() != self.vector_dim {
            return Err(VPKError::InvalidDimension {
                expected: self.vector_dim,
                actual: vector.len(),
            });
        }

        // Verify vector is normalized (L2 norm ≈ 1.0)
        let norm: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        if (norm - 1.0).abs() > 1e-6 {
            return Err(VPKError::EncryptionError(format!(
                "Input vector must be normalized (L2 norm = 1.0), got norm = {:.6}",
                norm
            )));
        }

        // Apply permutation: encrypted[i] = plaintext[permutation[i]]
        let mut encrypted = Array1::zeros(self.vector_dim);
        for (i, &perm_idx) in self.permutation.iter().enumerate() {
            encrypted[i] = vector[perm_idx];
        }

        Ok(encrypted)
    }

    /// Decrypt a vector using dimensional scrambling
    ///
    /// Algorithm: D(e) = inverse_permute(e, π^-1)
    ///
    /// # Arguments
    /// * `encrypted` - The encrypted vector to decrypt (must be `vector_dim` dimensions)
    ///
    /// # Returns
    /// Decrypted vector of `vector_dim` dimensions (same as original)
    pub fn decrypt(&self, encrypted: &Array1<f64>) -> VPKResult<Array1<f64>> {
        if encrypted.len() != self.vector_dim {
            return Err(VPKError::InvalidDimension {
                expected: self.vector_dim,
                actual: encrypted.len(),
            });
        }

        // Apply inverse permutation: plaintext[i] = encrypted[inverse_permutation[i]]
        let mut decrypted = Array1::zeros(self.vector_dim);
        for (i, &inv_perm_idx) in self.inverse_permutation.iter().enumerate() {
            decrypted[i] = encrypted[inv_perm_idx];
        }

        Ok(decrypted)
    }

    /// Get the vector dimension (unchanged by encryption)
    pub fn vector_dim(&self) -> usize {
        self.vector_dim
    }

    /// Get the padded dimension (deprecated, returns same as vector_dim for compatibility)
    #[deprecated(note = "Pure dimensional scrambling does not use padding")]
    pub fn padded_dim(&self) -> usize {
        self.vector_dim
    }

    /// Serialize the scrambling parameters to bytes
    pub fn to_bytes(&self) -> VPKResult<Vec<u8>> {
        bincode::serialize(&self)
            .map_err(|e| VPKError::SerializationError(e.to_string()))
    }

    /// Deserialize scrambling parameters from bytes
    pub fn from_bytes(bytes: &[u8]) -> VPKResult<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| VPKError::SerializationError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_encryption_decryption_roundtrip() {
        let ds = DimensionalScrambling::new(384, 128).unwrap();

        // Create a vector and normalize it
        let unnormalized = Array1::from_vec((0..384).map(|i| (i + 1) as f64).collect());
        let norm: f64 = unnormalized.iter().map(|x| x * x).sum::<f64>().sqrt();
        let original = &unnormalized / norm;

        // Verify it's normalized
        let check_norm: f64 = original.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_abs_diff_eq!(check_norm, 1.0, epsilon = 1e-10);

        let encrypted = ds.encrypt(&original).unwrap();
        assert_eq!(encrypted.len(), 384); // Same dimension (no padding)

        let decrypted = ds.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted.len(), 384);

        // Check roundtrip (should be exact)
        for i in 0..384 {
            assert_abs_diff_eq!(decrypted[i], original[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_similarity_ranking_preservation() {
        // Test that dimensional scrambling preserves cosine similarity EXACTLY
        // when operating on unit vectors (normalized)
        let ds = DimensionalScrambling::new(10, 5).unwrap();

        // Helper function to normalize a vector
        let normalize = |v: &Array1<f64>| -> Array1<f64> {
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v / norm
        };

        // Helper function to compute cosine similarity
        let cosine_sim = |v1: &Array1<f64>, v2: &Array1<f64>| -> f64 {
            let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
            let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
            dot / (norm1 * norm2)
        };

        // Create and normalize test vectors
        let query_raw = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let doc1_raw = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]); // Very similar
        let doc2_raw = Array1::from_vec(vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]); // Less similar
        let doc3_raw = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]); // Medium similarity

        let query = normalize(&query_raw);
        let doc1 = normalize(&doc1_raw);
        let doc2 = normalize(&doc2_raw);
        let doc3 = normalize(&doc3_raw);

        // Compute plaintext cosine similarities
        let plain_sim1 = cosine_sim(&query, &doc1);
        let plain_sim2 = cosine_sim(&query, &doc2);
        let plain_sim3 = cosine_sim(&query, &doc3);

        // Create plaintext ranking
        let mut plain_ranking = vec![
            (1, plain_sim1),
            (2, plain_sim2),
            (3, plain_sim3),
        ];
        plain_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let plain_order: Vec<usize> = plain_ranking.iter().map(|(id, _)| *id).collect();

        // Encrypt all normalized vectors
        let eq = ds.encrypt(&query).unwrap();
        let e1 = ds.encrypt(&doc1).unwrap();
        let e2 = ds.encrypt(&doc2).unwrap();
        let e3 = ds.encrypt(&doc3).unwrap();

        // Compute encrypted cosine similarities
        let encrypted_sim1 = cosine_sim(&eq, &e1);
        let encrypted_sim2 = cosine_sim(&eq, &e2);
        let encrypted_sim3 = cosine_sim(&eq, &e3);

        // Create encrypted ranking
        let mut encrypted_ranking = vec![
            (1, encrypted_sim1),
            (2, encrypted_sim2),
            (3, encrypted_sim3),
        ];
        encrypted_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let encrypted_order: Vec<usize> = encrypted_ranking.iter().map(|(id, _)| *id).collect();

        // The ranking order MUST be preserved EXACTLY for encrypted search to work
        assert_eq!(
            plain_order, encrypted_order,
            "Cosine similarity ranking must be preserved after encryption.\nPlaintext: {:?}\nEncrypted: {:?}",
            plain_ranking, encrypted_ranking
        );

        // Also verify the expected order: doc1 > doc3 > doc2
        assert_eq!(plain_order, vec![1, 3, 2]);

        // Verify cosine similarities are EXACTLY preserved (within floating point precision)
        for (i, &(plain_id, plain_sim)) in plain_ranking.iter().enumerate() {
            let (enc_id, enc_sim) = encrypted_ranking[i];
            assert_eq!(plain_id, enc_id);
            assert_abs_diff_eq!(plain_sim, enc_sim, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_permutation_valid() {
        let ds = DimensionalScrambling::new(100, 50).unwrap();

        // Check permutation is valid (contains each index exactly once)
        let mut seen = vec![false; 100];
        for &idx in &ds.permutation {
            assert!(idx < 100, "Permutation index {} out of range", idx);
            assert!(!seen[idx], "Permutation index {} appears twice", idx);
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&x| x), "Permutation is missing some indices");

        // Check inverse permutation is valid
        let mut seen_inv = vec![false; 100];
        for &idx in &ds.inverse_permutation {
            assert!(idx < 100, "Inverse permutation index {} out of range", idx);
            assert!(!seen_inv[idx], "Inverse permutation index {} appears twice", idx);
            seen_inv[idx] = true;
        }

        // Check permutation and inverse are actually inverses
        for i in 0..100 {
            let j = ds.permutation[i];
            let k = ds.inverse_permutation[j];
            assert_eq!(k, i, "Permutation and inverse don't match at index {}", i);
        }
    }

    #[test]
    fn test_invalid_dimension() {
        let ds = DimensionalScrambling::new(10, 5).unwrap();

        // Create a normalized vector with wrong dimension
        let wrong_vector_raw = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let norm: f64 = wrong_vector_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let wrong_vector = &wrong_vector_raw / norm;
        let result = ds.encrypt(&wrong_vector);
        assert!(result.is_err());

        // Wrong encrypted dimension
        let wrong_encrypted = Array1::from_vec(vec![1.0; 5]);
        let result = ds.decrypt(&wrong_encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn test_requires_normalized_input() {
        let ds = DimensionalScrambling::new(10, 5).unwrap();

        // Create a non-normalized vector
        let unnormalized = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        // Should fail because it's not normalized
        let result = ds.encrypt(&unnormalized);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("normalized"));
    }

    #[test]
    fn test_serialization() {
        let ds = DimensionalScrambling::new(384, 128).unwrap();

        let bytes = ds.to_bytes().unwrap();
        let restored = DimensionalScrambling::from_bytes(&bytes).unwrap();

        // Test that encryption/decryption works the same
        let unnormalized = Array1::from_vec((0..384).map(|i| (i + 1) as f64).collect());
        let norm: f64 = unnormalized.iter().map(|x| x * x).sum::<f64>().sqrt();
        let vector = &unnormalized / norm;

        let encrypted1 = ds.encrypt(&vector).unwrap();
        let encrypted2 = restored.encrypt(&vector).unwrap();

        // Should produce identical results
        for i in 0..384 {
            assert_abs_diff_eq!(encrypted1[i], encrypted2[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_zero_vector() {
        let ds = DimensionalScrambling::new(10, 5).unwrap();

        // Zero vector cannot be normalized (division by zero)
        // So encryption should fail
        let zero = Array1::zeros(10);
        let result = ds.encrypt(&zero);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("normalized"));

        // Note: In practice, zero vectors should never occur in embeddings
        // since they represent no semantic content
    }

    #[test]
    fn test_invalid_parameters() {
        // Zero vector dimension
        assert!(DimensionalScrambling::new(0, 10).is_err());

        // Note: padding_dim parameter is kept for API compatibility but ignored
        // Pure dimensional scrambling doesn't use padding
        assert!(DimensionalScrambling::new(10, 0).is_ok());
    }
}
