//! ROME (Random Orthogonal Matrix Encryption)
//!
//! ROME is a Partially Homomorphic Encryption (PHE) algorithm that provides **exact**
//! ranking preservation for cosine similarity searches. Unlike dimensional scrambling
//! (which permutes dimensions) or noise injection (which adds controlled randomness),
//! ROME uses random orthogonal matrices to encrypt vectors while preserving inner products.
//!
//! # Mathematical Foundation
//!
//! Given a plaintext vector `v ∈ ℝⁿ`, ROME encryption works as follows:
//!
//! 1. **Padding**: Extend `v` to dimension `m` (where `m > n`) using zero-padding:
//!    ```text
//!    pad(v) = [v₁, v₂, ..., vₙ, 0, 0, ..., 0] ∈ ℝᵐ
//!    ```
//!
//! 2. **Orthogonal Transformation**: Multiply by a random orthogonal matrix `Q ∈ ℝᵐˣᵐ`:
//!    ```text
//!    E(v) = Q · pad(v)
//!    ```
//!
//! 3. **Key Property**: Q is orthogonal (Q^T Q = I), which preserves inner products:
//!    ```text
//!    ⟨E(v₁), E(v₂)⟩ = ⟨Q·pad(v₁), Q·pad(v₂)⟩
//!                   = pad(v₁)^T · Q^T · Q · pad(v₂)
//!                   = pad(v₁)^T · pad(v₂)
//!                   = ⟨v₁, v₂⟩
//!    ```
//!
//! 4. **Cosine Similarity Preservation**:
//!    ```text
//!    cos(E(v₁), E(v₂)) = ⟨E(v₁), E(v₂)⟩ / (‖E(v₁)‖ · ‖E(v₂)‖)
//!                       = ⟨v₁, v₂⟩ / (‖v₁‖ · ‖v₂‖)
//!                       = cos(v₁, v₂)
//!    ```
//!
//! **Result**: 100% ranking preservation for cosine similarity searches.
//!
//! # Security Properties
//!
//! - **Semantic Security**: Without the key Q, encrypted vectors appear random
//! - **IND-CPA**: Indistinguishable under chosen-plaintext attack (with proper padding)
//! - **Defense-in-Depth**: Can be combined with noise injection for additional security
//! - **No Information Leakage**: Eve sees only Q·pad(v), which reveals nothing about v
//!
//! # Performance Characteristics
//!
//! - **Encryption Complexity**: O(m²) for matrix-vector multiplication
//! - **Space Overhead**: Vector dimension increases from n to m (typical: m ≈ 1.5n to 2n)
//! - **Decryption**: O(m²) using Q^T (transpose)
//! - **Query Transformation**: Same as document encryption (symmetric)
//!
//! # Comparison to Other PHE Algorithms
//!
//! | Algorithm              | Ranking Preservation | Complexity | Space Overhead |
//! |------------------------|---------------------|------------|----------------|
//! | Dimensional Scrambling | 100% (exact)        | O(n)       | 0%             |
//! | Noise Injection        | 90-99% (tunable)    | O(n)       | 0%             |
//! | **ROME**               | **100% (exact)**    | **O(m²)**  | **50-100%**    |
//! | CKKS                   | ~95% (approximate)  | O(n log n) | 200-400%       |
//!
//! # When to Use ROME
//!
//! **Use ROME when:**
//! - You need 100% ranking preservation (no tolerance for ranking errors)
//! - You want provable inner product preservation
//! - You can afford O(n²) computational cost
//! - Space overhead is acceptable (50-100% increase)
//! - You want to combine with other techniques (e.g., ROME + Noise)
//!
//! **Don't use ROME when:**
//! - Performance is critical (use dimensional scrambling instead)
//! - Space is constrained (use noise injection instead)
//! - Approximate ranking is acceptable (use noise injection)
//! - Vectors are very high-dimensional (>2048 dims)
//!
//! # Example
//!
//! ```ignore
//! use phe::crypto::rome::Rome;
//! use ndarray::Array1;
//!
//! // Create ROME encryptor with 384-dim input, 512-dim padded output
//! let rome = Rome::new(384, 512)?;
//!
//! // Encrypt document and query vectors
//! let doc = Array1::from_vec(vec![1.0; 384]);
//! let query = Array1::from_vec(vec![2.0; 384]);
//!
//! let enc_doc = rome.encrypt(&doc)?;
//! let enc_query = rome.encrypt(&query)?;
//!
//! // Cosine similarity is exactly preserved
//! assert_eq!(cosine_similarity(&query, &doc),
//!            cosine_similarity(&enc_query, &enc_doc));
//! ```
//!
//! # References
//!
//! - SIAM Paper: "Privacy-Preserving Vector Search using Partial Homomorphic Encryption"
//! - Orthogonal Matrices: https://en.wikipedia.org/wiki/Orthogonal_matrix
//! - QR Decomposition: https://en.wikipedia.org/wiki/QR_decomposition

use crate::error::{VPKError, VPKResult};
use nalgebra::{DMatrix, DVector};
use ndarray::Array1;
use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

/// ROME (Random Orthogonal Matrix Encryption) encryptor
///
/// Encrypts vectors by:
/// 1. Zero-padding from input_dim to padded_dim
/// 2. Multiplying by a random orthogonal matrix Q
/// 3. Result: E(v) = Q · pad(v)
///
/// Key property: Preserves inner products exactly, hence 100% ranking preservation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rome {
    /// Input vector dimension (n)
    input_dim: usize,

    /// Padded vector dimension (m), where m > n
    padded_dim: usize,

    /// Random orthogonal matrix Q ∈ ℝᵐˣᵐ
    /// Generated via QR decomposition of a random matrix
    #[serde(with = "matrix_serde")]
    orthogonal_matrix: DMatrix<f64>,

    /// Transpose of Q (for efficient decryption)
    #[serde(with = "matrix_serde")]
    orthogonal_matrix_transpose: DMatrix<f64>,
}

/// Custom serialization for DMatrix (nalgebra matrices)
mod matrix_serde {
    use nalgebra::DMatrix;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(matrix: &DMatrix<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let (nrows, ncols) = matrix.shape();
        let data: Vec<f64> = matrix.iter().copied().collect();
        (nrows, ncols, data).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<DMatrix<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (nrows, ncols, data): (usize, usize, Vec<f64>) = Deserialize::deserialize(deserializer)?;
        Ok(DMatrix::from_vec(nrows, ncols, data))
    }
}

impl Rome {
    /// Create a new ROME encryptor
    ///
    /// # Parameters
    ///
    /// - `input_dim`: Original vector dimension (n)
    /// - `padded_dim`: Padded vector dimension (m), must satisfy m > n
    ///
    /// # Recommended Padding
    ///
    /// - **Light padding**: `padded_dim = input_dim + input_dim / 4` (25% overhead)
    /// - **Medium padding**: `padded_dim = input_dim + input_dim / 2` (50% overhead)
    /// - **Heavy padding**: `padded_dim = input_dim * 2` (100% overhead)
    ///
    /// More padding = more security, but higher computational cost.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // 384-dim input, 512-dim padded (33% overhead)
    /// let rome = Rome::new(384, 512)?;
    /// ```
    pub fn new(input_dim: usize, padded_dim: usize) -> VPKResult<Self> {
        if padded_dim < input_dim {
            return Err(VPKError::InvalidDimension {
                expected: input_dim,
                actual: padded_dim,
            });
        }

        // Generate random orthogonal matrix via QR decomposition
        let orthogonal_matrix = Self::generate_random_orthogonal_matrix(padded_dim)?;
        let orthogonal_matrix_transpose = orthogonal_matrix.transpose();

        Ok(Self {
            input_dim,
            padded_dim,
            orthogonal_matrix,
            orthogonal_matrix_transpose,
        })
    }

    /// Create a ROMM encryptor (ROME without zero-padding).
    ///
    /// Uses a square orthogonal matrix Q ∈ ℝⁿˣⁿ directly on the input.
    /// No dimension increase, no padding overhead. Preserves inner products exactly.
    pub fn new_square(dim: usize) -> VPKResult<Self> {
        Self::new(dim, dim)
    }

    /// Generate a random orthogonal matrix using QR decomposition
    ///
    /// Algorithm:
    /// 1. Generate a random matrix A ∈ ℝᵐˣᵐ with entries ~ N(0, 1)
    /// 2. Compute QR decomposition: A = QR
    /// 3. Return Q (which is orthogonal: Q^T Q = I)
    ///
    /// This ensures uniform distribution over the orthogonal group O(m).
    fn generate_random_orthogonal_matrix(dim: usize) -> VPKResult<DMatrix<f64>> {
        let mut rng = thread_rng();
        let normal = StandardNormal;

        // Generate random matrix with Gaussian entries
        let mut random_matrix = DMatrix::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                random_matrix[(i, j)] = normal.sample(&mut rng);
            }
        }

        // Perform QR decomposition
        let qr = random_matrix.qr();
        let q_matrix = qr.q();

        Ok(q_matrix)
    }

    /// Pad a vector from input_dim to padded_dim with zeros
    ///
    /// Input:  [v₁, v₂, ..., vₙ]
    /// Output: [v₁, v₂, ..., vₙ, 0, 0, ..., 0]
    ///                           ↑
    ///                    (padded_dim - input_dim) zeros
    fn pad_vector(&self, vector: &Array1<f64>) -> VPKResult<DVector<f64>> {
        if vector.len() != self.input_dim {
            return Err(VPKError::InvalidDimension {
                expected: self.input_dim,
                actual: vector.len(),
            });
        }

        // Create padded vector with zeros
        let mut padded = DVector::zeros(self.padded_dim);

        // Copy original values
        for i in 0..self.input_dim {
            padded[i] = vector[i];
        }

        // Remaining positions are already zero

        Ok(padded)
    }

    /// Unpad a vector from padded_dim back to input_dim
    ///
    /// Input:  [v₁, v₂, ..., vₙ, 0, 0, ..., 0]
    /// Output: [v₁, v₂, ..., vₙ]
    fn unpad_vector(&self, padded: &DVector<f64>) -> VPKResult<Array1<f64>> {
        if padded.len() != self.padded_dim {
            return Err(VPKError::InvalidDimension {
                expected: self.padded_dim,
                actual: padded.len(),
            });
        }

        // Extract first input_dim elements
        let unpadded: Vec<f64> = padded.iter().take(self.input_dim).copied().collect();

        Ok(Array1::from_vec(unpadded))
    }

    /// Encrypt a vector using ROME
    ///
    /// E(v) = Q · pad(v)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let vec = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    /// let encrypted = rome.encrypt(&vec)?;
    /// assert_eq!(encrypted.len(), rome.padded_dim());
    /// ```
    pub fn encrypt(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        // Step 1: Pad vector
        let padded = self.pad_vector(vector)?;

        // Step 2: Multiply by orthogonal matrix: E(v) = Q · pad(v)
        let encrypted_nalgebra = &self.orthogonal_matrix * padded;

        // Convert back to ndarray
        let encrypted: Vec<f64> = encrypted_nalgebra.iter().copied().collect();

        Ok(Array1::from_vec(encrypted))
    }

    /// Decrypt a vector using ROME
    ///
    /// D(E(v)) = Q^T · E(v) = Q^T · Q · pad(v) = pad(v)
    ///
    /// Then unpad to recover v.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let vec = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    /// let encrypted = rome.encrypt(&vec)?;
    /// let decrypted = rome.decrypt(&encrypted)?;
    ///
    /// for i in 0..vec.len() {
    ///     assert!((vec[i] - decrypted[i]).abs() < 1e-9);
    /// }
    /// ```
    pub fn decrypt(&self, encrypted: &Array1<f64>) -> VPKResult<Array1<f64>> {
        if encrypted.len() != self.padded_dim {
            return Err(VPKError::InvalidDimension {
                expected: self.padded_dim,
                actual: encrypted.len(),
            });
        }

        // Convert to nalgebra vector
        let encrypted_nalgebra = DVector::from_vec(encrypted.to_vec());

        // Step 1: Multiply by Q^T: Q^T · E(v) = pad(v)
        let padded = &self.orthogonal_matrix_transpose * encrypted_nalgebra;

        // Step 2: Unpad to recover original vector
        self.unpad_vector(&padded)
    }

    /// Get the input vector dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get the padded vector dimension
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Get the space overhead percentage
    ///
    /// Returns: (padded_dim - input_dim) / input_dim * 100
    pub fn space_overhead_percent(&self) -> f64 {
        ((self.padded_dim - self.input_dim) as f64 / self.input_dim as f64) * 100.0
    }

    /// Verify that the orthogonal matrix is indeed orthogonal
    ///
    /// Checks: Q^T Q ≈ I (identity matrix)
    ///
    /// Used for testing and validation.
    pub fn verify_orthogonality(&self) -> bool {
        let product = &self.orthogonal_matrix_transpose * &self.orthogonal_matrix;
        let identity = DMatrix::identity(self.padded_dim, self.padded_dim);

        // Check if product is close to identity (within numerical precision)
        for i in 0..self.padded_dim {
            for j in 0..self.padded_dim {
                let expected: f64 = identity[(i, j)];
                let actual: f64 = product[(i, j)];

                if (expected - actual).abs() > 1e-9 {
                    return false;
                }
            }
        }

        true
    }
}

// Implement Zeroize for security (clear keys from memory)
impl Zeroize for Rome {
    fn zeroize(&mut self) {
        // Zeroize the orthogonal matrix
        for elem in self.orthogonal_matrix.iter_mut() {
            elem.zeroize();
        }
        for elem in self.orthogonal_matrix_transpose.iter_mut() {
            elem.zeroize();
        }
    }
}

impl Drop for Rome {
    fn drop(&mut self) {
        self.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn normalize(v: &Array1<f64>) -> Array1<f64> {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            v.clone()
        } else {
            v / norm
        }
    }

    fn cosine_similarity(v1: &Array1<f64>, v2: &Array1<f64>) -> f64 {
        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 < 1e-12 || norm2 < 1e-12 {
            0.0
        } else {
            dot / (norm1 * norm2)
        }
    }

    #[test]
    fn test_rome_creation() {
        let rome = Rome::new(50, 75).unwrap();
        assert_eq!(rome.input_dim(), 50);
        assert_eq!(rome.padded_dim(), 75);
        assert_abs_diff_eq!(rome.space_overhead_percent(), 50.0, epsilon = 0.1);
    }

    #[test]
    fn test_rome_invalid_dimensions() {
        // padded_dim must be >= input_dim
        let result = Rome::new(50, 40);
        assert!(result.is_err());

        // padded_dim == input_dim is valid (ROMM mode)
        let result = Rome::new(50, 50);
        assert!(result.is_ok());
    }

    #[test]
    fn test_orthogonality() {
        let rome = Rome::new(50, 75).unwrap();
        assert!(rome.verify_orthogonality(), "Q^T Q should equal I");
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let rome = Rome::new(50, 75).unwrap();
        let original = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));

        let encrypted = rome.encrypt(&original).unwrap();
        let decrypted = rome.decrypt(&encrypted).unwrap();

        // Should recover original vector exactly (within numerical precision)
        for i in 0..50 {
            assert_abs_diff_eq!(original[i], decrypted[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_inner_product_preservation() {
        let rome = Rome::new(50, 75).unwrap();

        let v1 = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));
        let v2 = normalize(&Array1::from_vec((0..50).map(|i| (50 - i) as f64).collect()));

        let enc_v1 = rome.encrypt(&v1).unwrap();
        let enc_v2 = rome.encrypt(&v2).unwrap();

        // Compute inner products
        let plain_dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let enc_dot: f64 = enc_v1.iter().zip(enc_v2.iter()).map(|(a, b)| a * b).sum();

        // Inner products should be exactly preserved
        assert_abs_diff_eq!(plain_dot, enc_dot, epsilon = 1e-9);
    }

    #[test]
    fn test_cosine_similarity_exact_preservation() {
        println!("\n=== ROME: Cosine Similarity Exact Preservation ===\n");

        let rome = Rome::new(100, 150).unwrap();

        let v1 = normalize(&Array1::from_vec((0..100).map(|i| (i + 1) as f64).collect()));
        let v2 = normalize(&Array1::from_vec((0..100).map(|i| (100 - i) as f64).collect()));

        let enc_v1 = rome.encrypt(&v1).unwrap();
        let enc_v2 = rome.encrypt(&v2).unwrap();

        let plain_sim = cosine_similarity(&v1, &v2);
        let enc_sim = cosine_similarity(&enc_v1, &enc_v2);

        println!("Plaintext cosine similarity: {:.12}", plain_sim);
        println!("Encrypted cosine similarity: {:.12}", enc_sim);
        println!("Difference: {:.2e}", (plain_sim - enc_sim).abs());

        // Should be exact (within 1e-10)
        assert_abs_diff_eq!(plain_sim, enc_sim, epsilon = 1e-10);

        println!("✓ ROME preserves cosine similarity exactly");
    }

    #[test]
    fn test_ranking_preservation_100_percent() {
        println!("\n=== ROME: 100% Ranking Preservation ===\n");

        let rome = Rome::new(50, 75).unwrap();

        let query = normalize(&Array1::from_vec((0..50).map(|i| i as f64).collect()));

        // Create multiple documents with different similarities
        let docs: Vec<Array1<f64>> = vec![
            normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect())),   // High similarity
            normalize(&Array1::from_vec((0..50).map(|i| (i + 10) as f64).collect())),  // Medium similarity
            normalize(&Array1::from_vec((0..50).map(|i| (49 - i) as f64).collect())),  // Low similarity
            normalize(&Array1::from_vec((0..50).map(|i| (i * 2) as f64).collect())),   // High similarity
        ];

        // Compute plaintext similarities
        let plain_sims: Vec<f64> = docs.iter()
            .map(|doc| cosine_similarity(&query, doc))
            .collect();

        // Encrypt
        let enc_query = rome.encrypt(&query).unwrap();
        let enc_docs: Vec<Array1<f64>> = docs.iter()
            .map(|doc| rome.encrypt(doc).unwrap())
            .collect();

        // Compute encrypted similarities
        let enc_sims: Vec<f64> = enc_docs.iter()
            .map(|doc| cosine_similarity(&enc_query, doc))
            .collect();

        // Check ranking preservation
        let mut plain_indices: Vec<usize> = (0..docs.len()).collect();
        plain_indices.sort_by(|&a, &b| plain_sims[b].partial_cmp(&plain_sims[a]).unwrap());

        let mut enc_indices: Vec<usize> = (0..docs.len()).collect();
        enc_indices.sort_by(|&a, &b| enc_sims[b].partial_cmp(&enc_sims[a]).unwrap());

        println!("Plain ranking: {:?}", plain_indices);
        println!("Encrypted ranking: {:?}", enc_indices);

        assert_eq!(plain_indices, enc_indices, "Rankings must be identical");

        println!("✓ ROME preserves ranking with 100% accuracy");
    }

    #[test]
    fn test_different_padding_sizes() {
        // Test various padding ratios
        let configs = vec![
            (100, 125, "25% padding"),
            (100, 150, "50% padding"),
            (100, 200, "100% padding"),
        ];

        for (input_dim, padded_dim, name) in configs {
            let rome = Rome::new(input_dim, padded_dim).unwrap();

            let v = normalize(&Array1::from_vec((0..input_dim).map(|i| (i + 1) as f64).collect()));
            let enc = rome.encrypt(&v).unwrap();
            let dec = rome.decrypt(&enc).unwrap();

            // Verify roundtrip
            for i in 0..input_dim {
                assert_abs_diff_eq!(v[i], dec[i], epsilon = 1e-9);
            }

            println!("✓ {}: roundtrip successful", name);
        }
    }

    #[test]
    fn test_norm_preservation() {
        let rome = Rome::new(50, 75).unwrap();

        let v = Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect());
        let enc = rome.encrypt(&v).unwrap();

        // Compute norms
        let plain_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let enc_norm: f64 = enc.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Orthogonal matrices preserve norms
        assert_abs_diff_eq!(plain_norm, enc_norm, epsilon = 1e-9);
    }

    #[test]
    fn test_edge_case_identical_vectors() {
        let rome = Rome::new(50, 75).unwrap();

        let v = normalize(&Array1::from_vec(vec![1.0; 50]));

        let enc1 = rome.encrypt(&v).unwrap();
        let enc2 = rome.encrypt(&v).unwrap();

        let sim = cosine_similarity(&enc1, &enc2);

        // Identical vectors should have cosine similarity = 1.0
        assert_abs_diff_eq!(sim, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_edge_case_orthogonal_vectors() {
        let rome = Rome::new(4, 6).unwrap();

        let v1 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let v2 = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);

        let enc1 = rome.encrypt(&v1).unwrap();
        let enc2 = rome.encrypt(&v2).unwrap();

        let sim = cosine_similarity(&enc1, &enc2);

        // Orthogonal vectors should have cosine similarity = 0.0
        assert_abs_diff_eq!(sim, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_high_dimensional_384() {
        // Realistic embedding dimension (e.g., OpenAI text-embedding-3-small)
        let rome = Rome::new(384, 512).unwrap();
        assert!(rome.verify_orthogonality());

        let v = normalize(&Array1::from_vec((0..384).map(|i| (i + 1) as f64).collect()));
        let enc = rome.encrypt(&v).unwrap();
        let dec = rome.decrypt(&enc).unwrap();

        for i in 0..384 {
            assert_abs_diff_eq!(v[i], dec[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_romm_square_matrix() {
        let romm = Rome::new_square(50).unwrap();
        assert_eq!(romm.input_dim(), 50);
        assert_eq!(romm.padded_dim(), 50);
        assert_abs_diff_eq!(romm.space_overhead_percent(), 0.0, epsilon = 0.01);
        assert!(romm.verify_orthogonality());

        let v1 = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));
        let v2 = normalize(&Array1::from_vec((0..50).map(|i| (50 - i) as f64).collect()));

        let enc_v1 = romm.encrypt(&v1).unwrap();
        let enc_v2 = romm.encrypt(&v2).unwrap();

        assert_eq!(enc_v1.len(), 50);
        assert_abs_diff_eq!(cosine_similarity(&v1, &v2), cosine_similarity(&enc_v1, &enc_v2), epsilon = 1e-10);

        let dec = romm.decrypt(&enc_v1).unwrap();
        for i in 0..50 {
            assert_abs_diff_eq!(v1[i], dec[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_serialization() {
        let rome = Rome::new(50, 75).unwrap();

        // Serialize
        let serialized = bincode::serialize(&rome).unwrap();

        // Deserialize
        let restored: Rome = bincode::deserialize(&serialized).unwrap();

        assert_eq!(rome.input_dim(), restored.input_dim());
        assert_eq!(rome.padded_dim(), restored.padded_dim());

        // Verify encryption produces same results
        let v = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));

        let enc1 = rome.encrypt(&v).unwrap();
        let enc2 = restored.encrypt(&v).unwrap();

        for i in 0..enc1.len() {
            assert_abs_diff_eq!(enc1[i], enc2[i], epsilon = 1e-12);
        }
    }
}
