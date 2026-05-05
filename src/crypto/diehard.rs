//! DIEHARD (Dual Independent Encryption for Hardened Asymmetric Retrieval Defense)
//!
//! DIEHARD is a PHE algorithm that uses **two distinct linear maps** for documents
//! and queries, preserving inner products exactly while making it harder for an
//! adversary to correlate encrypted documents with encrypted queries.
//!
//! # Mathematical Foundation
//!
//! Given plaintext vectors in ℝⁿ, DIEHARD maps them to ℝᵐ (m ≥ n):
//!
//! 1. **Key generation**: Build two m×n matrices A and B such that A^T B = Iₙ.
//!    - Generate a random m×n matrix, QR-decompose, take Q[:, :n] as A (orthonormal columns).
//!    - Set B = A + P_perp · B_hat, where P_perp = I_m − A A^+ projects onto the
//!      null space of A^T. This guarantees A^T B = A^T A = I while B ≠ A.
//!
//! 2. **Document encryption**: E_doc(d) = B · d
//! 3. **Query encryption**:   E_query(q) = A · q
//!
//! 4. **Inner product preservation**:
//!    ```text
//!    ⟨E_query(q), E_doc(d)⟩ = (Aq)^T (Bd) = q^T A^T B d = q^T d = ⟨q, d⟩
//!    ```
//!
//! # Difference from ROME
//!
//! ROME uses a single symmetric orthogonal matrix Q for both documents and queries.
//! DIEHARD uses **asymmetric** keys: documents and queries are encrypted with
//! different matrices (B and A), so even if an attacker obtains encrypted documents,
//! they cannot directly compute encrypted queries or vice versa.

use crate::error::{VPKError, VPKResult};
use nalgebra::{DMatrix, DVector};
use ndarray::Array1;
use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diehard {
    /// Input vector dimension (n)
    input_dim: usize,

    /// Output vector dimension (m), where m ≥ n (default: ⌈1.5n⌉)
    output_dim: usize,

    /// Query encryption matrix A ∈ ℝᵐˣⁿ (orthonormal columns)
    #[serde(with = "matrix_serde")]
    matrix_a: DMatrix<f64>,

    /// Document encryption matrix B ∈ ℝᵐˣⁿ where A^T B = Iₙ
    #[serde(with = "matrix_serde")]
    matrix_b: DMatrix<f64>,
}

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

impl Diehard {
    /// Create a new DIEHARD encryptor.
    ///
    /// # Parameters
    ///
    /// - `input_dim`: Original vector dimension (n)
    /// - `output_dim`: Encrypted vector dimension (m), must satisfy m ≥ n
    ///
    /// The default for `output_dim` is ⌈1.5 × input_dim⌉, giving 50% overhead.
    pub fn new(input_dim: usize, output_dim: usize) -> VPKResult<Self> {
        if output_dim < input_dim {
            return Err(VPKError::InvalidDimension {
                expected: input_dim,
                actual: output_dim,
            });
        }

        let (matrix_a, matrix_b) = Self::generate_key_pair(input_dim, output_dim)?;

        Ok(Self {
            input_dim,
            output_dim,
            matrix_a,
            matrix_b,
        })
    }

    /// Create with default output dimension: ⌈1.5 × input_dim⌉.
    pub fn new_default(input_dim: usize) -> VPKResult<Self> {
        let output_dim = (input_dim as f64 * 1.5).ceil() as usize;
        Self::new(input_dim, output_dim)
    }

    /// Generate the asymmetric key pair (A, B) such that A^T B = I.
    ///
    /// 1. Sample random m×n matrix, QR-decompose → take first n columns as A.
    /// 2. Compute P_perp = I_m − A A^+  (projection onto null-space of A^T).
    /// 3. B = A + P_perp · B_hat  for random B_hat, so A^T B = A^T A = I.
    fn generate_key_pair(
        n: usize,
        m: usize,
    ) -> VPKResult<(DMatrix<f64>, DMatrix<f64>)> {
        let mut rng = thread_rng();
        let normal = StandardNormal;

        // Step 1: Generate random m×n matrix and QR-decompose
        let mut a_hat = DMatrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                a_hat[(i, j)] = normal.sample(&mut rng);
            }
        }
        let qr = a_hat.qr();
        let q_full = qr.q(); // m×m orthogonal
        let a = q_full.columns(0, n).into_owned(); // m×n with orthonormal columns

        // Step 2: P_perp = I_m − A (A^T A)^{-1} A^T  =  I_m − A A^T  (since A has orthonormal cols)
        let identity_m = DMatrix::identity(m, m);
        let p_perp = &identity_m - &a * a.transpose();

        // Step 3: B = A + P_perp · B_hat
        let mut b_hat = DMatrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                b_hat[(i, j)] = normal.sample(&mut rng);
            }
        }
        let b = &a + &p_perp * &b_hat;

        Ok((a, b))
    }

    /// Encrypt a document vector: E_doc(d) = B · d
    pub fn encrypt_document(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        if vector.len() != self.input_dim {
            return Err(VPKError::InvalidDimension {
                expected: self.input_dim,
                actual: vector.len(),
            });
        }

        let v = DVector::from_vec(vector.to_vec());
        let encrypted = &self.matrix_b * v;
        Ok(Array1::from_vec(encrypted.iter().copied().collect()))
    }

    /// Encrypt a query vector: E_query(q) = A · q
    pub fn encrypt_query(&self, vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        if vector.len() != self.input_dim {
            return Err(VPKError::InvalidDimension {
                expected: self.input_dim,
                actual: vector.len(),
            });
        }

        let v = DVector::from_vec(vector.to_vec());
        let encrypted = &self.matrix_a * v;
        Ok(Array1::from_vec(encrypted.iter().copied().collect()))
    }

    /// Decrypt a document vector using B's pseudo-inverse: d = A^T · E_doc(d)
    ///
    /// Since A^T B = I, we have A^T (B d) = d.
    pub fn decrypt_document(&self, encrypted: &Array1<f64>) -> VPKResult<Array1<f64>> {
        if encrypted.len() != self.output_dim {
            return Err(VPKError::InvalidDimension {
                expected: self.output_dim,
                actual: encrypted.len(),
            });
        }

        let e = DVector::from_vec(encrypted.to_vec());
        let decrypted = self.matrix_a.transpose() * e;
        Ok(Array1::from_vec(decrypted.iter().copied().collect()))
    }

    /// Decrypt a query vector using A's pseudo-inverse: q = B^T · E_query(q)
    ///
    /// Since B^T A = (A^T B)^T = I, we have B^T (A q) = q.
    pub fn decrypt_query(&self, encrypted: &Array1<f64>) -> VPKResult<Array1<f64>> {
        if encrypted.len() != self.output_dim {
            return Err(VPKError::InvalidDimension {
                expected: self.output_dim,
                actual: encrypted.len(),
            });
        }

        let e = DVector::from_vec(encrypted.to_vec());
        let decrypted = self.matrix_b.transpose() * e;
        Ok(Array1::from_vec(decrypted.iter().copied().collect()))
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn space_overhead_percent(&self) -> f64 {
        ((self.output_dim - self.input_dim) as f64 / self.input_dim as f64) * 100.0
    }

    /// Verify that A^T B ≈ I (within numerical precision).
    pub fn verify_key_property(&self) -> bool {
        let product = self.matrix_a.transpose() * &self.matrix_b;
        let identity: DMatrix<f64> = DMatrix::identity(self.input_dim, self.input_dim);

        for i in 0..self.input_dim {
            for j in 0..self.input_dim {
                let diff: f64 = product[(i, j)] - identity[(i, j)];
                if diff.abs() > 1e-9 {
                    return false;
                }
            }
        }
        true
    }
}

impl Zeroize for Diehard {
    fn zeroize(&mut self) {
        for elem in self.matrix_a.iter_mut() {
            elem.zeroize();
        }
        for elem in self.matrix_b.iter_mut() {
            elem.zeroize();
        }
    }
}

impl Drop for Diehard {
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
    fn test_diehard_creation() {
        let dh = Diehard::new(50, 75).unwrap();
        assert_eq!(dh.input_dim(), 50);
        assert_eq!(dh.output_dim(), 75);
        assert_abs_diff_eq!(dh.space_overhead_percent(), 50.0, epsilon = 0.1);
    }

    #[test]
    fn test_diehard_default_creation() {
        let dh = Diehard::new_default(100).unwrap();
        assert_eq!(dh.input_dim(), 100);
        assert_eq!(dh.output_dim(), 150);
    }

    #[test]
    fn test_diehard_invalid_dimensions() {
        let result = Diehard::new(50, 40);
        assert!(result.is_err());

        // output_dim == input_dim should work (degenerate case)
        let result = Diehard::new(50, 50);
        assert!(result.is_ok());
    }

    #[test]
    fn test_key_property_at_b_equals_identity() {
        let dh = Diehard::new(50, 75).unwrap();
        assert!(dh.verify_key_property(), "A^T B should equal I");
    }

    #[test]
    fn test_inner_product_preservation() {
        let dh = Diehard::new(50, 75).unwrap();

        let v1 = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));
        let v2 = normalize(&Array1::from_vec((0..50).map(|i| (50 - i) as f64).collect()));

        let enc_query = dh.encrypt_query(&v1).unwrap();
        let enc_doc = dh.encrypt_document(&v2).unwrap();

        let plain_dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let enc_dot: f64 = enc_query.iter().zip(enc_doc.iter()).map(|(a, b)| a * b).sum();

        assert_abs_diff_eq!(plain_dot, enc_dot, epsilon = 1e-9);
    }

    #[test]
    fn test_asymmetric_encryption() {
        let dh = Diehard::new(50, 75).unwrap();

        let v = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));

        let enc_doc = dh.encrypt_document(&v).unwrap();
        let enc_query = dh.encrypt_query(&v).unwrap();

        // Document and query encryptions should differ (asymmetric keys)
        let diff: f64 = enc_doc
            .iter()
            .zip(enc_query.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "DIEHARD should produce different ciphertexts for docs vs queries");
    }

    #[test]
    fn test_cosine_similarity_preservation() {
        let dh = Diehard::new(100, 150).unwrap();

        let v1 = normalize(&Array1::from_vec((0..100).map(|i| (i + 1) as f64).collect()));
        let v2 = normalize(&Array1::from_vec((0..100).map(|i| (100 - i) as f64).collect()));

        let enc_query = dh.encrypt_query(&v1).unwrap();
        let enc_doc = dh.encrypt_document(&v2).unwrap();

        let plain_sim = cosine_similarity(&v1, &v2);

        // For DIEHARD, inner product is preserved but norms may differ,
        // so we check dot-product preservation directly.
        let enc_dot: f64 = enc_query.iter().zip(enc_doc.iter()).map(|(a, b)| a * b).sum();
        let plain_dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();

        assert_abs_diff_eq!(plain_dot, enc_dot, epsilon = 1e-9);

        // Since v1, v2 are unit vectors, plain_dot == plain_sim
        assert_abs_diff_eq!(plain_sim, enc_dot, epsilon = 1e-9);
    }

    #[test]
    fn test_ranking_preservation_100_percent() {
        let dh = Diehard::new(50, 75).unwrap();

        let query = normalize(&Array1::from_vec((0..50).map(|i| i as f64).collect()));

        let docs: Vec<Array1<f64>> = vec![
            normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect())),
            normalize(&Array1::from_vec((0..50).map(|i| (i + 10) as f64).collect())),
            normalize(&Array1::from_vec((0..50).map(|i| (49 - i) as f64).collect())),
            normalize(&Array1::from_vec((0..50).map(|i| (i * 2) as f64).collect())),
        ];

        let plain_dots: Vec<f64> = docs
            .iter()
            .map(|doc| query.iter().zip(doc.iter()).map(|(a, b)| a * b).sum())
            .collect();

        let enc_query = dh.encrypt_query(&query).unwrap();
        let enc_docs: Vec<Array1<f64>> = docs
            .iter()
            .map(|doc| dh.encrypt_document(doc).unwrap())
            .collect();

        let enc_dots: Vec<f64> = enc_docs
            .iter()
            .map(|doc| enc_query.iter().zip(doc.iter()).map(|(a, b)| a * b).sum())
            .collect();

        let mut plain_indices: Vec<usize> = (0..docs.len()).collect();
        plain_indices.sort_by(|&a, &b| plain_dots[b].partial_cmp(&plain_dots[a]).unwrap());

        let mut enc_indices: Vec<usize> = (0..docs.len()).collect();
        enc_indices.sort_by(|&a, &b| enc_dots[b].partial_cmp(&enc_dots[a]).unwrap());

        assert_eq!(plain_indices, enc_indices, "Rankings must be identical");
    }

    #[test]
    fn test_document_decrypt_roundtrip() {
        let dh = Diehard::new(50, 75).unwrap();
        let original = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));

        let encrypted = dh.encrypt_document(&original).unwrap();
        let decrypted = dh.decrypt_document(&encrypted).unwrap();

        for i in 0..50 {
            assert_abs_diff_eq!(original[i], decrypted[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_query_decrypt_roundtrip() {
        let dh = Diehard::new(50, 75).unwrap();
        let original = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));

        let encrypted = dh.encrypt_query(&original).unwrap();
        let decrypted = dh.decrypt_query(&encrypted).unwrap();

        for i in 0..50 {
            assert_abs_diff_eq!(original[i], decrypted[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_high_dimensional_384() {
        let dh = Diehard::new(384, 576).unwrap();
        assert!(dh.verify_key_property());

        let v1 = normalize(&Array1::from_vec((0..384).map(|i| (i + 1) as f64).collect()));
        let v2 = normalize(&Array1::from_vec((0..384).map(|i| (384 - i) as f64).collect()));

        let enc_q = dh.encrypt_query(&v1).unwrap();
        let enc_d = dh.encrypt_document(&v2).unwrap();

        let plain_dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let enc_dot: f64 = enc_q.iter().zip(enc_d.iter()).map(|(a, b)| a * b).sum();

        assert_abs_diff_eq!(plain_dot, enc_dot, epsilon = 1e-8);
    }

    #[test]
    fn test_serialization() {
        let dh = Diehard::new(50, 75).unwrap();

        let serialized = bincode::serialize(&dh).unwrap();
        let restored: Diehard = bincode::deserialize(&serialized).unwrap();

        assert_eq!(dh.input_dim(), restored.input_dim());
        assert_eq!(dh.output_dim(), restored.output_dim());

        let v = normalize(&Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect()));

        let enc1 = dh.encrypt_document(&v).unwrap();
        let enc2 = restored.encrypt_document(&v).unwrap();

        for i in 0..enc1.len() {
            assert_abs_diff_eq!(enc1[i], enc2[i], epsilon = 1e-12);
        }
    }
}
