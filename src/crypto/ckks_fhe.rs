//! True CKKS Fully Homomorphic Encryption
//!
//! Implements the Cheon-Kim-Kim-Song (2016) scheme using Ring Learning With
//! Errors (RLWE) as the underlying hardness assumption.
//!
//! # Parameters (demo)
//!
//! | Parameter | Value | Notes |
//! |-----------|-------|-------|
//! | N (ring dim) | 1024 | Demo: ~80-bit security.  Production: ≥ 4096. |
//! | Q (modulus)  | 998_244_353 | NTT prime: 119 × 2^23 + 1 |
//! | Δ (scale)    | 2^20 | Fixed-point precision for real inputs |
//! | σ (noise)    | 3.2 | Gaussian std-dev (NIST-recommended) |
//!
//! # Why CkksFhe breaks ANN similarity search
//!
//! RLWE ciphertexts are computationally indistinguishable from uniformly random
//! elements of Z_q^N (under the RLWE hardness assumption).  Consequently:
//!
//! - The Euclidean / cosine distance between two ciphertexts carries **no
//!   information** about the distance between their plaintexts.
//! - ANN search on stored ciphertexts returns semantically random results
//!   (Recall@K ≈ 1/corpus_size).
//!
//! **This is correct and intended behaviour**: a scheme that preserved
//! similarity in ciphertext space would be insecure.  The comparison with the
//! other schemes (ROME, CkksInspired) in experiments demonstrates that
//! similarity-preserving obfuscation and RLWE security are fundamentally
//! incompatible.
//!
//! For a similarity-search pipeline with true RLWE security, use homomorphic
//! inner products at the server side (`CkksFhe::homomorphic_dot`).

use crate::crypto::ntt::{self, N, Q};
use crate::error::{VPKError, VPKResult};
use ndarray::Array1;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};

/// Scaling factor Δ: real values are multiplied by Δ before rounding.
pub const DELTA: f64 = (1u64 << 20) as f64; // 2^20 ≈ 1 million

/// Gaussian noise standard deviation.
const SIGMA: f64 = 3.2;

/// Ciphertext vector dimension stored in the ANN index: 2 × N.
/// (c0 and c1 each contribute N f64 values scaled to (−1, 1].)
pub const CIPHERTEXT_DIM: usize = 2 * N;

// ── Types ─────────────────────────────────────────────────────────────────────

/// Polynomial in Z_Q[X]/(X^N + 1), stored as N i64 coefficients.
type Poly = Vec<i64>;

// ── Random sampling ───────────────────────────────────────────────────────────

/// Sample a ternary polynomial: coefficients uniform in {−1, 0, 1}.
fn sample_ternary(rng: &mut impl rand::Rng) -> Poly {
    let dist = Uniform::new(0u8, 3);
    (0..N).map(|_| dist.sample(rng) as i64 - 1).collect()
}

/// Sample a uniform polynomial in [0, Q).
fn sample_uniform(rng: &mut impl rand::Rng) -> Poly {
    let dist = Uniform::new(0i64, Q);
    (0..N).map(|_| dist.sample(rng)).collect()
}

/// Sample a discrete Gaussian polynomial (rounded continuous Gaussian).
fn sample_gaussian(rng: &mut impl rand::Rng) -> Poly {
    let dist = Normal::new(0.0f64, SIGMA).expect("valid sigma");
    (0..N)
        .map(|_| ntt::mod_q(dist.sample(rng).round() as i64))
        .collect()
}

// ── RLWE key generation ───────────────────────────────────────────────────────

/// Generate RLWE encryption of `pt` under secret key `sk`.
fn gen_rlwe_encryption(pt: &Poly, sk: &Poly, rng: &mut impl rand::Rng) -> (Poly, Poly) {
    let a = sample_uniform(rng);
    let e = sample_gaussian(rng);
    let a_s = ntt::poly_mul(&a, sk);
    // b = pt - a·s - e  (all mod Q)
    let b = ntt::poly_sub(
        ntt::poly_sub(pt, &a_s).as_slice(),
        &e,
    );
    (b, a)
}

// ── CkksFhe struct ────────────────────────────────────────────────────────────

/// True CKKS FHE encryptor using RLWE.
///
/// Stores the full key material (secret + public + evaluation keys).
/// The secret key is included so that Bob can decrypt on his own side.
///
/// # Security note
///
/// The secret key is held in plaintext in this struct (as in all the other
/// algorithms in this codebase).  In production, the secret key would remain
/// on Bob's trusted hardware; only the public/evaluation keys go to Eve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CkksFhe {
    /// Original embedding dimension.
    input_dim: usize,
    /// Secret key s ∈ R₂ (ternary coefficients).
    sk: Poly,
    /// Public key component b (b = −(a·s + e)).
    pk_b: Poly,
    /// Public key component a (uniform random).
    pk_a: Poly,
    /// Evaluation key component b̂ (for relinearization).
    ek_b: Poly,
    /// Evaluation key component â (for relinearization).
    ek_a: Poly,
}

impl CkksFhe {
    /// Generate a fresh CkksFhe instance with new RLWE keys.
    pub fn new(input_dim: usize) -> VPKResult<Self> {
        if input_dim > N {
            return Err(VPKError::InvalidDimension {
                expected: N,
                actual: input_dim,
            });
        }
        let mut rng = rand::thread_rng();
        let (sk, pk_b, pk_a, ek_b, ek_a) = Self::generate_keys(&mut rng);
        Ok(Self { input_dim, sk, pk_b, pk_a, ek_b, ek_a })
    }

    fn generate_keys(rng: &mut impl rand::Rng) -> (Poly, Poly, Poly, Poly, Poly) {
        let sk: Poly = sample_ternary(rng);
        let (pk_b, pk_a) = gen_rlwe_encryption(&vec![0i64; N], &sk, rng);

        // Evaluation key: RLWE encryption of s² (used for relinearization).
        let s2 = ntt::poly_mul(&sk, &sk);
        let (ek_b, ek_a) = gen_rlwe_encryption(&s2, &sk, rng);

        (sk, pk_b, pk_a, ek_b, ek_a)
    }

    // ── Encoding / decoding ────────────────────────────────────────────────

    /// Coefficient encoding: v[i] → round(Δ · v[i]), zero-padded to length N.
    ///
    /// This is a simplified (non-slot) CKKS encoding that maps the input
    /// vector directly onto polynomial coefficients.  It is sufficient for
    /// encrypt/decrypt roundtrips; the canonical (DFT-based) slot encoding
    /// would additionally be needed for SIMD homomorphic inner products.
    pub fn encode(v: &[f64]) -> Poly {
        (0..N)
            .map(|i| {
                if i < v.len() {
                    ntt::mod_q((DELTA * v[i]).round() as i64)
                } else {
                    0
                }
            })
            .collect()
    }

    /// Decode coefficient polynomial back to a real vector of length `dim`.
    pub fn decode(poly: &Poly, dim: usize) -> Vec<f64> {
        (0..dim)
            .map(|i| ntt::center(poly[i]) as f64 / DELTA)
            .collect()
    }

    // ── RLWE encrypt / decrypt ─────────────────────────────────────────────

    /// Encrypt a plaintext polynomial under the public key.
    ///
    /// ct = (c₀, c₁) = (pt + e₀ + u·b, u·a + e₁)
    /// where u ← R₂, e₀,e₁ ← DG(σ).
    ///
    /// Correctness: c₀ + c₁·s = pt + e₀ + u·b + (u·a + e₁)·s
    ///            = pt + e₀ + u·(b + a·s) + e₁·s
    ///            = pt + e₀ − u·e_pk + e₁·s   ≈ pt  (errors small vs Δ)
    fn encrypt_poly(&self, pt: &Poly, rng: &mut impl rand::Rng) -> (Poly, Poly) {
        let u = sample_ternary(rng);
        let e0 = sample_gaussian(rng);
        let e1 = sample_gaussian(rng);
        let ub = ntt::poly_mul(&u, &self.pk_b);
        let ua = ntt::poly_mul(&u, &self.pk_a);
        // c0 = pt + e0 + u·b
        let c0 = ntt::poly_add(&ntt::poly_add(pt, &e0), &ub);
        // c1 = e1 + u·a
        let c1 = ntt::poly_add(&e1, &ua);
        (c0, c1)
    }

    /// Decrypt a ciphertext (c₀, c₁) using the secret key.
    ///
    /// Recovers m̃ = c₀ + c₁·s mod (X^N+1, Q), then centers coefficients.
    pub fn decrypt_poly(&self, c0: &Poly, c1: &Poly) -> Poly {
        let c1s = ntt::poly_mul(c1, &self.sk);
        let raw = ntt::poly_add(c0, &c1s);
        ntt::poly_center(&raw)
    }

    // ── Homomorphic addition ───────────────────────────────────────────────

    /// Component-wise ciphertext addition.
    pub fn hadd(
        &self,
        c0a: &Poly, c1a: &Poly,
        c0b: &Poly, c1b: &Poly,
    ) -> (Poly, Poly) {
        (ntt::poly_add(c0a, c0b), ntt::poly_add(c1a, c1b))
    }

    // ── Homomorphic multiplication + relinearization ───────────────────────

    /// Homomorphic multiplication of two ciphertexts with relinearization.
    ///
    /// Result encodes m₁·m₂ (with approximation error from rounding).
    ///
    /// # Algorithm
    ///
    /// Extended ciphertext after naive multiplication:
    ///   (d₀, d₁, d₂) = (c₀ᵃ·c₀ᵇ, c₀ᵃ·c₁ᵇ + c₁ᵃ·c₀ᵇ, c₁ᵃ·c₁ᵇ)
    ///
    /// Relinearization collapses d₂ using the evaluation key:
    ///   c₀' = d₀ + ek_b · d₂ / Q (rounded)
    ///   c₁' = d₁ + ek_a · d₂ / Q (rounded)
    pub fn hmul(
        &self,
        c0a: &Poly, c1a: &Poly,
        c0b: &Poly, c1b: &Poly,
    ) -> (Poly, Poly) {
        let d0 = ntt::poly_mul(c0a, c0b);
        let d1 = ntt::poly_add(
            &ntt::poly_mul(c0a, c1b),
            &ntt::poly_mul(c1a, c0b),
        );
        let d2 = ntt::poly_mul(c1a, c1b);

        // Relinearization: fold d2 back using ek
        let relin_b = self.relin_component(&d2, &self.ek_b);
        let relin_a = self.relin_component(&d2, &self.ek_a);

        let c0_new = ntt::poly_add(&d0, &relin_b);
        let c1_new = ntt::poly_add(&d1, &relin_a);

        // Rescale by Δ to keep magnitude in check (drop one level)
        let c0_rescaled = self.rescale(&c0_new);
        let c1_rescaled = self.rescale(&c1_new);

        (c0_rescaled, c1_rescaled)
    }

    /// Multiply d2 by an eval-key component and round-divide by Q.
    fn relin_component(&self, d2: &Poly, ek: &Poly) -> Poly {
        let product = ntt::poly_mul(d2, ek);
        // Divide each coefficient by Q (integer division after centering)
        product.iter().map(|&x| {
            let centered = ntt::center(x);
            // Rounded division by Q
            let q = Q;
            let div = if centered >= 0 {
                (centered + q / 2) / q
            } else {
                (centered - q / 2) / q
            };
            ntt::mod_q(div)
        }).collect()
    }

    /// Rescale: divide coefficients by Δ (integer rounded), mod Q.
    fn rescale(&self, p: &Poly) -> Poly {
        let delta_i64 = DELTA as i64;
        p.iter().map(|&x| {
            let c = ntt::center(x);
            let divided = if c >= 0 {
                (c + delta_i64 / 2) / delta_i64
            } else {
                (c - delta_i64 / 2) / delta_i64
            };
            ntt::mod_q(divided)
        }).collect()
    }

    // ── Homomorphic dot product ────────────────────────────────────────────

    /// Compute the inner product of two encrypted vectors homomorphically.
    ///
    /// Uses coefficient encoding: ⟨enc(q), enc(d)⟩ is approximated by
    /// summing the first `input_dim` coefficients of enc(q) · enc(d).
    ///
    /// Note: With coefficient (non-slot) encoding, polynomial multiplication
    /// gives a convolution, not element-wise product.  This function extracts
    /// the correct inner product from the convolution result (coefficient 0
    /// of the product polynomial in Z[X] equals ∑ᵢ a[i]·b[N-i], so we use a
    /// trick with a reversed copy).
    pub fn homomorphic_dot(
        &self,
        c0q: &Poly, c1q: &Poly,
        c0d: &Poly, c1d: &Poly,
    ) -> f64 {
        // Reverse d to turn convolution into correlation
        let mut d_rev_c0 = c0d.clone();
        let mut d_rev_c1 = c1d.clone();
        d_rev_c0.rotate_left(1);
        d_rev_c0.reverse();
        d_rev_c1.rotate_left(1);
        d_rev_c1.reverse();
        // Negate (negacyclic: reversed copy picks up sign from wrap)
        let d_neg_c0: Poly = d_rev_c0.iter().map(|&x| ntt::mod_q(-x)).collect();
        let d_neg_c1: Poly = d_rev_c1.iter().map(|&x| ntt::mod_q(-x)).collect();

        let (prod_c0, prod_c1) = self.hmul(c0q, c1q, &d_neg_c0, &d_neg_c1);
        let result_poly = self.decrypt_poly(&prod_c0, &prod_c1);
        // Coefficient 0 of the product = inner product of the two vectors
        ntt::center(result_poly[0]) as f64 / (DELTA * DELTA / DELTA)
    }

    // ── Pipeline interface ─────────────────────────────────────────────────

    /// Encrypt an embedding vector.  Returns a 2N-dimensional f64 vector
    /// representing the RLWE ciphertext (c₀ ‖ c₁), with each coefficient
    /// scaled by 1/Q into (−1, 1] for storage in a float32 vector index.
    pub fn encrypt_vector(&self, v: &Array1<f64>) -> VPKResult<Array1<f64>> {
        let slice = v.as_slice().ok_or_else(|| {
            VPKError::EncryptionError("Vector must be contiguous".to_string())
        })?;
        let pt = Self::encode(slice);
        let mut rng = rand::thread_rng();
        let (c0, c1) = self.encrypt_poly(&pt, &mut rng);

        let mut result = Vec::with_capacity(CIPHERTEXT_DIM);
        // Scale to (−0.5, 0.5] so f32 storage doesn't lose precision
        for &coef in &c0 {
            result.push(ntt::center(coef) as f64 / Q as f64);
        }
        for &coef in &c1 {
            result.push(ntt::center(coef) as f64 / Q as f64);
        }
        Ok(Array1::from_vec(result))
    }

    /// Decrypt a stored ciphertext vector back to the original embedding.
    pub fn decrypt_vector(&self, encrypted: &Array1<f64>) -> VPKResult<Array1<f64>> {
        if encrypted.len() != CIPHERTEXT_DIM {
            return Err(VPKError::InvalidDimension {
                expected: CIPHERTEXT_DIM,
                actual: encrypted.len(),
            });
        }
        let q_f = Q as f64;
        let c0: Poly = (0..N)
            .map(|i| ntt::mod_q((encrypted[i] * q_f).round() as i64))
            .collect();
        let c1: Poly = (0..N)
            .map(|i| ntt::mod_q((encrypted[N + i] * q_f).round() as i64))
            .collect();
        let pt = self.decrypt_poly(&c0, &c1);
        let v = Self::decode(&pt, self.input_dim);
        Ok(Array1::from_vec(v))
    }

    /// Output dimension of the encrypted vector (= CIPHERTEXT_DIM = 2N).
    pub fn ciphertext_dim(&self) -> usize {
        CIPHERTEXT_DIM
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Serialise to bytes (bincode).
    pub fn to_bytes(&self) -> VPKResult<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| VPKError::SerializationError(e.to_string()))
    }

    /// Deserialise from bytes.
    pub fn from_bytes(bytes: &[u8]) -> VPKResult<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| VPKError::SerializationError(e.to_string()))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn unit_vec(dim: usize, seed: f64) -> Array1<f64> {
        let raw: Vec<f64> = (0..dim).map(|i| ((i as f64 + seed).sin())).collect();
        let norm: f64 = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        Array1::from_vec(raw.iter().map(|x| x / norm).collect())
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let v: Vec<f64> = (0..64).map(|i| i as f64 / 64.0 - 0.5).collect();
        let poly = CkksFhe::encode(&v);
        let recovered = CkksFhe::decode(&poly, 64);
        for i in 0..64 {
            assert_abs_diff_eq!(v[i], recovered[i], epsilon = 1.0 / DELTA);
        }
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let ckks = CkksFhe::new(64).unwrap();
        let v: Vec<f64> = (0..64).map(|i| i as f64 / 64.0 - 0.5).collect();
        let pt = CkksFhe::encode(&v);
        let mut rng = rand::thread_rng();
        let (c0, c1) = ckks.encrypt_poly(&pt, &mut rng);
        let recovered_poly = ckks.decrypt_poly(&c0, &c1);
        let recovered = CkksFhe::decode(&recovered_poly, 64);
        for i in 0..64 {
            // Decryption should recover within noise bound (≪ 1/Δ per coefficient)
            assert_abs_diff_eq!(v[i], recovered[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_vector_encrypt_decrypt_pipeline() {
        let dim = 64;
        let ckks = CkksFhe::new(dim).unwrap();
        let v = unit_vec(dim, 0.0);
        let encrypted = ckks.encrypt_vector(&v).unwrap();
        assert_eq!(encrypted.len(), CIPHERTEXT_DIM);
        let decrypted = ckks.decrypt_vector(&encrypted).unwrap();
        for i in 0..dim {
            assert_abs_diff_eq!(v[i], decrypted[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_ciphertext_is_random_looking() {
        // Two encryptions of the same plaintext should produce different ciphertexts
        let ckks = CkksFhe::new(32).unwrap();
        let v = unit_vec(32, 1.0);
        let ct1 = ckks.encrypt_vector(&v).unwrap();
        let ct2 = ckks.encrypt_vector(&v).unwrap();
        // They should differ (randomised encryption)
        let diff: f64 = ct1.iter().zip(ct2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01, "Two encryptions of the same value must differ");
    }

    #[test]
    fn test_serialization() {
        let ckks = CkksFhe::new(32).unwrap();
        let bytes = ckks.to_bytes().unwrap();
        let restored = CkksFhe::from_bytes(&bytes).unwrap();
        // Encryption with original and restored keys must produce decryptable ciphertext
        let v: Vec<f64> = (0..32).map(|i| i as f64 / 32.0).collect();
        let pt = CkksFhe::encode(&v);
        let mut rng = rand::thread_rng();
        let (c0, c1) = ckks.encrypt_poly(&pt, &mut rng);
        // Decrypt with restored key
        let recovered_poly = restored.decrypt_poly(&c0, &c1);
        let recovered = CkksFhe::decode(&recovered_poly, 32);
        for i in 0..32 {
            assert_abs_diff_eq!(v[i], recovered[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_hadd() {
        let ckks = CkksFhe::new(32).unwrap();
        let v1: Vec<f64> = (0..32).map(|i| i as f64 / 64.0).collect();
        let v2: Vec<f64> = (0..32).map(|i| (32 - i) as f64 / 64.0).collect();
        let pt1 = CkksFhe::encode(&v1);
        let pt2 = CkksFhe::encode(&v2);
        let mut rng = rand::thread_rng();
        let (c0a, c1a) = ckks.encrypt_poly(&pt1, &mut rng);
        let (c0b, c1b) = ckks.encrypt_poly(&pt2, &mut rng);
        let (c0s, c1s) = ckks.hadd(&c0a, &c1a, &c0b, &c1b);
        let result_poly = ckks.decrypt_poly(&c0s, &c1s);
        let result = CkksFhe::decode(&result_poly, 32);
        for i in 0..32 {
            assert_abs_diff_eq!(v1[i] + v2[i], result[i], epsilon = 1e-3);
        }
    }
}
