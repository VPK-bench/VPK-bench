//! Real CKKS Encryption Layer (Microsoft SEAL via sealy)
//!
//! Implements the CKKS (Cheon-Kim-Kim-Song) fully homomorphic encryption scheme
//! using Microsoft SEAL through the `sealy` Rust bindings.
//!
//! Unlike the other encryption layers (DS, ROME) which produce f64 vectors
//! suitable for cosine-similarity search in a vector DB, CKKS produces
//! opaque ciphertexts. Similarity must be computed homomorphically:
//!
//!   score = Dec( ct_query ⊙ ct_doc )   (element-wise multiply + sum slots)
//!
//! This means the CKKS path bypasses the regular vector-DB search and instead
//! uses `CkksVectorStorage` which stores serialized ciphertexts and performs
//! homomorphic dot-product search.
//!
//! # SEAL Parameters
//!
//! | Parameter              | Value               |
//! |------------------------|---------------------|
//! | Scheme                 | CKKS                |
//! | Poly modulus degree    | 8192                |
//! | Coeff modulus bits     | [60, 40, 40, 60]    |
//! | Scale                  | 2^40                |
//! | Security level         | TC128               |
//! | Slot count             | 4096 (N/2)          |

use crate::error::{VPKError, VPKResult};
use ndarray::Array1;
use sealy::{
    AsymmetricEncryptor, CKKSEncoder, CKKSEncryptionParametersBuilder, CKKSEvaluator,
    CoefficientModulusFactory, Context, Ciphertext as SealCiphertext, Decryptor, DegreeType, Evaluator,
    FromBytes, GaloisKey, KeyGenerator, RelinearizationKey, SecurityLevel, ToBytes,
};
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::Arc;

extern "C" {
    fn KeyGenerator_CreateGaloisKeysFromSteps(
        thisptr: *mut c_void,
        count: u64,
        steps: *const i32,
        save_seed: bool,
        galois_keys: *mut *mut c_void,
    ) -> std::os::raw::c_long;

    fn Evaluator_RotateVector(
        thisptr: *mut c_void,
        encrypted: *mut c_void,
        steps: i32,
        galois_keys: *mut c_void,
        destination: *mut c_void,
        pool: *mut c_void,
    ) -> std::os::raw::c_long;

    fn Evaluator_RescaleToNext(
        thisptr: *mut c_void,
        encrypted: *mut c_void,
        destination: *mut c_void,
        pool: *mut c_void,
    ) -> std::os::raw::c_long;
}

/// Extract the raw SEAL handle from any sealy wrapper struct whose layout is
/// `struct Foo { handle: AtomicPtr<c_void> }` (or a newtype of such).
/// All sealy types (KeyGenerator, EvaluatorBase, CKKSEvaluator, Ciphertext,
/// GaloisKey) follow this pattern.
unsafe fn raw_handle<T>(obj: &T) -> *mut c_void {
    let ptr = obj as *const T as *const *mut c_void;
    *ptr
}

/// Generate galois keys for only the specified rotation steps (powers of 2).
/// Much cheaper than `create_galois_keys()` which generates keys for ALL rotations.
fn create_galois_keys_from_steps(keygen: &KeyGenerator, steps: &[i32]) -> Result<GaloisKey, String> {
    let mut handle: *mut c_void = null_mut();
    let hr = unsafe {
        KeyGenerator_CreateGaloisKeysFromSteps(
            raw_handle(keygen),
            steps.len() as u64,
            steps.as_ptr(),
            false,
            &mut handle,
        )
    };
    if hr != 0 {
        return Err(format!("SEAL CreateGaloisKeysFromSteps failed: 0x{:08x}", hr));
    }
    Ok(unsafe { std::mem::transmute(handle) })
}

/// CKKS rotate_vector via direct FFI (sealy only exposes BFV rotate_rows).
fn ckks_rotate_vector(
    evaluator: &CKKSEvaluator,
    ct: &SealCiphertext,
    steps: i32,
    galois_keys: &GaloisKey,
) -> Result<SealCiphertext, String> {
    let out = SealCiphertext::new()
        .map_err(|e| format!("Ciphertext alloc: {}", e))?;
    let hr = unsafe {
        Evaluator_RotateVector(
            raw_handle(evaluator),
            raw_handle(ct),
            steps,
            galois_keys.get_handle(),
            raw_handle(&out),
            null_mut(),
        )
    };
    if hr != 0 {
        return Err(format!("SEAL RotateVector(step={}) failed: 0x{:08x}", steps, hr));
    }
    Ok(out)
}

/// CKKS rescale_to_next via direct FFI.
fn ckks_rescale_to_next(
    evaluator: &CKKSEvaluator,
    ct: &SealCiphertext,
) -> Result<SealCiphertext, String> {
    let out = SealCiphertext::new()
        .map_err(|e| format!("Ciphertext alloc: {}", e))?;
    let hr = unsafe {
        Evaluator_RescaleToNext(
            raw_handle(evaluator),
            raw_handle(ct),
            raw_handle(&out),
            null_mut(),
        )
    };
    if hr != 0 {
        return Err(format!("SEAL RescaleToNext failed: 0x{:08x}", hr));
    }
    Ok(out)
}

/// CKKS encryption context holding all SEAL objects needed for
/// encode / encrypt / evaluate / decrypt.
///
/// This struct is intentionally NOT `Serialize`/`Deserialize` because the
/// SEAL objects are opaque C++ pointers. Key material is persisted separately.
pub struct Ckks {
    input_dim: usize,
    ctx: Arc<Context>,
    encoder: CKKSEncoder,
    encryptor: AsymmetricEncryptor,
    decryptor: Decryptor,
    evaluator: CKKSEvaluator,
    relin_keys: RelinearizationKey,
    galois_keys: GaloisKey,
    scale: f64,
    slot_count: usize,
    /// Pre-encrypted query ciphertext cache. VPK core encrypts the query in
    /// step 2 (timed as `encrypt_us`) and stashes the bytes here so that
    /// `CkksVectorStorage::search` can skip the redundant encrypt.
    query_ct_cache: std::sync::Mutex<Option<Vec<u8>>>,
}

// SEAL types have unsafe Send/Sync impls; verify Ckks can be shared across tasks.
unsafe impl Send for Ckks {}
unsafe impl Sync for Ckks {}

impl std::fmt::Debug for Ckks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ckks")
            .field("input_dim", &self.input_dim)
            .field("slot_count", &self.slot_count)
            .field("scale", &self.scale)
            .finish()
    }
}

impl Ckks {
    /// Create a new CKKS encryption instance.
    ///
    /// `input_dim` is the embedding vector length (e.g. 384).
    /// `_padded_dim` is accepted for API compatibility but unused —
    /// CKKS slot count is determined by the poly modulus degree.
    pub fn new(input_dim: usize, _padded_dim: usize) -> VPKResult<Self> {
        let scale: f64 = 2.0_f64.powi(40);

        let params = CKKSEncryptionParametersBuilder::new()
            .set_poly_modulus_degree(DegreeType::D8192)
            .set_coefficient_modulus(
                CoefficientModulusFactory::build(DegreeType::D8192, &[60, 40, 40, 60])
                    .map_err(|e| VPKError::KeyGenerationError(format!("SEAL coeff mod: {}", e)))?,
            )
            .build()
            .map_err(|e| VPKError::KeyGenerationError(format!("SEAL params: {}", e)))?;

        let ctx = Context::new(&params, false, SecurityLevel::TC128)
            .map_err(|e| VPKError::KeyGenerationError(format!("SEAL context: {}", e)))?;
        let ctx = Arc::new(ctx);

        let keygen = KeyGenerator::new(&ctx)
            .map_err(|e| VPKError::KeyGenerationError(format!("SEAL keygen: {}", e)))?;

        let public_key = keygen.create_public_key();
        let secret_key = keygen.secret_key();
        let relin_keys = keygen
            .create_relinearization_keys()
            .map_err(|e| VPKError::KeyGenerationError(format!("SEAL relin keys: {}", e)))?;

        // Only generate galois keys for the power-of-2 rotation steps we actually
        // use in dot_product_decrypt (1, 2, 4, ..., slot_count/2). This is ~12 keys
        // for D8192 instead of ~8191 from create_galois_keys(), avoiding OOM.
        let temp_encoder = CKKSEncoder::new(&ctx, scale)
            .map_err(|e| VPKError::KeyGenerationError(format!("SEAL temp encoder: {}", e)))?;
        let sc = temp_encoder.get_slot_count();
        let mut steps_needed: Vec<i32> = Vec::new();
        let mut s = 1i32;
        while (s as usize) < sc {
            steps_needed.push(s);
            s *= 2;
        }
        let galois_keys = create_galois_keys_from_steps(&keygen, &steps_needed)
            .map_err(|e| VPKError::KeyGenerationError(format!("SEAL galois keys: {}", e)))?;

        let encoder = CKKSEncoder::new(&ctx, scale)
            .map_err(|e| VPKError::KeyGenerationError(format!("SEAL encoder: {}", e)))?;

        let slot_count = encoder.get_slot_count();

        if input_dim > slot_count {
            return Err(VPKError::InvalidDimension {
                expected: slot_count,
                actual: input_dim,
            });
        }

        let encryptor = AsymmetricEncryptor::new(&ctx, &public_key)
            .map_err(|e| VPKError::KeyGenerationError(format!("SEAL encryptor: {}", e)))?;

        let decryptor = Decryptor::new(&ctx, &secret_key)
            .map_err(|e| VPKError::KeyGenerationError(format!("SEAL decryptor: {}", e)))?;

        let evaluator = CKKSEvaluator::new(&ctx)
            .map_err(|e| VPKError::KeyGenerationError(format!("SEAL evaluator: {}", e)))?;

        Ok(Self {
            input_dim,
            ctx,
            encoder,
            encryptor,
            decryptor,
            evaluator,
            relin_keys,
            galois_keys,
            scale,
            slot_count,
            query_ct_cache: std::sync::Mutex::new(None),
        })
    }

    /// Store a pre-encrypted query ciphertext so that search shards can
    /// reuse it instead of encrypting again.
    pub fn cache_query_ct(&self, ct: Vec<u8>) {
        *self.query_ct_cache.lock().unwrap() = Some(ct);
    }

    /// Clone the cached query ciphertext (returns `None` if not set).
    pub fn cached_query_ct(&self) -> Option<Vec<u8>> {
        self.query_ct_cache.lock().unwrap().clone()
    }

    /// Clear the cached query ciphertext.
    pub fn clear_query_ct_cache(&self) {
        *self.query_ct_cache.lock().unwrap() = None;
    }

    /// Encode and encrypt a vector, returning the serialized ciphertext bytes.
    pub fn encrypt(&self, vector: &Array1<f64>) -> VPKResult<Vec<u8>> {
        if vector.len() != self.input_dim {
            return Err(VPKError::InvalidDimension {
                expected: self.input_dim,
                actual: vector.len(),
            });
        }

        let data: Vec<f64> = vector.to_vec();
        let plaintext = self
            .encoder
            .encode_f64(&data)
            .map_err(|e| VPKError::EncryptionError(format!("CKKS encode: {}", e)))?;

        let ciphertext = self
            .encryptor
            .encrypt(&plaintext)
            .map_err(|e| VPKError::EncryptionError(format!("CKKS encrypt: {}", e)))?;

        ciphertext
            .as_bytes()
            .map_err(|e| VPKError::SerializationError(format!("CKKS serialize ct: {}", e)))
    }

    /// Deserialize a ciphertext from bytes.
    pub fn deserialize_ciphertext(&self, bytes: &[u8]) -> VPKResult<SealCiphertext> {
        SealCiphertext::from_bytes(&self.ctx, bytes)
            .map_err(|e| VPKError::SerializationError(format!("CKKS deserialize ct: {}", e)))
    }

    /// Compute the dot product of two ciphertexts homomorphically and
    /// decrypt the scalar result.
    ///
    /// Algorithm:
    ///   1. Element-wise multiply: ct_prod = ct_a ⊙ ct_b
    ///   2. Relinearize to keep ciphertext size at 2
    ///   3. Decrypt the element-wise products (scale = 2^80)
    ///   4. Sum the first input_dim decoded slots in plaintext domain
    ///
    /// Summing in plaintext avoids operating at scale 2^80 through log(N)
    /// homomorphic rotations, which causes precision loss that flips rankings
    /// for closely-scored documents.  The multiply is still computed over
    /// ciphertexts; only the final reduction is done after decryption.
    pub fn dot_product_decrypt(
        &self,
        ct_a_bytes: &[u8],
        ct_b_bytes: &[u8],
    ) -> VPKResult<f64> {
        let ct_a = self.deserialize_ciphertext(ct_a_bytes)?;
        let ct_b = self.deserialize_ciphertext(ct_b_bytes)?;

        let ct_prod = self
            .evaluator
            .multiply(&ct_a, &ct_b)
            .map_err(|e| VPKError::EncryptionError(format!("CKKS multiply: {}", e)))?;

        let ct_relin = self
            .evaluator
            .relinearize(&ct_prod, &self.relin_keys)
            .map_err(|e| VPKError::EncryptionError(format!("CKKS relinearize: {}", e)))?;

        let decrypted_pt = self
            .decryptor
            .decrypt(&ct_relin)
            .map_err(|e| VPKError::DecryptionError(format!("CKKS decrypt: {}", e)))?;

        let decoded = self
            .encoder
            .decode_f64(&decrypted_pt)
            .map_err(|e| VPKError::DecryptionError(format!("CKKS decode: {}", e)))?;

        let dot: f64 = decoded[..self.input_dim].iter().sum();
        Ok(dot)
    }

    /// Decrypt a ciphertext back to a vector (for testing / debugging).
    pub fn decrypt(&self, ct_bytes: &[u8]) -> VPKResult<Array1<f64>> {
        let ct = self.deserialize_ciphertext(ct_bytes)?;

        let pt = self
            .decryptor
            .decrypt(&ct)
            .map_err(|e| VPKError::DecryptionError(format!("CKKS decrypt: {}", e)))?;

        let decoded = self
            .encoder
            .decode_f64(&pt)
            .map_err(|e| VPKError::DecryptionError(format!("CKKS decode: {}", e)))?;

        Ok(Array1::from_vec(decoded[..self.input_dim].to_vec()))
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn slot_count(&self) -> usize {
        self.slot_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize(v: &Array1<f64>) -> Array1<f64> {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            v.clone()
        } else {
            v / norm
        }
    }

    #[test]
    fn test_ckks_encrypt_decrypt_roundtrip() {
        let dim = 64;
        let ckks = Ckks::new(dim, dim + 32).unwrap();
        let v = normalize(&Array1::from_vec((0..dim).map(|i| (i + 1) as f64).collect()));

        let ct_bytes = ckks.encrypt(&v).unwrap();
        let decrypted = ckks.decrypt(&ct_bytes).unwrap();

        for i in 0..dim {
            assert!(
                (v[i] - decrypted[i]).abs() < 1e-3,
                "Mismatch at dim {}: {} vs {}",
                i,
                v[i],
                decrypted[i]
            );
        }
    }

    #[test]
    fn test_ckks_dot_product() {
        let dim = 64;
        let ckks = Ckks::new(dim, dim + 32).unwrap();

        let v1 = normalize(&Array1::from_vec((0..dim).map(|i| (i + 1) as f64).collect()));
        let v2 = normalize(&Array1::from_vec((0..dim).map(|i| i as f64).collect()));

        let plain_dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();

        let ct1 = ckks.encrypt(&v1).unwrap();
        let ct2 = ckks.encrypt(&v2).unwrap();
        let enc_dot = ckks.dot_product_decrypt(&ct1, &ct2).unwrap();

        println!("Plain dot product:     {:.8}", plain_dot);
        println!("Encrypted dot product: {:.8}", enc_dot);
        println!("Difference:            {:.8}", (plain_dot - enc_dot).abs());

        assert!(
            (plain_dot - enc_dot).abs() < 0.1,
            "Dot product not preserved: plain={} enc={}",
            plain_dot,
            enc_dot
        );
    }

    #[test]
    fn test_ckks_ranking_preservation() {
        let dim = 64;
        let ckks = Ckks::new(dim, dim + 32).unwrap();

        let query = normalize(&Array1::from_vec((0..dim).map(|i| i as f64).collect()));

        let docs: Vec<Array1<f64>> = (0..5)
            .map(|k| {
                normalize(&Array1::from_vec(
                    (0..dim)
                        .map(|i| (i as f64) + (k as f64) * 5.0 * ((i * k) as f64).sin())
                        .collect(),
                ))
            })
            .collect();

        let ct_query = ckks.encrypt(&query).unwrap();
        let ct_docs: Vec<Vec<u8>> = docs.iter().map(|d| ckks.encrypt(d).unwrap()).collect();

        let mut plain_scores: Vec<(usize, f64)> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let dot: f64 = query.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
                (i, dot)
            })
            .collect();
        plain_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut enc_scores: Vec<(usize, f64)> = ct_docs
            .iter()
            .enumerate()
            .map(|(i, ct_d)| {
                let score = ckks.dot_product_decrypt(&ct_query, ct_d).unwrap();
                (i, score)
            })
            .collect();
        enc_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let plain_top: Vec<usize> = plain_scores.iter().take(3).map(|s| s.0).collect();
        let enc_top: Vec<usize> = enc_scores.iter().take(3).map(|s| s.0).collect();

        println!("Plain top-3: {:?}", plain_top);
        println!("Enc   top-3: {:?}", enc_top);

        assert_eq!(plain_top[0], enc_top[0], "Top-1 ranking must be preserved");
    }
}
