//! Cryptographic primitives for VPK
//!
//! This module contains:
//! - Dimensional Scrambling (permutation-based PHE with exact homomorphism)
//! - Noise Injection (signal processing-based obfuscation)
//! - ROME (Random Orthogonal Matrix Encryption with exact inner product preservation)
//! - DIEHARD (Dual Independent Encryption for Hardened Asymmetric Retrieval Defense)
//! - CKKS (real CKKS FHE via Microsoft SEAL / sealy)
//! - Encryption Pipeline (flexible composition of encryption layers)
//! - Key management

pub mod dimensional_scrambling;
pub mod noise_injection;
pub mod rome;
pub mod diehard;
pub mod ckks;
pub mod encryption_pipeline;
pub mod keys;

pub use dimensional_scrambling::DimensionalScrambling;
pub use noise_injection::NoiseInjection;
pub use rome::Rome;
pub use diehard::Diehard;
pub use ckks::Ckks;
pub use encryption_pipeline::{EncryptionPipeline, EncryptionLayer, ScramblingLayer, NoiseLayer, RomeLayer, DiehardLayer, CkksLayer};
pub use keys::{EncryptionKeys, KeyManager, KeyStatus};
