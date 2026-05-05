//! VPK (Virtual Private Knowledge) - Secure vector search with homomorphic encryption
//!
//! Privacy-preserving vector search: documents are embedded, encrypted with a
//! partial homomorphic scheme, and distributed across N untrusted Eve shards.
//! Queries fan out to all shards in parallel; results are merged at VPK and
//! real document IDs are returned to Bob's application.

// Public modules
pub mod api;
pub mod config;
pub mod error;
pub mod mcp;
pub mod vpk;

// Re-export main types
pub use config::VPKConfig;
pub use error::{VPKError, VPKResult};
pub use vpk::VPK;

// Internal modules (crypto public for testing)
pub mod crypto;
mod database;
mod embedding;
mod index;
mod shard;
mod utils;
mod vectordb;
