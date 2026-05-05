//! VPK configuration

use crate::error::{VPKError, VPKResult};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Eve shard endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardEndpoint {
    /// Shard identifier (must be unique)
    pub shard_id: i32,
    /// Human-readable name (e.g., "eve-1")
    pub name: String,
    /// Optional URL; None = in-memory MVP mode
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Tenant UUID returned by Eve after registration.
    /// Populated on first use so Bob re-uses the same tenant across restarts.
    #[serde(default)]
    pub tenant_id: Option<String>,
}

fn default_shards() -> Vec<ShardEndpoint> {
    vec![
        ShardEndpoint { shard_id: 1, name: "eve-1".to_string(), endpoint: None, tenant_id: None },
        ShardEndpoint { shard_id: 2, name: "eve-2".to_string(), endpoint: None, tenant_id: None },
        ShardEndpoint { shard_id: 3, name: "eve-3".to_string(), endpoint: None, tenant_id: None },
    ]
}

/// Main VPK configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPKConfig {
    /// Database configuration
    pub database: DatabaseConfig,

    /// Vector database configuration
    pub vectordb: VectorDBConfig,

    /// Cryptography configuration
    pub crypto: CryptoConfig,

    /// API server configuration (optional)
    pub api: Option<APIConfig>,

    /// Eve shard endpoints (horizontal partitioning)
    #[serde(default = "default_shards")]
    pub shards: Vec<ShardEndpoint>,

    /// Bob's peer ID (e.g. libp2p base58 string).
    /// Sent to Eve during tenant registration so Eve can associate the
    /// tenant with a specific Bob on the distributed peer network.
    #[serde(default)]
    pub peer_id: Option<String>,
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// PostgreSQL connection string
    pub connection_string: String,

    /// Maximum number of connections in the pool
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,

    /// Minimum number of connections in the pool
    #[serde(default = "default_min_connections")]
    pub min_connections: u32,
}

fn default_max_connections() -> u32 {
    10
}

fn default_min_connections() -> u32 {
    2
}

/// Vector database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDBConfig {
    /// Vector database endpoint (e.g., "http://localhost:6333" for Qdrant)
    pub endpoint: String,

    /// Vector database type (qdrant, milvus, weaviate)
    #[serde(default = "default_vectordb_type")]
    pub db_type: String,

    /// Collection/index name
    #[serde(default = "default_collection_name")]
    pub collection_name: String,

    /// Use TLS for connection
    #[serde(default = "default_use_tls")]
    pub use_tls: bool,
}

fn default_vectordb_type() -> String {
    "qdrant".to_string()
}

fn default_collection_name() -> String {
    "vpk_vectors".to_string()
}

fn default_use_tls() -> bool {
    true
}

/// Encryption algorithm selection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EncryptionAlgorithm {
    /// Dimensional scrambling only (permutation-based, 100% ranking, O(n))
    Scrambling,
    /// Noise injection only (signal processing, 90-99% ranking, O(n))
    Noise,
    /// Combined: Scrambling + Noise (defense-in-depth, 90-99% ranking, O(n))
    Combined,
    /// ROME: Random Orthogonal Matrix Encryption (100% ranking, IND-CPA, O(m²))
    Rome,
    /// ROME + Noise: Maximum security (95-99% ranking, IND-CPA + obfuscation, O(m²))
    RomeCombined,
    /// ROMM: Random Orthogonal Matrix Multiplication — ROME without zero-padding (100% ranking, O(n²))
    Romm,
    /// ROMM + Noise: ROMM with additional noise layer (95-99% ranking, O(n²))
    RommCombined,
    /// DIEHARD: Dual Independent Encryption (asymmetric A/B matrices, 100% ranking, O(mn))
    Diehard,
    /// DIEHARD + Noise: DIEHARD with additional noise layer (95-99% ranking)
    DiehardCombined,
    /// CKKS: real CKKS FHE via Microsoft SEAL (homomorphic dot-product search)
    Ckks,
    /// CKKS + Noise: real CKKS FHE with additional noise layer
    CkksCombined,
}

impl Default for EncryptionAlgorithm {
    fn default() -> Self {
        Self::Combined
    }
}

impl EncryptionAlgorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scrambling       => "scrambling",
            Self::Noise            => "noise",
            Self::Combined         => "combined",
            Self::Rome             => "rome",
            Self::RomeCombined     => "rome-combined",
            Self::Romm             => "romm",
            Self::RommCombined     => "romm-combined",
            Self::Diehard          => "diehard",
            Self::DiehardCombined  => "diehard-combined",
            Self::Ckks             => "ckks",
            Self::CkksCombined     => "ckks-combined",
        }
    }
}

/// Cryptography configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoConfig {
    /// Original vector dimension (e.g., 1536 for OpenAI embeddings)
    pub vector_dim: usize,

    /// Padding dimension to add
    pub padding_dim: usize,

    /// Encryption algorithm to use
    #[serde(default)]
    pub algorithm: EncryptionAlgorithm,

    /// Noise range (min, max) for noise-based algorithms
    #[serde(default = "default_noise_range")]
    pub noise_range: (f64, f64),

    /// Whether to auto-rotate keys
    #[serde(default = "default_auto_rotate")]
    pub auto_rotate_keys: bool,

    /// Key rotation interval in days
    #[serde(default = "default_rotation_interval")]
    pub rotation_interval_days: u64,
}

fn default_noise_range() -> (f64, f64) {
    (0.9, 1.1)
}

fn default_auto_rotate() -> bool {
    false
}

fn default_rotation_interval() -> u64 {
    90
}

/// API server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIConfig {
    /// Bind address
    #[serde(default = "default_bind_address")]
    pub bind_address: String,

    /// Port number
    #[serde(default = "default_port")]
    pub port: u16,

    /// Enable CORS
    #[serde(default = "default_enable_cors")]
    pub enable_cors: bool,

    /// Base URL the frontend uses to reach the API (empty = relative URLs)
    #[serde(default)]
    pub api_base: String,
}

fn default_bind_address() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_enable_cors() -> bool {
    false
}

impl VPKConfig {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> VPKResult<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| VPKError::ConfigError(format!("Failed to read config file: {}", e)))?;

        let config: VPKConfig = toml::from_str(&content)
            .map_err(|e| VPKError::ConfigError(format!("Failed to parse config: {}", e)))?;

        config.validate()?;
        Ok(config)
    }

    /// Create a default configuration for testing
    pub fn default_test() -> Self {
        Self {
            database: DatabaseConfig {
                connection_string: "postgresql://localhost/vpk_test".to_string(),
                max_connections: 5,
                min_connections: 1,
            },
            vectordb: VectorDBConfig {
                endpoint: "http://localhost:6333".to_string(),
                db_type: "qdrant".to_string(),
                collection_name: "vpk_test".to_string(),
                use_tls: false,
            },
            crypto: CryptoConfig {
                vector_dim: 384,  // all-MiniLM-L6-v2 dimension
                padding_dim: 128,  // Adjusted for smaller base dimension
                algorithm: EncryptionAlgorithm::Combined,
                noise_range: (0.9, 1.1),
                auto_rotate_keys: false,
                rotation_interval_days: 90,
            },
            api: Some(APIConfig {
                bind_address: "127.0.0.1".to_string(),
                port: 8080,
                enable_cors: false,
                api_base: String::new(),
            }),
            shards: default_shards(),
            peer_id: None,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> VPKResult<()> {
        // Validate vector dimensions
        if self.crypto.vector_dim == 0 {
            return Err(VPKError::ConfigError(
                "vector_dim must be greater than 0".to_string(),
            ));
        }

        if self.crypto.padding_dim == 0 {
            return Err(VPKError::ConfigError(
                "padding_dim must be greater than 0".to_string(),
            ));
        }

        // Validate database connection string
        if self.database.connection_string.is_empty() {
            return Err(VPKError::ConfigError(
                "database connection_string cannot be empty".to_string(),
            ));
        }

        // Validate vector DB endpoint
        if self.vectordb.endpoint.is_empty() {
            return Err(VPKError::ConfigError(
                "vectordb endpoint cannot be empty".to_string(),
            ));
        }

        // Validate connection pool settings
        if self.database.max_connections < self.database.min_connections {
            return Err(VPKError::ConfigError(
                "max_connections must be >= min_connections".to_string(),
            ));
        }

        Ok(())
    }

    /// Get the total padded dimension
    pub fn padded_dim(&self) -> usize {
        self.crypto.vector_dim + self.crypto.padding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config_validation() {
        let config = VPKConfig::default_test();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_vector_dim() {
        let mut config = VPKConfig::default_test();
        config.crypto.vector_dim = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_connection_pool() {
        let mut config = VPKConfig::default_test();
        config.database.max_connections = 1;
        config.database.min_connections = 5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_padded_dim_calculation() {
        let config = VPKConfig::default_test();
        assert_eq!(config.padded_dim(), 384 + 128);
    }

    #[test]
    fn test_load_from_toml() {
        let toml_content = r#"
[database]
connection_string = "postgresql://localhost/vpk"
max_connections = 20
min_connections = 5

[vectordb]
endpoint = "http://localhost:6333"
db_type = "qdrant"
collection_name = "vpk_vectors"
use_tls = false

[crypto]
vector_dim = 1536
padding_dim = 512
auto_rotate_keys = false
rotation_interval_days = 90

[api]
bind_address = "0.0.0.0"
port = 8080
enable_cors = true
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();

        let config = VPKConfig::from_file(temp_file.path()).unwrap();
        assert_eq!(config.database.max_connections, 20);
        assert_eq!(config.crypto.vector_dim, 1536);
        assert_eq!(config.vectordb.endpoint, "http://localhost:6333");
        assert!(config.api.unwrap().enable_cors);
    }
}
