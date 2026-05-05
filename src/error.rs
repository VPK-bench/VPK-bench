//! Error types for VPK

use thiserror::Error;

/// Main error type for VPK
#[derive(Error, Debug)]
pub enum VPKError {
    /// Encryption errors
    #[error("Encryption error: {0}")]
    EncryptionError(String),

    /// Key generation errors
    #[error("Key generation error: {0}")]
    KeyGenerationError(String),

    /// Decryption errors
    #[error("Decryption error: {0}")]
    DecryptionError(String),

    /// Database errors
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// Shard connection errors
    #[error("Shard connection error: {0}")]
    ShardConnectionError(String),

    /// Index mapping errors
    #[error("Index mapping error: {0}")]
    IndexMappingError(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Vector database errors
    #[error("Vector database error: {0}")]
    VectorDBError(String),

    /// Embedding errors
    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    /// Invalid vector dimension
    #[error("Invalid vector dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    /// Invalid state transition
    #[error("Invalid state transition from {from:?} to {to:?}")]
    InvalidStateTransition { from: String, to: String },

    /// Key not found
    #[error("Key not found: {0}")]
    KeyNotFound(String),

    /// Document not found
    #[error("Document not found: {0}")]
    DocumentNotFound(i64),

    /// IO errors
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Matrix operation errors
    #[error("Matrix operation error: {0}")]
    MatrixError(String),

    /// Padding errors
    #[error("Padding error: {0}")]
    PaddingError(String),

    /// HTTP request errors
    #[error("HTTP error: {0}")]
    HttpError(String),

    /// Timeout errors
    #[error("Timeout error: {0}")]
    TimeoutError(String),

    /// Generic errors
    #[error("{0}")]
    Other(String),
}

/// Result type for VPK operations
pub type VPKResult<T> = Result<T, VPKError>;

// Implement From for common error conversions
impl From<serde_json::Error> for VPKError {
    fn from(err: serde_json::Error) -> Self {
        VPKError::SerializationError(err.to_string())
    }
}

impl From<bincode::Error> for VPKError {
    fn from(err: bincode::Error) -> Self {
        VPKError::SerializationError(err.to_string())
    }
}

impl From<reqwest::Error> for VPKError {
    fn from(err: reqwest::Error) -> Self {
        VPKError::HttpError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = VPKError::InvalidDimension {
            expected: 1536,
            actual: 768,
        };
        assert_eq!(
            err.to_string(),
            "Invalid vector dimension: expected 1536, got 768"
        );
    }

    #[test]
    fn test_error_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json");
        assert!(json_err.is_err());
        let vpk_err: VPKError = json_err.unwrap_err().into();
        assert!(matches!(vpk_err, VPKError::SerializationError(_)));
    }
}
