//! Encryption key management

use crate::crypto::DimensionalScrambling;
use crate::error::VPKResult;
use deadpool_postgres::Pool;
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

/// Key status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyStatus {
    Active,
    Rotating,
    Retired,
}

/// Encryption keys
#[derive(Clone)]
pub struct EncryptionKeys {
    pub key_id: Uuid,
    pub dimensional_scrambling: DimensionalScrambling,
    pub scramble_salt: [u8; 32],
    pub shard_secret: [u8; 32],
    pub status: KeyStatus,
}

impl fmt::Debug for EncryptionKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncryptionKeys")
            .field("key_id", &self.key_id)
            .field("status", &self.status)
            .field("dimensional_scrambling", &self.dimensional_scrambling)
            .field("scramble_salt", &"<redacted>")
            .field("shard_secret", &"<redacted>")
            .finish()
    }
}

/// Key manager
pub struct KeyManager {
    active_keys: HashMap<Uuid, EncryptionKeys>,
    #[allow(dead_code)]
    db_pool: Pool,
}

impl KeyManager {
    /// Create a new key manager
    pub fn new(db_pool: Pool) -> Self {
        Self {
            active_keys: HashMap::new(),
            db_pool,
        }
    }

    /// Generate new encryption keys
    pub async fn generate_keys(
        &mut self,
        vector_dim: usize,
        padding_dim: usize,
    ) -> VPKResult<Uuid> {
        use rand::Rng;

        let key_id = Uuid::new_v4();

        // Generate dimensional scrambling parameters
        let dimensional_scrambling = DimensionalScrambling::new(vector_dim, padding_dim)?;

        // Generate random salts
        let mut rng = rand::thread_rng();
        let mut scramble_salt = [0u8; 32];
        let mut shard_secret = [0u8; 32];
        rng.fill(&mut scramble_salt);
        rng.fill(&mut shard_secret);

        let keys = EncryptionKeys {
            key_id,
            dimensional_scrambling,
            scramble_salt,
            shard_secret,
            status: KeyStatus::Active,
        };

        // Store in memory
        self.active_keys.insert(key_id, keys);

        Ok(key_id)
    }

    /// Get active keys
    pub fn get_active_keys(&self) -> Option<&EncryptionKeys> {
        self.active_keys
            .values()
            .find(|k| k.status == KeyStatus::Active)
    }

    /// Get keys by ID
    pub fn get_keys(&self, key_id: &Uuid) -> Option<&EncryptionKeys> {
        self.active_keys.get(key_id)
    }

    /// Update key status
    pub fn update_status(&mut self, key_id: &Uuid, status: KeyStatus) -> VPKResult<()> {
        if let Some(keys) = self.active_keys.get_mut(key_id) {
            keys.status = status;
            Ok(())
        } else {
            Err(crate::error::VPKError::KeyNotFound(key_id.to_string()))
        }
    }
}

