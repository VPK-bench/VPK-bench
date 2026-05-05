//! Index scrambling using BLAKE3
//!
//! Provides deterministic bijective mapping between document IDs and scrambled IDs
//! to prevent correlation attacks on the vector database.

use blake3::Hasher;

/// Index scrambler for security
///
/// Uses BLAKE3 hash to create a deterministic but unpredictable mapping
/// between document IDs and scrambled IDs stored in the vector database.
///
/// Properties:
/// - Deterministic: same doc_id always maps to same scrambled_id
/// - Bijective: no collisions (in practice, for reasonable ID ranges)
/// - Unpredictable: knowing some mappings doesn't reveal others
pub struct IndexScrambler {
    salt: [u8; 32],
    rotation_epoch: u64,
}

impl IndexScrambler {
    /// Create a new index scrambler
    ///
    /// # Arguments
    /// * `salt` - 32-byte salt for the hash function (from encryption keys)
    /// * `rotation_epoch` - Epoch number for key rotation support
    pub fn new(salt: [u8; 32], rotation_epoch: u64) -> Self {
        Self {
            salt,
            rotation_epoch,
        }
    }

    /// Scramble a document ID to a scrambled ID
    ///
    /// Uses BLAKE3 hash to create a deterministic mapping:
    /// scrambled_id = hash(salt || rotation_epoch || doc_id)
    ///
    /// # Arguments
    /// * `doc_id` - Original document ID from Bob's database
    ///
    /// # Returns
    /// Scrambled ID to use in Eve's vector database
    pub fn scramble(&self, doc_id: i64) -> i64 {
        let mut hasher = Hasher::new();

        // Input: salt || rotation_epoch || doc_id
        hasher.update(&self.salt);
        hasher.update(&self.rotation_epoch.to_le_bytes());
        hasher.update(&doc_id.to_le_bytes());

        // Get hash output
        let hash = hasher.finalize();

        // Convert first 8 bytes to i64
        // Use absolute value to ensure positive IDs
        let bytes = hash.as_bytes();
        let raw_id = i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]);

        // Return absolute value (ensure positive)
        raw_id.abs()
    }

    /// Note: Unscrambling requires database lookup
    ///
    /// BLAKE3 is a one-way hash function, so we cannot reverse the scrambling.
    /// To unscramble, we must query the index_mapping table in Bob's database.
    ///
    /// This is intentional for security - even if someone gets the scrambled IDs,
    /// they cannot determine the original doc_ids without access to Bob's database.
    ///
    /// Use `IndexMapping::get_original()` instead.
    #[allow(dead_code)]
    fn unscramble_note(&self) {
        // This function exists only for documentation purposes.
        // Actual unscrambling is done via database lookup in IndexMapping.
    }

    /// Get the rotation epoch
    pub fn rotation_epoch(&self) -> u64 {
        self.rotation_epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinism() {
        let salt = [42u8; 32];
        let scrambler = IndexScrambler::new(salt, 0);

        let doc_id = 12345;
        let scrambled1 = scrambler.scramble(doc_id);
        let scrambled2 = scrambler.scramble(doc_id);

        // Same input should always produce same output
        assert_eq!(scrambled1, scrambled2);
    }

    #[test]
    fn test_uniqueness() {
        let salt = [42u8; 32];
        let scrambler = IndexScrambler::new(salt, 0);

        // Test that different doc_ids produce different scrambled_ids
        let mut seen = std::collections::HashSet::new();

        for doc_id in 1..=1000 {
            let scrambled = scrambler.scramble(doc_id);
            assert!(
                !seen.contains(&scrambled),
                "Collision detected: doc_id {} maps to already-seen scrambled_id {}",
                doc_id,
                scrambled
            );
            seen.insert(scrambled);
        }

        // All 1000 doc_ids should produce 1000 unique scrambled_ids
        assert_eq!(seen.len(), 1000);
    }

    #[test]
    fn test_different_salts_produce_different_mappings() {
        let salt1 = [1u8; 32];
        let salt2 = [2u8; 32];

        let scrambler1 = IndexScrambler::new(salt1, 0);
        let scrambler2 = IndexScrambler::new(salt2, 0);

        let doc_id = 12345;
        let scrambled1 = scrambler1.scramble(doc_id);
        let scrambled2 = scrambler2.scramble(doc_id);

        // Different salts should produce different scrambled IDs
        assert_ne!(scrambled1, scrambled2);
    }

    #[test]
    fn test_different_epochs_produce_different_mappings() {
        let salt = [42u8; 32];

        let scrambler1 = IndexScrambler::new(salt, 0);
        let scrambler2 = IndexScrambler::new(salt, 1);

        let doc_id = 12345;
        let scrambled1 = scrambler1.scramble(doc_id);
        let scrambled2 = scrambler2.scramble(doc_id);

        // Different epochs should produce different scrambled IDs
        // This is important for key rotation
        assert_ne!(scrambled1, scrambled2);
    }

    #[test]
    fn test_scrambled_ids_are_positive() {
        let salt = [42u8; 32];
        let scrambler = IndexScrambler::new(salt, 0);

        // Test a range of doc_ids including negative ones
        for doc_id in -100..=100 {
            let scrambled = scrambler.scramble(doc_id);
            assert!(scrambled >= 0, "Scrambled ID should be positive, got {}", scrambled);
        }
    }

    #[test]
    fn test_no_correlation_between_adjacent_ids() {
        let salt = [42u8; 32];
        let scrambler = IndexScrambler::new(salt, 0);

        // Adjacent doc_ids should produce uncorrelated scrambled_ids
        let scrambled1 = scrambler.scramble(1000);
        let scrambled2 = scrambler.scramble(1001);
        let scrambled3 = scrambler.scramble(1002);

        // The differences should not be similar
        let diff1 = (scrambled2 - scrambled1).abs();
        let diff2 = (scrambled3 - scrambled2).abs();

        // If there was correlation, these differences would be similar
        // With BLAKE3, they should be completely different
        assert_ne!(diff1, diff2);

        // Also check that adjacent IDs don't map to adjacent scrambled IDs
        assert!(diff1 > 1000, "Adjacent doc_ids should not map to adjacent scrambled_ids");
    }

    #[test]
    fn test_large_doc_ids() {
        let salt = [42u8; 32];
        let scrambler = IndexScrambler::new(salt, 0);

        // Test with large doc_ids (like in production)
        let large_id = i64::MAX / 2;
        let scrambled = scrambler.scramble(large_id);

        assert!(scrambled > 0);

        // Should be deterministic even for large IDs
        let scrambled2 = scrambler.scramble(large_id);
        assert_eq!(scrambled, scrambled2);
    }
}

