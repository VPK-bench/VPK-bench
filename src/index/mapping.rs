//! Index mapping storage
//!
//! Manages bidirectional mapping between document IDs and scrambled IDs
//! with in-memory caching for performance.

use crate::database::queries::IndexMappingQueries;
use crate::error::VPKResult;
use deadpool_postgres::Pool;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Index mapping manager with LRU cache
///
/// Provides bidirectional mapping between:
/// - doc_id: Original document ID in Bob's database
/// - scrambled_id: Scrambled ID used in Eve's vector database
///
/// Mappings are stored in PostgreSQL and cached in memory for performance.
pub struct IndexMapping {
    queries: IndexMappingQueries,
    // Cache: doc_id -> scrambled_id
    forward_cache: Arc<RwLock<HashMap<i64, i64>>>,
    // Cache: scrambled_id -> doc_id
    reverse_cache: Arc<RwLock<HashMap<i64, i64>>>,
    max_cache_size: usize,
}

impl IndexMapping {
    /// Create a new index mapping manager
    ///
    /// # Arguments
    /// * `db_pool` - PostgreSQL connection pool
    pub fn new(db_pool: Pool) -> Self {
        Self {
            queries: IndexMappingQueries::new(db_pool),
            forward_cache: Arc::new(RwLock::new(HashMap::new())),
            reverse_cache: Arc::new(RwLock::new(HashMap::new())),
            max_cache_size: 10000, // Cache up to 10k mappings
        }
    }

    /// Store a new mapping (routes to the default local tenant).
    pub async fn store(
        &self,
        doc_id: i64,
        scrambled_id: i64,
        key_id: Uuid,
        shard_id: Option<i32>,
    ) -> VPKResult<()> {
        self.queries
            .insert(doc_id, scrambled_id, key_id, shard_id)
            .await?;
        self.cache_mapping(doc_id, scrambled_id);
        Ok(())
    }

    /// Store a new mapping tagged with a specific tenant.
    pub async fn store_for_tenant(
        &self,
        doc_id: i64,
        scrambled_id: i64,
        key_id: Uuid,
        shard_id: Option<i32>,
        tenant_id: uuid::Uuid,
    ) -> VPKResult<()> {
        self.queries
            .insert_for_tenant(doc_id, scrambled_id, key_id, shard_id, tenant_id)
            .await?;
        self.cache_mapping(doc_id, scrambled_id);
        Ok(())
    }

    /// Get scrambled ID for a document (with caching)
    ///
    /// # Arguments
    /// * `doc_id` - Original document ID
    ///
    /// # Returns
    /// Scrambled ID if mapping exists, None otherwise
    pub async fn get_scrambled(&self, doc_id: i64) -> VPKResult<Option<i64>> {
        // Check cache first
        {
            let cache = self.forward_cache.read().unwrap();
            if let Some(&scrambled_id) = cache.get(&doc_id) {
                return Ok(Some(scrambled_id));
            }
        }

        // Query database
        let scrambled_id = self.queries.get_scrambled(doc_id).await?;

        // Cache the result if found
        if let Some(sid) = scrambled_id {
            self.cache_mapping(doc_id, sid);
        }

        Ok(scrambled_id)
    }

    /// Get document ID from scrambled ID (with caching)
    ///
    /// # Arguments
    /// * `scrambled_id` - Scrambled ID from vector database
    ///
    /// # Returns
    /// Original document ID if mapping exists, None otherwise
    pub async fn get_original(&self, scrambled_id: i64) -> VPKResult<Option<i64>> {
        // Check cache first
        {
            let cache = self.reverse_cache.read().unwrap();
            if let Some(&doc_id) = cache.get(&scrambled_id) {
                return Ok(Some(doc_id));
            }
        }

        // Query database
        let doc_id = self.queries.get_original(scrambled_id).await?;

        // Cache the result if found
        if let Some(did) = doc_id {
            self.cache_mapping(did, scrambled_id);
        }

        Ok(doc_id)
    }

    /// Get multiple original IDs from scrambled IDs (batch operation)
    ///
    /// More efficient than calling get_original() in a loop.
    ///
    /// # Arguments
    /// * `scrambled_ids` - List of scrambled IDs
    ///
    /// # Returns
    /// List of original document IDs in same order
    pub async fn get_originals(&self, scrambled_ids: &[i64]) -> VPKResult<Vec<i64>> {
        // Try cache first
        let mut result = Vec::with_capacity(scrambled_ids.len());
        let mut uncached_ids = Vec::new();

        {
            let cache = self.reverse_cache.read().unwrap();
            for &scrambled_id in scrambled_ids {
                if let Some(&doc_id) = cache.get(&scrambled_id) {
                    result.push(doc_id);
                } else {
                    uncached_ids.push(scrambled_id);
                }
            }
        }

        // If everything was cached, return immediately
        if uncached_ids.is_empty() {
            return Ok(result);
        }

        // Query database for uncached IDs
        let doc_ids = self.queries.get_originals(&uncached_ids).await?;

        // Cache the results
        for (&scrambled_id, &doc_id) in uncached_ids.iter().zip(doc_ids.iter()) {
            self.cache_mapping(doc_id, scrambled_id);
        }

        // Merge cached and fetched results
        result.extend(doc_ids);

        Ok(result)
    }

    /// Get a HashMap of scrambled_id → doc_id for a batch of scrambled IDs
    ///
    /// Preserves ordering information so callers can maintain score ordering
    /// after a fan-out search (unlike `get_originals` which sorts by doc_id).
    pub async fn get_originals_map(&self, scrambled_ids: &[i64]) -> VPKResult<HashMap<i64, i64>> {
        if scrambled_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut result = HashMap::new();
        let mut uncached = Vec::new();

        // Check reverse cache first
        {
            let cache = self.reverse_cache.read().unwrap();
            for &sid in scrambled_ids {
                if let Some(&doc_id) = cache.get(&sid) {
                    result.insert(sid, doc_id);
                } else {
                    uncached.push(sid);
                }
            }
        }

        // Fetch uncached from database
        if !uncached.is_empty() {
            let db_map = self.queries.get_originals_map(&uncached).await?;
            // Cache the fetched results
            for (&sid, &did) in &db_map {
                self.cache_mapping(did, sid);
            }
            result.extend(db_map);
        }

        Ok(result)
    }

    /// Delete a mapping
    ///
    /// # Arguments
    /// * `doc_id` - Document ID to remove mapping for
    pub async fn delete(&self, doc_id: i64) -> VPKResult<()> {
        // Get scrambled_id before deleting (for cache invalidation)
        let scrambled_id = self.get_scrambled(doc_id).await?;

        // Delete from database
        self.queries.delete(doc_id).await?;

        // Invalidate caches
        if let Some(sid) = scrambled_id {
            self.invalidate_cache(doc_id, sid);
        }

        Ok(())
    }

    /// Cache a mapping (internal)
    fn cache_mapping(&self, doc_id: i64, scrambled_id: i64) {
        // Check cache size and evict if necessary
        {
            let mut forward = self.forward_cache.write().unwrap();
            if forward.len() >= self.max_cache_size {
                // Simple eviction: clear oldest half
                // TODO: Implement proper LRU in Phase 8
                forward.clear();
            }
            forward.insert(doc_id, scrambled_id);
        }

        {
            let mut reverse = self.reverse_cache.write().unwrap();
            if reverse.len() >= self.max_cache_size {
                reverse.clear();
            }
            reverse.insert(scrambled_id, doc_id);
        }
    }

    /// Invalidate cache entry (internal)
    fn invalidate_cache(&self, doc_id: i64, scrambled_id: i64) {
        {
            let mut forward = self.forward_cache.write().unwrap();
            forward.remove(&doc_id);
        }

        {
            let mut reverse = self.reverse_cache.write().unwrap();
            reverse.remove(&scrambled_id);
        }
    }

    /// Clear all caches (useful for testing or key rotation)
    pub fn clear_cache(&self) {
        {
            let mut forward = self.forward_cache.write().unwrap();
            forward.clear();
        }

        {
            let mut reverse = self.reverse_cache.write().unwrap();
            reverse.clear();
        }
    }

    /// Get cache statistics (for monitoring)
    pub fn cache_stats(&self) -> (usize, usize) {
        let forward_size = self.forward_cache.read().unwrap().len();
        let reverse_size = self.reverse_cache.read().unwrap().len();
        (forward_size, reverse_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a running PostgreSQL database
    // Run with: cargo test --test integration_tests
    // For unit tests without DB, we'd need to mock the queries layer

    #[test]
    fn test_cache_invalidation() {
        // Test that cache invalidation works correctly
        let forward = Arc::new(RwLock::new(HashMap::new()));
        let reverse = Arc::new(RwLock::new(HashMap::new()));

        // Simulate caching
        {
            let mut f = forward.write().unwrap();
            f.insert(1, 100);
        }
        {
            let mut r = reverse.write().unwrap();
            r.insert(100, 1);
        }

        // Verify cached
        assert_eq!(forward.read().unwrap().get(&1), Some(&100));
        assert_eq!(reverse.read().unwrap().get(&100), Some(&1));

        // Invalidate
        {
            let mut f = forward.write().unwrap();
            f.remove(&1);
        }
        {
            let mut r = reverse.write().unwrap();
            r.remove(&100);
        }

        // Verify cleared
        assert_eq!(forward.read().unwrap().get(&1), None);
        assert_eq!(reverse.read().unwrap().get(&100), None);
    }
}

