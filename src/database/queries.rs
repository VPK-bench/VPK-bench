//! Database queries for Bob's PostgreSQL

use crate::crypto::DimensionalScrambling;
use crate::error::{VPKError, VPKResult};
use chrono::{DateTime, Utc};
use deadpool_postgres::Pool;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use uuid::Uuid;

/// Document with metadata
#[derive(Debug, Clone)]
pub struct Document {
    pub doc_id: i64,
    pub content: String,
    pub metadata: Option<JsonValue>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// Tenant this document belongs to (None = pre-migration legacy row).
    pub tenant_id: Option<Uuid>,
}

/// Encryption key record
#[derive(Debug, Clone)]
pub struct EncryptionKeyRecord {
    pub key_id: Uuid,
    pub dimensional_scrambling: Vec<u8>,  // Serialized
    pub scramble_salt: Vec<u8>,
    pub shard_secret: Vec<u8>,
    pub status: String,
    pub vector_dim: i32,
    pub padded_dim: i32,
    pub created_at: DateTime<Utc>,
    pub retired_at: Option<DateTime<Utc>>,
}

/// Index mapping record
#[derive(Debug, Clone)]
pub struct IndexMappingRecord {
    pub doc_id: i64,
    pub scrambled_id: i64,
    pub key_id: Uuid,
    pub shard_id: Option<i32>,
    pub created_at: DateTime<Utc>,
}

/// Document queries
pub struct DocumentQueries {
    pool: Pool,
}

impl DocumentQueries {
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    /// Insert a new document (routes to the default local tenant).
    pub async fn insert(&self, content: &str, metadata: Option<JsonValue>) -> VPKResult<i64> {
        self.insert_for_tenant(content, metadata, TenantQueries::default_id()).await
    }

    /// Insert a new document tagged with a specific tenant.
    pub async fn insert_for_tenant(
        &self,
        content: &str,
        metadata: Option<JsonValue>,
        tenant_id: Uuid,
    ) -> VPKResult<i64> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let row = client
            .query_one(
                "INSERT INTO documents (content, metadata, tenant_id) VALUES ($1, $2, $3) RETURNING doc_id",
                &[&content, &metadata, &tenant_id],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to insert document: {}", e)))?;

        Ok(row.get::<_, i64>("doc_id"))
    }

    /// Get a document by ID
    pub async fn get(&self, doc_id: i64) -> VPKResult<Option<Document>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let row = client
            .query_opt(
                r#"
                SELECT doc_id, content, metadata, created_at, updated_at, tenant_id
                FROM documents
                WHERE doc_id = $1
                "#,
                &[&doc_id],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to get document: {}", e)))?;

        Ok(row.map(|r| Document {
            doc_id: r.get::<_, i64>("doc_id"),
            content: r.get::<_, String>("content"),
            metadata: r.get::<_, Option<JsonValue>>("metadata"),
            created_at: r.get::<_, DateTime<Utc>>("created_at"),
            updated_at: r.get::<_, DateTime<Utc>>("updated_at"),
            tenant_id: r.get::<_, Option<Uuid>>("tenant_id"),
        }))
    }

    /// Get multiple documents by IDs
    pub async fn get_many(&self, doc_ids: &[i64]) -> VPKResult<Vec<Document>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let rows = client
            .query(
                r#"
                SELECT doc_id, content, metadata, created_at, updated_at, tenant_id
                FROM documents
                WHERE doc_id = ANY($1)
                ORDER BY doc_id
                "#,
                &[&doc_ids],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to get documents: {}", e)))?;

        Ok(rows
            .iter()
            .map(|r| Document {
                doc_id: r.get::<_, i64>("doc_id"),
                content: r.get::<_, String>("content"),
                metadata: r.get::<_, Option<JsonValue>>("metadata"),
                created_at: r.get::<_, DateTime<Utc>>("created_at"),
                updated_at: r.get::<_, DateTime<Utc>>("updated_at"),
                tenant_id: r.get::<_, Option<Uuid>>("tenant_id"),
            })
            .collect())
    }

    /// Update document content
    pub async fn update(&self, doc_id: i64, content: &str) -> VPKResult<()> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let rows_affected = client
            .execute(
                "UPDATE documents SET content = $1 WHERE doc_id = $2",
                &[&content, &doc_id],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to update document: {}", e)))?;

        if rows_affected == 0 {
            return Err(VPKError::DocumentNotFound(doc_id));
        }

        Ok(())
    }

    /// Delete a document
    pub async fn delete(&self, doc_id: i64) -> VPKResult<()> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let rows_affected = client
            .execute("DELETE FROM documents WHERE doc_id = $1", &[&doc_id])
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to delete document: {}", e)))?;

        if rows_affected == 0 {
            return Err(VPKError::DocumentNotFound(doc_id));
        }

        Ok(())
    }

    /// List documents with pagination
    pub async fn list(&self, limit: i64, offset: i64) -> VPKResult<Vec<Document>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let rows = client
            .query(
                r#"
                SELECT doc_id, content, metadata, created_at, updated_at, tenant_id
                FROM documents
                ORDER BY doc_id
                LIMIT $1 OFFSET $2
                "#,
                &[&limit, &offset],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to list documents: {}", e)))?;

        Ok(rows
            .iter()
            .map(|r| Document {
                doc_id: r.get::<_, i64>("doc_id"),
                content: r.get::<_, String>("content"),
                metadata: r.get::<_, Option<JsonValue>>("metadata"),
                created_at: r.get::<_, DateTime<Utc>>("created_at"),
                updated_at: r.get::<_, DateTime<Utc>>("updated_at"),
                tenant_id: r.get::<_, Option<Uuid>>("tenant_id"),
            })
            .collect())
    }

    /// Count total documents
    pub async fn count(&self) -> VPKResult<i64> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let row = client
            .query_one("SELECT COUNT(*) as count FROM documents", &[])
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to count documents: {}", e)))?;

        Ok(row.get::<_, i64>("count"))
    }
}

/// Encryption key queries
pub struct KeyQueries {
    pool: Pool,
}

impl KeyQueries {
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    /// Insert a new encryption key
    pub async fn insert(
        &self,
        key_id: Uuid,
        ds: &DimensionalScrambling,
        scramble_salt: &[u8; 32],
        shard_secret: &[u8; 32],
        status: &str,
    ) -> VPKResult<()> {
        let ds_bytes = ds.to_bytes()?;
        let vector_dim = ds.vector_dim() as i32;
        let padded_dim = ds.padded_dim() as i32;
        let salt_vec: Vec<u8> = scramble_salt.to_vec();
        let secret_vec: Vec<u8> = shard_secret.to_vec();

        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        client
            .execute(
                r#"
                INSERT INTO encryption_keys
                (key_id, dimensional_scrambling, scramble_salt, shard_secret, status, vector_dim, padded_dim)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                "#,
                &[&key_id, &ds_bytes, &salt_vec, &secret_vec, &status, &vector_dim, &padded_dim],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to insert key: {}", e)))?;

        Ok(())
    }

    /// Get active encryption key
    pub async fn get_active(&self) -> VPKResult<Option<EncryptionKeyRecord>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let row = client
            .query_opt(
                r#"
                SELECT key_id, dimensional_scrambling, scramble_salt, shard_secret,
                       status, vector_dim, padded_dim, created_at, retired_at
                FROM encryption_keys
                WHERE status = 'Active'
                ORDER BY created_at DESC
                LIMIT 1
                "#,
                &[],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to get active key: {}", e)))?;

        Ok(row.map(|r| EncryptionKeyRecord {
            key_id: r.get::<_, Uuid>("key_id"),
            dimensional_scrambling: r.get::<_, Vec<u8>>("dimensional_scrambling"),
            scramble_salt: r.get::<_, Vec<u8>>("scramble_salt"),
            shard_secret: r.get::<_, Vec<u8>>("shard_secret"),
            status: r.get::<_, String>("status"),
            vector_dim: r.get::<_, i32>("vector_dim"),
            padded_dim: r.get::<_, i32>("padded_dim"),
            created_at: r.get::<_, DateTime<Utc>>("created_at"),
            retired_at: r.get::<_, Option<DateTime<Utc>>>("retired_at"),
        }))
    }

    /// Get encryption key by ID
    pub async fn get(&self, key_id: Uuid) -> VPKResult<Option<EncryptionKeyRecord>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let row = client
            .query_opt(
                r#"
                SELECT key_id, dimensional_scrambling, scramble_salt, shard_secret,
                       status, vector_dim, padded_dim, created_at, retired_at
                FROM encryption_keys
                WHERE key_id = $1
                "#,
                &[&key_id],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to get key: {}", e)))?;

        Ok(row.map(|r| EncryptionKeyRecord {
            key_id: r.get::<_, Uuid>("key_id"),
            dimensional_scrambling: r.get::<_, Vec<u8>>("dimensional_scrambling"),
            scramble_salt: r.get::<_, Vec<u8>>("scramble_salt"),
            shard_secret: r.get::<_, Vec<u8>>("shard_secret"),
            status: r.get::<_, String>("status"),
            vector_dim: r.get::<_, i32>("vector_dim"),
            padded_dim: r.get::<_, i32>("padded_dim"),
            created_at: r.get::<_, DateTime<Utc>>("created_at"),
            retired_at: r.get::<_, Option<DateTime<Utc>>>("retired_at"),
        }))
    }

    /// Update key status
    pub async fn update_status(&self, key_id: Uuid, status: &str) -> VPKResult<()> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        client
            .execute(
                "UPDATE encryption_keys SET status = $1 WHERE key_id = $2",
                &[&status, &key_id],
            )
            .await
            .map_err(|e| {
                VPKError::DatabaseError(format!("Failed to update key status: {}", e))
            })?;

        Ok(())
    }
}

/// Index mapping queries
pub struct IndexMappingQueries {
    pool: Pool,
}

impl IndexMappingQueries {
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    /// Insert index mapping (routes to the default local tenant).
    pub async fn insert(
        &self,
        doc_id: i64,
        scrambled_id: i64,
        key_id: Uuid,
        shard_id: Option<i32>,
    ) -> VPKResult<()> {
        self.insert_for_tenant(doc_id, scrambled_id, key_id, shard_id, TenantQueries::default_id()).await
    }

    /// Insert index mapping tagged with a specific tenant.
    pub async fn insert_for_tenant(
        &self,
        doc_id: i64,
        scrambled_id: i64,
        key_id: Uuid,
        shard_id: Option<i32>,
        tenant_id: Uuid,
    ) -> VPKResult<()> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        client
            .execute(
                "INSERT INTO index_mapping (doc_id, scrambled_id, key_id, shard_id, tenant_id) \
                 VALUES ($1, $2, $3, $4, $5)",
                &[&doc_id, &scrambled_id, &key_id, &shard_id, &tenant_id],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to insert mapping: {}", e)))?;

        Ok(())
    }

    /// Get scrambled ID for document
    pub async fn get_scrambled(&self, doc_id: i64) -> VPKResult<Option<i64>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let row = client
            .query_opt(
                "SELECT scrambled_id FROM index_mapping WHERE doc_id = $1",
                &[&doc_id],
            )
            .await
            .map_err(|e| {
                VPKError::DatabaseError(format!("Failed to get scrambled ID: {}", e))
            })?;

        Ok(row.map(|r| r.get::<_, i64>("scrambled_id")))
    }

    /// Get document ID from scrambled ID
    pub async fn get_original(&self, scrambled_id: i64) -> VPKResult<Option<i64>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let row = client
            .query_opt(
                "SELECT doc_id FROM index_mapping WHERE scrambled_id = $1",
                &[&scrambled_id],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to get doc ID: {}", e)))?;

        Ok(row.map(|r| r.get::<_, i64>("doc_id")))
    }

    /// Get multiple original IDs from scrambled IDs
    pub async fn get_originals(&self, scrambled_ids: &[i64]) -> VPKResult<Vec<i64>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let rows = client
            .query(
                "SELECT doc_id FROM index_mapping WHERE scrambled_id = ANY($1) ORDER BY doc_id",
                &[&scrambled_ids],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to get doc IDs: {}", e)))?;

        Ok(rows.iter().map(|r| r.get::<_, i64>("doc_id")).collect())
    }

    /// Get a map from scrambled_id → doc_id for a batch of scrambled IDs
    ///
    /// Unlike `get_originals`, this preserves the ability to look up by scrambled_id
    /// and maintains correct score-to-doc mapping in the fan-out query path.
    pub async fn get_originals_map(
        &self,
        scrambled_ids: &[i64],
    ) -> VPKResult<HashMap<i64, i64>> {
        if scrambled_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let rows = client
            .query(
                "SELECT scrambled_id, doc_id FROM index_mapping WHERE scrambled_id = ANY($1)",
                &[&scrambled_ids],
            )
            .await
            .map_err(|e| {
                VPKError::DatabaseError(format!("Failed to get doc ID map: {}", e))
            })?;

        Ok(rows
            .iter()
            .map(|r| {
                (
                    r.get::<_, i64>("scrambled_id"),
                    r.get::<_, i64>("doc_id"),
                )
            })
            .collect())
    }

    /// Delete mapping
    pub async fn delete(&self, doc_id: i64) -> VPKResult<()> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        client
            .execute(
                "DELETE FROM index_mapping WHERE doc_id = $1",
                &[&doc_id],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to delete mapping: {}", e)))?;

        Ok(())
    }
}

/// Audit log queries
pub struct AuditLogQueries {
    pool: Pool,
}

impl AuditLogQueries {
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    /// Log an audit event
    pub async fn log(
        &self,
        event_type: &str,
        user_id: Option<&str>,
        details: Option<JsonValue>,
        ip_address: Option<&str>,
    ) -> VPKResult<i64> {
        let ip: Option<String> = ip_address.map(|s| s.to_string());
        let user: Option<String> = user_id.map(|s| s.to_string());

        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let row = client
            .query_one(
                r#"
                INSERT INTO audit_log (event_type, user_id, details, ip_address)
                VALUES ($1, $2, $3, $4::inet)
                RETURNING audit_id
                "#,
                &[&event_type, &user, &details, &ip],
            )
            .await
            .map_err(|e| {
                VPKError::DatabaseError(format!("Failed to log audit event: {}", e))
            })?;

        Ok(row.get::<_, i64>("audit_id"))
    }

    /// Get recent audit events (firewall-style: no plaintext content, no Alice identity)
    pub async fn get_recent(&self, limit: i64) -> VPKResult<Vec<JsonValue>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let rows = client
            .query(
                r#"
                SELECT event_type, details, created_at
                FROM audit_log
                ORDER BY created_at DESC
                LIMIT $1
                "#,
                &[&limit],
            )
            .await
            .map_err(|e| {
                VPKError::DatabaseError(format!("Failed to get audit events: {}", e))
            })?;

        let events: Vec<JsonValue> = rows
            .iter()
            .map(|row| {
                serde_json::json!({
                    "event_type": row.get::<_, String>("event_type"),
                    "details": row.get::<_, Option<JsonValue>>("details"),
                    "timestamp": row.get::<_, DateTime<Utc>>("created_at"),
                })
            })
            .collect();

        Ok(events)
    }

    /// Get audit events since a given timestamp
    pub async fn get_since(
        &self,
        since: DateTime<Utc>,
        limit: i64,
    ) -> VPKResult<Vec<JsonValue>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let rows = client
            .query(
                r#"
                SELECT event_type, details, created_at
                FROM audit_log
                WHERE created_at > $1
                ORDER BY created_at DESC
                LIMIT $2
                "#,
                &[&since, &limit],
            )
            .await
            .map_err(|e| {
                VPKError::DatabaseError(format!("Failed to get audit events since: {}", e))
            })?;

        let events: Vec<JsonValue> = rows
            .iter()
            .map(|row| {
                serde_json::json!({
                    "event_type": row.get::<_, String>("event_type"),
                    "details": row.get::<_, Option<JsonValue>>("details"),
                    "timestamp": row.get::<_, DateTime<Utc>>("created_at"),
                })
            })
            .collect();

        Ok(events)
    }
}

/// Shard configuration record
#[derive(Debug, Clone)]
pub struct ShardConfigRecord {
    pub shard_id: i32,
    pub name: String,
    pub endpoint: Option<String>,
    pub status: String,
    pub num_vectors: i64,
}

/// Shard configuration queries
pub struct ShardConfigQueries {
    pool: Pool,
}

impl ShardConfigQueries {
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    /// Upsert shard configuration (insert or update)
    pub async fn upsert(&self, shard_id: i32, name: &str, endpoint: Option<&str>) -> VPKResult<()> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        client
            .execute(
                r#"
                INSERT INTO shard_configs (shard_id, name, endpoint, status)
                VALUES ($1, $2, $3, 'Active')
                ON CONFLICT (shard_id) DO UPDATE
                    SET name = EXCLUDED.name,
                        endpoint = EXCLUDED.endpoint,
                        updated_at = NOW()
                "#,
                &[&shard_id, &name, &endpoint],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to upsert shard: {}", e)))?;
        Ok(())
    }

    /// Get all active shard configurations
    pub async fn get_all(&self) -> VPKResult<Vec<ShardConfigRecord>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let rows = client
            .query(
                "SELECT shard_id, name, endpoint, status, num_vectors FROM shard_configs ORDER BY shard_id",
                &[],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to get shards: {}", e)))?;

        Ok(rows
            .iter()
            .map(|r| ShardConfigRecord {
                shard_id: r.get::<_, i32>("shard_id"),
                name: r.get::<_, String>("name"),
                endpoint: r.get::<_, Option<String>>("endpoint"),
                status: r.get::<_, String>("status"),
                num_vectors: r.get::<_, i64>("num_vectors"),
            })
            .collect())
    }

    /// Update shard status and vector count
    pub async fn update_status(
        &self,
        shard_id: i32,
        status: &str,
        num_vectors: i64,
    ) -> VPKResult<()> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        client
            .execute(
                "UPDATE shard_configs SET status = $1, num_vectors = $2, updated_at = NOW() WHERE shard_id = $3",
                &[&status, &num_vectors, &shard_id],
            )
            .await
            .map_err(|e| {
                VPKError::DatabaseError(format!("Failed to update shard status: {}", e))
            })?;
        Ok(())
    }
}

/// Server config key-value queries
pub struct ServerConfigQueries {
    pool: Pool,
}

impl ServerConfigQueries {
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    pub async fn set(&self, key: &str, value: &str) -> VPKResult<()> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        client
            .execute(
                r#"
                INSERT INTO server_config (key, value, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                "#,
                &[&key, &value],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to set config: {}", e)))?;
        Ok(())
    }

    pub async fn get(&self, key: &str) -> VPKResult<Option<String>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let rows = client
            .query("SELECT value FROM server_config WHERE key = $1", &[&key])
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to get config: {}", e)))?;
        Ok(rows.first().map(|r| r.get(0)))
    }
}

// ---------------------------------------------------------------------------
// Tenant queries (Phase 1 — multi-tenancy)
// ---------------------------------------------------------------------------

/// Fixed UUID for the implicit default ("local") tenant created on migration.
///
/// Existing single-tenant installs have their rows adopted into this tenant,
/// so legacy code that doesn't pass a tenant_id continues to work correctly.
pub const DEFAULT_TENANT_ID: &str = "00000000-0000-0000-0000-000000000001";

/// A tenant record from Bob's database.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TenantRecord {
    pub tenant_id: Uuid,
    /// Bob's PeerId (libp2p base58) or "local" for the default tenant.
    pub peer_id: String,
    pub display_name: Option<String>,
    pub capacity_limit_mb: i64,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// CRUD queries for the `tenants` table.
pub struct TenantQueries {
    pool: Pool,
}

impl TenantQueries {
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    /// Return the UUID of the default local tenant.
    pub fn default_id() -> Uuid {
        DEFAULT_TENANT_ID.parse().expect("hardcoded UUID is valid")
    }

    /// Create a new tenant.
    pub async fn create(
        &self,
        peer_id: &str,
        capacity_limit_mb: i64,
        display_name: Option<&str>,
    ) -> VPKResult<TenantRecord> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let row = client
            .query_one(
                r#"
                INSERT INTO tenants (peer_id, display_name, capacity_limit_mb)
                VALUES ($1, $2, $3)
                RETURNING tenant_id, peer_id, display_name, capacity_limit_mb,
                          status, created_at, updated_at
                "#,
                &[&peer_id, &display_name, &capacity_limit_mb],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to create tenant: {}", e)))?;

        Ok(Self::row_to_record(&row))
    }

    /// Get a tenant by UUID.
    pub async fn get(&self, tenant_id: Uuid) -> VPKResult<Option<TenantRecord>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let row = client
            .query_opt(
                r#"
                SELECT tenant_id, peer_id, display_name, capacity_limit_mb,
                       status, created_at, updated_at
                FROM tenants
                WHERE tenant_id = $1 AND status != 'Deleted'
                "#,
                &[&tenant_id],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to get tenant: {}", e)))?;

        Ok(row.as_ref().map(Self::row_to_record))
    }

    /// Look up a tenant by peer ID (first non-deleted match).
    pub async fn get_by_peer_id(&self, peer_id: &str) -> VPKResult<Option<TenantRecord>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let row = client
            .query_opt(
                r#"
                SELECT tenant_id, peer_id, display_name, capacity_limit_mb,
                       status, created_at, updated_at
                FROM tenants
                WHERE peer_id = $1 AND status != 'Deleted'
                ORDER BY created_at
                LIMIT 1
                "#,
                &[&peer_id],
            )
            .await
            .map_err(|e| {
                VPKError::DatabaseError(format!("Failed to get tenant by peer_id: {}", e))
            })?;

        Ok(row.as_ref().map(Self::row_to_record))
    }

    /// List all active tenants.
    pub async fn list_active(&self) -> VPKResult<Vec<TenantRecord>> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let rows = client
            .query(
                r#"
                SELECT tenant_id, peer_id, display_name, capacity_limit_mb,
                       status, created_at, updated_at
                FROM tenants
                WHERE status = 'Active'
                ORDER BY created_at
                "#,
                &[],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to list tenants: {}", e)))?;

        Ok(rows.iter().map(Self::row_to_record).collect())
    }

    /// Soft-delete a tenant (sets status = 'Deleted').
    pub async fn delete(&self, tenant_id: Uuid) -> VPKResult<()> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let n = client
            .execute(
                "UPDATE tenants SET status = 'Deleted', updated_at = NOW() WHERE tenant_id = $1",
                &[&tenant_id],
            )
            .await
            .map_err(|e| VPKError::DatabaseError(format!("Failed to delete tenant: {}", e)))?;

        if n == 0 {
            return Err(VPKError::DatabaseError(format!(
                "tenant not found: {}",
                tenant_id
            )));
        }
        Ok(())
    }

    /// Update a tenant's status (Active / Suspended / Deleted).
    pub async fn update_status(&self, tenant_id: Uuid, status: &str) -> VPKResult<()> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        client
            .execute(
                "UPDATE tenants SET status = $1, updated_at = NOW() WHERE tenant_id = $2",
                &[&status, &tenant_id],
            )
            .await
            .map_err(|e| {
                VPKError::DatabaseError(format!("Failed to update tenant status: {}", e))
            })?;
        Ok(())
    }

    /// Document count for a tenant (across Bob's plaintext DB).
    pub async fn document_count(&self, tenant_id: Uuid) -> VPKResult<i64> {
        let client = self.pool.get().await.map_err(|e| {
            VPKError::DatabaseError(format!("Pool error: {}", e))
        })?;
        let row = client
            .query_one(
                "SELECT COUNT(*) FROM documents WHERE tenant_id = $1",
                &[&tenant_id],
            )
            .await
            .map_err(|e| {
                VPKError::DatabaseError(format!("Failed to count tenant documents: {}", e))
            })?;
        Ok(row.get(0))
    }

    fn row_to_record(r: &tokio_postgres::Row) -> TenantRecord {
        TenantRecord {
            tenant_id:         r.get("tenant_id"),
            peer_id:           r.get("peer_id"),
            display_name:      r.get("display_name"),
            capacity_limit_mb: r.get("capacity_limit_mb"),
            status:            r.get("status"),
            created_at:        r.get("created_at"),
            updated_at:        r.get("updated_at"),
        }
    }
}
