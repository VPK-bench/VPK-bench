//! Core VPK orchestrator
//!
//! The VPK struct integrates all components and provides the main API
//! for encrypted vector search.

use crate::config::{EncryptionAlgorithm, VPKConfig};
use crate::crypto::{Ckks, DimensionalScrambling, EncryptionPipeline, KeyManager};
use crate::database::{queries::*, Database, DatabaseConfig};
use crate::embedding::{EmbeddingModel, HttpEmbeddingModel};
use crate::error::{VPKError, VPKResult};
use crate::index::{IndexMapping, IndexScrambler};
use crate::shard::{ShardHealthStatus, ShardManager};
use ndarray::Array1;
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// VPK state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VPKState {
    /// Initial state - not yet initialized
    Uninitialized,
    /// Currently initializing (loading keys, connecting to DBs)
    Initializing,
    /// Ready for normal operations
    Active,
    /// Currently rotating encryption keys
    RotatingKeys,
    /// Degraded mode (some components unhealthy)
    Degraded,
}

/// Upload result statistics
#[derive(Debug, Clone)]
pub struct UploadResult {
    pub documents_inserted: usize,
    pub vectors_uploaded: usize,
    pub errors: Vec<String>,
    /// Real document IDs assigned by Bob's database (in insertion order)
    pub doc_ids: Vec<i64>,
    /// Total wall-clock time for the operation in microseconds
    pub elapsed_us: u64,
    /// Time spent on embedding generation in microseconds
    pub embed_us: u64,
    /// Time spent on encryption in microseconds
    pub encrypt_us: u64,
}

/// Query result with decrypted documents
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub documents: Vec<Document>,
    pub scores: Vec<f64>,
    pub doc_ids: Vec<i64>,
    /// Time spent embedding the query (μs)
    pub embed_us: u64,
    /// Time spent encrypting the query vector (μs)
    pub encrypt_us: u64,
    /// Time spent in shard fan-out + merge (μs)
    pub search_us: u64,
    /// Total wall-clock time for the full query (μs)
    pub total_us: u64,
}

/// System health status (returned by get_health MCP tool)
#[derive(Debug, Serialize)]
pub struct SystemHealthStatus {
    pub state: String,
    pub documents: usize,
    pub total_vectors: usize,
    pub shards: Vec<ShardHealthStatus>,
    pub embedding_dimension: usize,
    pub shard_count: usize,
    pub active_key_id: Option<String>,
}

/// System statistics
#[derive(Debug, Clone)]
pub struct SystemStats {
    pub state: VPKState,
    pub documents: usize,
    pub vectors: usize,
    pub cache_entries: usize,
    pub active_key_id: Option<Uuid>,
}

/// Main VPK struct - integrates all components
///
/// This is the primary interface for VPK operations:
/// - Document upload and indexing
/// - Encrypted vector search with multi-shard fan-out
/// - Key management and rotation
pub struct VPK {
    // Configuration
    config: VPKConfig,

    // State management
    state: Arc<RwLock<VPKState>>,

    // Bob's database (trusted)
    database: Database,
    doc_queries: DocumentQueries,
    key_queries: KeyQueries,
    audit_queries: AuditLogQueries,
    shard_queries: ShardConfigQueries,
    server_config_queries: ServerConfigQueries,
    tenant_queries: TenantQueries,

    // Encryption and key management
    key_manager: KeyManager,
    current_key_id: Option<Uuid>,
    pub dimensional_scrambling: Option<DimensionalScrambling>,
    encryption_pipeline: Option<EncryptionPipeline>,

    // Index scrambling and mapping
    scrambler: Option<IndexScrambler>,
    index_mapping: IndexMapping,

    // Embedding generation (model identity is a hidden security parameter)
    pub embedding_model: Box<dyn EmbeddingModel>,

    // Real CKKS context (Microsoft SEAL). Present only when CKKS algorithm is active.
    ckks_context: Option<Arc<Ckks>>,

    // Eve's sharded vector databases (untrusted)
    shard_manager: ShardManager,
}

impl VPK {
    /// Create a new VPK instance (uninitialized)
    pub async fn new(config: VPKConfig) -> VPKResult<Self> {
        let db_config = DatabaseConfig {
            connection_string: config.database.connection_string.clone(),
            max_connections: config.database.max_connections,
            min_connections: config.database.min_connections,
        };

        let database = Database::new(&db_config).await?;
        let pool = database.pool().clone();

        let doc_queries = DocumentQueries::new(pool.clone());
        let key_queries = KeyQueries::new(pool.clone());
        let audit_queries = AuditLogQueries::new(pool.clone());
        let shard_queries = ShardConfigQueries::new(pool.clone());
        let server_config_queries = ServerConfigQueries::new(pool.clone());
        let key_manager = KeyManager::new(pool.clone());
        let index_mapping = IndexMapping::new(pool.clone());
        let tenant_queries = TenantQueries::new(pool.clone());

        // Build ShardManager from config (MVP: all in-memory)
        let is_ckks = matches!(
            config.crypto.algorithm,
            EncryptionAlgorithm::Ckks | EncryptionAlgorithm::CkksCombined
        );
        let vectordb_dim = match config.crypto.algorithm {
            EncryptionAlgorithm::Rome | EncryptionAlgorithm::RomeCombined => {
                config.crypto.vector_dim + config.crypto.padding_dim
            }
            EncryptionAlgorithm::Diehard | EncryptionAlgorithm::DiehardCombined => {
                config.crypto.vector_dim + config.crypto.padding_dim
            }
            EncryptionAlgorithm::Ckks | EncryptionAlgorithm::CkksCombined => {
                config.crypto.vector_dim
            }
            _ => config.crypto.vector_dim,
        };

        let ckks_context = if is_ckks {
            Some(Arc::new(Ckks::new(
                config.crypto.vector_dim,
                config.crypto.vector_dim + config.crypto.padding_dim,
            )?))
        } else {
            None
        };

        let use_dot_product = matches!(
            config.crypto.algorithm,
            EncryptionAlgorithm::Diehard | EncryptionAlgorithm::DiehardCombined
        );
        let shard_manager = ShardManager::new(
            &config.shards,
            config.peer_id.as_deref().unwrap_or(""),
            vectordb_dim,
            ckks_context.clone(),
            use_dot_product,
        );

        let embedding_model: Box<dyn EmbeddingModel> =
            Box::new(HttpEmbeddingModel::new("http://localhost:5001", config.crypto.vector_dim));

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(VPKState::Uninitialized)),
            database,
            doc_queries,
            key_queries,
            audit_queries,
            shard_queries,
            server_config_queries,
            tenant_queries,
            key_manager,
            current_key_id: None,
            dimensional_scrambling: None,
            encryption_pipeline: None,
            scrambler: None,
            index_mapping,
            embedding_model,
            ckks_context,
            shard_manager,
        })
    }

    /// Initialize the VPK system
    ///
    /// 1. Run database migrations
    /// 2. Load or generate encryption keys
    /// 3. Initialize index scrambler
    /// 4. Persist shard config to database
    /// 5. Verify health
    /// 6. Transition to Active state
    pub async fn initialize(&mut self) -> VPKResult<()> {
        self.set_state(VPKState::Initializing)?;

        self.database.migrate().await?;
        self.load_persisted_config().await?;

        // Rebuild shard manager with correct dimension for the (possibly restored) algorithm
        let is_ckks = matches!(
            self.config.crypto.algorithm,
            EncryptionAlgorithm::Ckks | EncryptionAlgorithm::CkksCombined
        );
        let vectordb_dim = match self.config.crypto.algorithm {
            EncryptionAlgorithm::Rome | EncryptionAlgorithm::RomeCombined => {
                self.config.crypto.vector_dim + self.config.crypto.padding_dim
            }
            EncryptionAlgorithm::Diehard | EncryptionAlgorithm::DiehardCombined => {
                self.config.crypto.vector_dim + self.config.crypto.padding_dim
            }
            EncryptionAlgorithm::Ckks | EncryptionAlgorithm::CkksCombined => {
                self.config.crypto.vector_dim
            }
            _ => self.config.crypto.vector_dim,
        };
        if is_ckks && self.ckks_context.is_none() {
            self.ckks_context = Some(Arc::new(Ckks::new(
                self.config.crypto.vector_dim,
                self.config.crypto.vector_dim + self.config.crypto.padding_dim,
            )?));
        } else if !is_ckks {
            self.ckks_context = None;
        }
        let use_dot_product = matches!(
            self.config.crypto.algorithm,
            EncryptionAlgorithm::Diehard | EncryptionAlgorithm::DiehardCombined
        );
        self.shard_manager = ShardManager::new(
            &self.config.shards,
            self.config.peer_id.as_deref().unwrap_or(""),
            vectordb_dim,
            self.ckks_context.clone(),
            use_dot_product,
        );

        self.load_or_generate_keys().await?;
        self.init_scrambler()?;
        self.persist_shard_configs().await?;
        self.run_health_checks().await?;

        self.set_state(VPKState::Active)?;
        Ok(())
    }

    /// Load persisted server config (algorithm, noise range) from the database
    async fn load_persisted_config(&mut self) -> VPKResult<()> {
        if let Some(algo_str) = self.server_config_queries.get("active_algorithm").await? {
            let algorithm = match algo_str.as_str() {
                "scrambling"    => EncryptionAlgorithm::Scrambling,
                "noise"         => EncryptionAlgorithm::Noise,
                "combined"      => EncryptionAlgorithm::Combined,
                "rome"          => EncryptionAlgorithm::Rome,
                "rome-combined" => EncryptionAlgorithm::RomeCombined,
                "romm"             => EncryptionAlgorithm::Romm,
                "romm-combined"    => EncryptionAlgorithm::RommCombined,
                "diehard"          => EncryptionAlgorithm::Diehard,
                "diehard-combined" => EncryptionAlgorithm::DiehardCombined,
                "ckks"             => EncryptionAlgorithm::Ckks,
                "ckks-combined" => EncryptionAlgorithm::CkksCombined,
                _               => EncryptionAlgorithm::Combined,
            };
            self.config.crypto.algorithm = algorithm;
        }
        if let Some(min_str) = self.server_config_queries.get("noise_min").await? {
            if let Ok(min) = min_str.parse::<f64>() {
                self.config.crypto.noise_range.0 = min;
            }
        }
        if let Some(max_str) = self.server_config_queries.get("noise_max").await? {
            if let Ok(max) = max_str.parse::<f64>() {
                self.config.crypto.noise_range.1 = max;
            }
        }
        Ok(())
    }

    /// Persist shard configurations to database
    async fn persist_shard_configs(&self) -> VPKResult<()> {
        for shard in &self.config.shards {
            self.shard_queries
                .upsert(shard.shard_id, &shard.name, shard.endpoint.as_deref())
                .await?;
        }
        Ok(())
    }

    /// Load existing keys or generate new ones
    async fn load_or_generate_keys(&mut self) -> VPKResult<()> {
        if let Some(key_record) = self.key_queries.get_active().await? {
            let ds = DimensionalScrambling::from_bytes(&key_record.dimensional_scrambling)?;
            self.dimensional_scrambling = Some(ds);
            self.current_key_id = Some(key_record.key_id);

            let salt: [u8; 32] = key_record
                .scramble_salt
                .try_into()
                .map_err(|_| VPKError::KeyGenerationError("Invalid salt length".to_string()))?;
            self.scrambler = Some(IndexScrambler::new(salt, 0));
            self.init_encryption_pipeline()?;
            return Ok(());
        }

        let key_id = self
            .key_manager
            .generate_keys(self.config.crypto.vector_dim, self.config.crypto.padding_dim)
            .await?;

        let keys = self
            .key_manager
            .get_active_keys()
            .ok_or_else(|| VPKError::KeyGenerationError("Failed to get generated keys".to_string()))?;

        self.key_queries
            .insert(
                key_id,
                &keys.dimensional_scrambling,
                &keys.scramble_salt,
                &keys.shard_secret,
                "Active",
            )
            .await?;

        self.dimensional_scrambling = Some(keys.dimensional_scrambling.clone());
        self.current_key_id = Some(key_id);
        self.scrambler = Some(IndexScrambler::new(keys.scramble_salt, 0));
        self.init_encryption_pipeline()?;
        Ok(())
    }

    fn init_scrambler(&mut self) -> VPKResult<()> {
        if self.scrambler.is_none() {
            return Err(VPKError::KeyGenerationError(
                "Scrambler not initialized".to_string(),
            ));
        }
        Ok(())
    }

    fn init_encryption_pipeline(&mut self) -> VPKResult<()> {
        let vector_dim = self.config.crypto.vector_dim;
        let padding_dim = self.config.crypto.padding_dim;
        let (noise_min, noise_max) = self.config.crypto.noise_range;

        let pipeline = match self.config.crypto.algorithm {
            EncryptionAlgorithm::Scrambling => EncryptionPipeline::builder()
                .add_scrambling(vector_dim)?
                .build()?,
            EncryptionAlgorithm::Noise | EncryptionAlgorithm::Combined => {
                EncryptionPipeline::builder()
                    .add_scrambling(vector_dim)?
                    .add_noise(noise_min, noise_max)?
                    .build()?
            }
            EncryptionAlgorithm::Rome => {
                let padded_dim = vector_dim + padding_dim;
                EncryptionPipeline::builder()
                    .add_rome(vector_dim, padded_dim)?
                    .build()?
            }
            EncryptionAlgorithm::RomeCombined => {
                let padded_dim = vector_dim + padding_dim;
                EncryptionPipeline::builder()
                    .add_rome(vector_dim, padded_dim)?
                    .add_noise(noise_min, noise_max)?
                    .build()?
            }
            EncryptionAlgorithm::Romm => {
                EncryptionPipeline::builder()
                    .add_romm(vector_dim)?
                    .build()?
            }
            EncryptionAlgorithm::RommCombined => {
                EncryptionPipeline::builder()
                    .add_romm(vector_dim)?
                    .add_noise(noise_min, noise_max)?
                    .build()?
            }
            EncryptionAlgorithm::Diehard => {
                let output_dim = vector_dim + padding_dim;
                EncryptionPipeline::builder()
                    .add_diehard(vector_dim, output_dim)?
                    .build()?
            }
            EncryptionAlgorithm::DiehardCombined => {
                let output_dim = vector_dim + padding_dim;
                EncryptionPipeline::builder()
                    .add_diehard(vector_dim, output_dim)?
                    .add_noise(noise_min, noise_max)?
                    .build()?
            }
            EncryptionAlgorithm::Ckks => {
                let padded_dim = vector_dim + padding_dim;
                EncryptionPipeline::builder()
                    .add_ckks(vector_dim, padded_dim)?
                    .build()?
            }
            EncryptionAlgorithm::CkksCombined => {
                let padded_dim = vector_dim + padding_dim;
                EncryptionPipeline::builder()
                    .add_ckks(vector_dim, padded_dim)?
                    .add_noise(noise_min, noise_max)?
                    .build()?
            }
        };

        self.encryption_pipeline = Some(pipeline);
        Ok(())
    }

    async fn run_health_checks(&self) -> VPKResult<()> {
        self.database.health_check().await?;

        let shard_statuses = self.shard_manager.health_check_all().await;
        let all_healthy = shard_statuses.iter().all(|s| s.healthy);
        if !all_healthy {
            tracing::warn!(
                "Some shards are unhealthy during startup: {:?}",
                shard_statuses
                    .iter()
                    .filter(|s| !s.healthy)
                    .collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    fn set_state(&self, new_state: VPKState) -> VPKResult<()> {
        let mut state = self.state.write().unwrap();
        match (*state, new_state) {
            (VPKState::Uninitialized, VPKState::Initializing) => {}
            (VPKState::Initializing, VPKState::Active) => {}
            (VPKState::Active, VPKState::RotatingKeys) => {}
            (VPKState::RotatingKeys, VPKState::Active) => {}
            (_, VPKState::Degraded) => {}
            (VPKState::Degraded, VPKState::Active) => {}
            _ => {
                return Err(VPKError::InvalidStateTransition {
                    from: format!("{:?}", *state),
                    to: format!("{:?}", new_state),
                });
            }
        }
        *state = new_state;
        Ok(())
    }

    pub fn state(&self) -> VPKState {
        *self.state.read().unwrap()
    }

    pub async fn set_algorithm(
        &mut self,
        algorithm: EncryptionAlgorithm,
        noise_range: Option<(f64, f64)>,
    ) -> VPKResult<()> {
        let is_ckks = matches!(
            algorithm,
            EncryptionAlgorithm::Ckks | EncryptionAlgorithm::CkksCombined
        );
        let vectordb_dim = match algorithm {
            EncryptionAlgorithm::Rome | EncryptionAlgorithm::RomeCombined => {
                self.config.crypto.vector_dim + self.config.crypto.padding_dim
            }
            EncryptionAlgorithm::Diehard | EncryptionAlgorithm::DiehardCombined => {
                self.config.crypto.vector_dim + self.config.crypto.padding_dim
            }
            EncryptionAlgorithm::Ckks | EncryptionAlgorithm::CkksCombined => {
                self.config.crypto.vector_dim
            }
            _ => self.config.crypto.vector_dim,
        };

        self.config.crypto.algorithm = algorithm.clone();
        if let Some(range) = noise_range {
            self.config.crypto.noise_range = range;
        }
        self.init_encryption_pipeline()?;

        if is_ckks && self.ckks_context.is_none() {
            self.ckks_context = Some(Arc::new(Ckks::new(
                self.config.crypto.vector_dim,
                self.config.crypto.vector_dim + self.config.crypto.padding_dim,
            )?));
        } else if !is_ckks {
            self.ckks_context = None;
        }

        // Rebuild shard manager with correct dimension and metric
        let use_dot_product = matches!(
            algorithm,
            EncryptionAlgorithm::Diehard | EncryptionAlgorithm::DiehardCombined
        );
        self.shard_manager = ShardManager::new(
            &self.config.shards,
            self.config.peer_id.as_deref().unwrap_or(""),
            vectordb_dim,
            self.ckks_context.clone(),
            use_dot_product,
        );

        // Persist so the algorithm survives server restarts
        let algo_str = self.config.crypto.algorithm.as_str();
        let _ = self.server_config_queries.set("active_algorithm", algo_str).await;
        let _ = self.server_config_queries.set("noise_min", &self.config.crypto.noise_range.0.to_string()).await;
        let _ = self.server_config_queries.set("noise_max", &self.config.crypto.noise_range.1.to_string()).await;

        Ok(())
    }

    /// Update only the noise component of the current pipeline, preserving the
    /// DS / ROME / DIEHARD / CKKS rotation matrix.
    ///
    /// Used by Experiment G: after establishing a DS matrix for a trial, call
    /// this to sweep noise levels without ever regenerating the rotation keys.
    /// The caller must still invoke `/api/reencrypt` after this, because the
    /// stored document vectors were encrypted with the old noise vector.
    pub fn set_noise_only(&mut self, noise_range: (f64, f64)) -> VPKResult<()> {
        self.config.crypto.noise_range = noise_range;
        match self.encryption_pipeline.as_mut() {
            Some(p) => p.reinit_noise_only(noise_range),
            None => Err(VPKError::KeyGenerationError(
                "No pipeline initialised; call set_algorithm before set_noise_only".to_string(),
            )),
        }
    }

    pub fn get_algorithm(&self) -> EncryptionAlgorithm {
        self.config.crypto.algorithm.clone()
    }

    pub fn get_noise_range(&self) -> (f64, f64) {
        self.config.crypto.noise_range
    }

    /// Number of configured Eve shards
    pub fn shard_count(&self) -> usize {
        self.shard_manager.num_shards()
    }

    /// Upload a single document (for MCP tool)
    ///
    /// Returns the real document ID assigned by Bob's database.
    pub async fn upload_document(
        &mut self,
        text: String,
        metadata: Option<JsonValue>,
    ) -> VPKResult<i64> {
        if self.state() != VPKState::Active {
            return Err(VPKError::Other(format!(
                "Cannot upload in state: {:?}",
                self.state()
            )));
        }

        let key_id = self
            .current_key_id
            .ok_or_else(|| VPKError::KeyNotFound("No active key".to_string()))?;

        let scrambler = self
            .scrambler
            .as_ref()
            .ok_or_else(|| VPKError::Other("Scrambler not initialized".to_string()))?;

        let pipeline = self
            .encryption_pipeline
            .as_ref()
            .ok_or_else(|| VPKError::Other("Encryption pipeline not initialized".to_string()))?;

        // Step 1: Store plaintext in Bob's DB
        let doc_id = self.doc_queries.insert(&text, metadata).await?;

        // Step 2: Embed
        let embedding = self.embedding_model.embed(&text).await?;

        // Step 3: Encrypt
        let encrypted_vector = pipeline.encrypt_document(&embedding)?;

        // Step 4: Scramble ID
        let scrambled_id = scrambler.scramble(doc_id);

        // Step 5: Assign to shard (round-robin)
        let shard_id = self.shard_manager.assign_shard();

        // Step 6: Store index mapping with shard assignment
        self.index_mapping
            .store(doc_id, scrambled_id, key_id, Some(shard_id))
            .await?;

        // Step 7: Upload to Eve shard
        self.shard_manager
            .upload_to_shard(shard_id, vec![(scrambled_id, encrypted_vector)])
            .await?;

        // Audit log: DOCUMENT_UPLOADED (no content logged)
        let _ = self
            .audit_queries
            .log(
                "DOCUMENT_UPLOADED",
                None,
                Some(serde_json::json!({ "shard_id": shard_id })),
                None,
            )
            .await;

        Ok(doc_id)
    }

    /// Upload a single document tagged with a specific tenant.
    ///
    /// Identical to `upload_document` but records `tenant_id` in both
    /// `documents` and `index_mapping` so the row is scoped to that tenant.
    pub async fn upload_document_for_tenant(
        &mut self,
        text: String,
        metadata: Option<JsonValue>,
        tenant_id: Uuid,
    ) -> VPKResult<i64> {
        if self.state() != VPKState::Active {
            return Err(VPKError::Other(format!(
                "Cannot upload in state: {:?}",
                self.state()
            )));
        }

        let key_id = self
            .current_key_id
            .ok_or_else(|| VPKError::KeyNotFound("No active key".to_string()))?;

        let scrambler = self
            .scrambler
            .as_ref()
            .ok_or_else(|| VPKError::Other("Scrambler not initialized".to_string()))?;

        let pipeline = self
            .encryption_pipeline
            .as_ref()
            .ok_or_else(|| VPKError::Other("Encryption pipeline not initialized".to_string()))?;

        let doc_id = self.doc_queries.insert_for_tenant(&text, metadata, tenant_id).await?;
        let embedding = self.embedding_model.embed(&text).await?;
        let encrypted_vector = pipeline.encrypt_document(&embedding)?;
        let scrambled_id = scrambler.scramble(doc_id);
        let shard_id = self.shard_manager.assign_shard();

        self.index_mapping
            .store_for_tenant(doc_id, scrambled_id, key_id, Some(shard_id), tenant_id)
            .await?;

        self.shard_manager
            .upload_to_shard(shard_id, vec![(scrambled_id, encrypted_vector)])
            .await?;

        let _ = self
            .audit_queries
            .log(
                "DOCUMENT_UPLOADED",
                None,
                Some(serde_json::json!({ "shard_id": shard_id, "tenant_id": tenant_id })),
                None,
            )
            .await;

        Ok(doc_id)
    }

    /// Query for similar documents scoped to a specific tenant.
    ///
    /// Same fan-out pipeline as `query` but the returned documents are filtered
    /// to those belonging to `tenant_id`, preventing cross-tenant data leakage.
    pub async fn query_for_tenant(
        &self,
        query_text: &str,
        top_k: usize,
        tenant_id: Uuid,
    ) -> VPKResult<QueryResult> {
        if self.state() != VPKState::Active {
            return Err(VPKError::Other(format!(
                "Cannot query in state: {:?}",
                self.state()
            )));
        }

        let pipeline = self
            .encryption_pipeline
            .as_ref()
            .ok_or_else(|| VPKError::Other("Encryption pipeline not initialized".to_string()))?;

        let t_total = std::time::Instant::now();
        let t0 = std::time::Instant::now();
        let query_embedding = self.embedding_model.embed(query_text).await?;
        let embed_us = t0.elapsed().as_micros() as u64;
        let t0 = std::time::Instant::now();
        let encrypted_query = pipeline.encrypt_query(&query_embedding)?;
        if let Some(ckks) = &self.ckks_context {
            let ct = ckks.encrypt(&encrypted_query)?;
            ckks.cache_query_ct(ct);
        }
        let encrypt_us = t0.elapsed().as_micros() as u64;
        let t0 = std::time::Instant::now();
        let shard_results = self.shard_manager.search_all(&encrypted_query, top_k).await?;
        let search_us = t0.elapsed().as_micros() as u64;
        if let Some(ckks) = &self.ckks_context {
            ckks.clear_query_ct_cache();
        }

        if shard_results.is_empty() {
            return Ok(QueryResult {
                documents: vec![], scores: vec![], doc_ids: vec![],
                embed_us, encrypt_us, search_us,
                total_us: t_total.elapsed().as_micros() as u64,
            });
        }

        let scrambled_ids: Vec<i64> = shard_results.iter().map(|r| r.scrambled_id).collect();
        let id_map = self.index_mapping.get_originals_map(&scrambled_ids).await?;

        let mut documents = Vec::new();
        let mut scores = Vec::new();
        let mut result_doc_ids = Vec::new();

        for result in &shard_results {
            if let Some(&doc_id) = id_map.get(&result.scrambled_id) {
                match self.doc_queries.get(doc_id).await? {
                    Some(doc) => {
                        // Tenant filter: skip documents that belong to a different tenant.
                        if doc.tenant_id != Some(tenant_id) {
                            continue;
                        }
                        documents.push(doc);
                        scores.push(result.score);
                        result_doc_ids.push(doc_id);
                    }
                    None => {
                        tracing::warn!("Document {} not found (stale index?)", doc_id);
                    }
                }
            }
        }

        let shard_ids_hit: Vec<i32> = shard_results.iter().map(|r| r.shard_id).collect();
        let _ = self
            .audit_queries
            .log(
                "SEARCH",
                None,
                Some(serde_json::json!({
                    "shards_queried": shard_ids_hit,
                    "results_returned": result_doc_ids.len(),
                    "top_k_requested": top_k,
                    "tenant_id": tenant_id,
                })),
                None,
            )
            .await;

        Ok(QueryResult {
            documents,
            scores,
            doc_ids: result_doc_ids,
            embed_us,
            encrypt_us,
            search_us,
            total_us: t_total.elapsed().as_micros() as u64,
        })
    }

    /// Upload documents in batch
    ///
    /// Complete flow:
    /// 1. Store plaintext in Bob's database
    /// 2. Generate embeddings
    /// 3. Encrypt embeddings
    /// 4. Scramble document IDs
    /// 5. Assign to shards (round-robin)
    /// 6. Store index mappings
    /// 7. Upload encrypted vectors per-shard
    pub async fn upload_documents(&mut self, contents: Vec<String>) -> VPKResult<UploadResult> {
        let total_start = std::time::Instant::now();

        if self.state() != VPKState::Active {
            return Err(VPKError::Other(format!(
                "Cannot upload documents in state: {:?}",
                self.state()
            )));
        }

        let mut documents_inserted = 0;
        let mut vectors_uploaded = 0;
        let mut errors = Vec::new();
        let mut doc_ids: Vec<i64> = Vec::new();
        let mut embed_us: u64 = 0;
        let mut encrypt_us: u64 = 0;

        let key_id = self
            .current_key_id
            .ok_or_else(|| VPKError::KeyNotFound("No active key".to_string()))?;

        let scrambler = self
            .scrambler
            .as_ref()
            .ok_or_else(|| VPKError::Other("Scrambler not initialized".to_string()))?;

        let pipeline = self
            .encryption_pipeline
            .as_ref()
            .ok_or_else(|| VPKError::Other("Encryption pipeline not initialized".to_string()))?;

        // Collect (shard_id → vectors) for batch upload
        let mut vectors_per_shard: HashMap<i32, Vec<(i64, Array1<f64>)>> = HashMap::new();

        for content in contents {
            let doc_id = match self.doc_queries.insert(&content, None).await {
                Ok(id) => id,
                Err(e) => {
                    errors.push(format!("Failed to insert document: {}", e));
                    continue;
                }
            };

            let t0 = std::time::Instant::now();
            let embedding = match self.embedding_model.embed(&content).await {
                Ok(emb) => emb,
                Err(e) => {
                    errors.push(format!("Failed to embed doc {}: {}", doc_id, e));
                    continue;
                }
            };
            embed_us += t0.elapsed().as_micros() as u64;

            let t0 = std::time::Instant::now();
            let encrypted_vector = match pipeline.encrypt_document(&embedding) {
                Ok(enc) => enc,
                Err(e) => {
                    errors.push(format!("Failed to encrypt doc {}: {}", doc_id, e));
                    continue;
                }
            };
            encrypt_us += t0.elapsed().as_micros() as u64;

            let scrambled_id = scrambler.scramble(doc_id);
            let shard_id = self.shard_manager.assign_shard();

            if let Err(e) = self
                .index_mapping
                .store(doc_id, scrambled_id, key_id, Some(shard_id))
                .await
            {
                errors.push(format!("Failed to store mapping for doc {}: {}", doc_id, e));
                continue;
            }

            vectors_per_shard
                .entry(shard_id)
                .or_default()
                .push((scrambled_id, encrypted_vector));

            doc_ids.push(doc_id);
            documents_inserted += 1;
        }

        // Batch upload per shard
        for (shard_id, vectors) in vectors_per_shard {
            match self.shard_manager.upload_to_shard(shard_id, vectors).await {
                Ok(count) => vectors_uploaded += count,
                Err(e) => {
                    errors.push(format!("Failed to upload to shard {}: {}", shard_id, e));
                }
            }
        }

        Ok(UploadResult {
            documents_inserted,
            vectors_uploaded,
            errors,
            doc_ids,
            elapsed_us: total_start.elapsed().as_micros() as u64,
            embed_us,
            encrypt_us,
        })
    }

    /// Clear all documents, index mappings, and shard vectors
    pub async fn clear_all_documents(&mut self) -> VPKResult<()> {
        let client = self
            .database
            .pool()
            .get()
            .await
            .map_err(|e| VPKError::Other(format!("Pool error: {}", e)))?;
        client
            .execute("TRUNCATE documents, index_mapping CASCADE", &[])
            .await
            .map_err(|e| VPKError::Other(format!("Failed to clear documents: {}", e)))?;
        self.shard_manager.clear_all().await?;
        Ok(())
    }

    /// Re-encrypt all documents with current algorithm (clears shards and re-uploads).
    /// If `progress_tx` is provided, sends progress updates as `(processed, total, embed_us, encrypt_us)`.
    pub async fn reencrypt_all_documents_with_progress(
        &mut self,
        progress_tx: Option<tokio::sync::mpsc::Sender<(usize, usize, u64, u64)>>,
    ) -> VPKResult<UploadResult> {
        let total_start = std::time::Instant::now();

        if self.state() != VPKState::Active {
            return Err(VPKError::Other(format!(
                "Cannot re-encrypt in state: {:?}",
                self.state()
            )));
        }

        // Clear index mappings
        let clear_client = self
            .database
            .pool()
            .get()
            .await
            .map_err(|e| VPKError::Other(format!("Pool error: {}", e)))?;
        clear_client
            .execute("TRUNCATE index_mapping CASCADE", &[])
            .await
            .map_err(|e| VPKError::Other(format!("Failed to clear index mappings: {}", e)))?;

        // Clear all Eve shards
        self.shard_manager.clear_all().await?;

        let documents = self.doc_queries.list(100000, 0).await?;
        if documents.is_empty() {
            return Ok(UploadResult {
                documents_inserted: 0,
                vectors_uploaded: 0,
                errors: vec!["No documents found to re-encrypt".to_string()],
                doc_ids: vec![],
                elapsed_us: total_start.elapsed().as_micros() as u64,
                embed_us: 0,
                encrypt_us: 0,
            });
        }

        let key_id = self.current_key_id.ok_or_else(|| {
            VPKError::Other("No active encryption key".to_string())
        })?;

        let scrambler = self
            .scrambler
            .as_ref()
            .ok_or_else(|| VPKError::Other("Index scrambler not initialized".to_string()))?;

        let mut vectors_per_shard: HashMap<i32, Vec<(i64, Array1<f64>)>> = HashMap::new();
        let mut errors = Vec::new();
        let documents_count = documents.len();
        let mut processed = 0usize;
        let mut total_embed_us: u64 = 0;
        let mut total_encrypt_us: u64 = 0;

        for doc in &documents {
            let t0 = std::time::Instant::now();
            let embedding = match self.embedding_model.embed(&doc.content).await {
                Ok(emb) => emb,
                Err(e) => {
                    errors.push(format!("Failed to embed doc {}: {}", doc.doc_id, e));
                    processed += 1;
                    continue;
                }
            };
            total_embed_us += t0.elapsed().as_micros() as u64;

            let t0 = std::time::Instant::now();
            let encrypted_vector = if let Some(pipeline) = &self.encryption_pipeline {
                match pipeline.encrypt_document(&embedding) {
                    Ok(enc) => enc,
                    Err(e) => {
                        errors.push(format!("Failed to encrypt doc {}: {}", doc.doc_id, e));
                        processed += 1;
                        continue;
                    }
                }
            } else if let Some(ds) = &self.dimensional_scrambling {
                match ds.encrypt(&embedding) {
                    Ok(enc) => enc,
                    Err(e) => {
                        errors.push(format!("Failed to encrypt doc {}: {}", doc.doc_id, e));
                        processed += 1;
                        continue;
                    }
                }
            } else {
                errors.push(format!("No encryption for doc {}", doc.doc_id));
                processed += 1;
                continue;
            };
            total_encrypt_us += t0.elapsed().as_micros() as u64;

            let scrambled_id = scrambler.scramble(doc.doc_id);
            let shard_id = self.shard_manager.assign_shard();

            if let Err(e) = self
                .index_mapping
                .store(doc.doc_id, scrambled_id, key_id, Some(shard_id))
                .await
            {
                errors.push(format!("Failed to store mapping for doc {}: {}", doc.doc_id, e));
                processed += 1;
                continue;
            }

            vectors_per_shard
                .entry(shard_id)
                .or_default()
                .push((scrambled_id, encrypted_vector));

            processed += 1;

            // Send progress every 10 documents
            if let Some(ref tx) = progress_tx {
                if processed.is_multiple_of(10) || processed == documents_count {
                    let _ = tx.send((processed, documents_count, total_embed_us, total_encrypt_us)).await;
                }
            }
        }

        let vectors_uploaded: usize = {
            let mut total = 0;
            for (shard_id, vectors) in vectors_per_shard {
                match self.shard_manager.upload_to_shard(shard_id, vectors).await {
                    Ok(count) => total += count,
                    Err(e) => errors.push(format!("Upload to shard {} failed: {}", shard_id, e)),
                }
            }
            total
        };

        // Send final progress
        if let Some(ref tx) = progress_tx {
            let _ = tx.send((documents_count, documents_count, total_embed_us, total_encrypt_us)).await;
        }

        Ok(UploadResult {
            documents_inserted: documents_count,
            vectors_uploaded,
            errors,
            doc_ids: vec![],
            elapsed_us: total_start.elapsed().as_micros() as u64,
            embed_us: total_embed_us,
            encrypt_us: total_encrypt_us,
        })
    }

    /// Re-encrypt all documents (convenience wrapper without progress)
    pub async fn reencrypt_all_documents(&mut self) -> VPKResult<UploadResult> {
        self.reencrypt_all_documents_with_progress(None).await
    }

    /// Get document by ID
    pub async fn get_document(&self, doc_id: i64) -> VPKResult<Option<Document>> {
        self.doc_queries.get(doc_id).await
    }

    /// Encrypt a query vector (for visualization in API handlers)
    pub fn encrypt_query_vector(&self, query_vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
        if let Some(pipeline) = &self.encryption_pipeline {
            pipeline.encrypt_query(query_vector)
        } else if let Some(ds) = &self.dimensional_scrambling {
            ds.encrypt(query_vector)
        } else {
            Err(VPKError::Other(
                "No encryption pipeline available".to_string(),
            ))
        }
    }

    /// Query for similar documents using encrypted multi-shard fan-out search
    ///
    /// Pipeline:
    /// 1. Embed query text
    /// 2. Encrypt query
    /// 3. Fan-out to all Eve shards in parallel
    /// 4. Merge top-K across shards
    /// 5. Reverse-map scrambled IDs → real doc IDs (preserving score order)
    /// 6. Fetch plaintext docs from Bob's DB
    pub async fn query(&self, query_text: &str, top_k: usize) -> VPKResult<QueryResult> {
        if self.state() != VPKState::Active {
            return Err(VPKError::Other(format!(
                "Cannot query in state: {:?}",
                self.state()
            )));
        }

        let t_total = std::time::Instant::now();

        let pipeline = self
            .encryption_pipeline
            .as_ref()
            .ok_or_else(|| VPKError::Other("Encryption pipeline not initialized".to_string()))?;

        // Step 1: Embed query
        let t0 = std::time::Instant::now();
        let query_embedding = self.embedding_model.embed(query_text).await?;
        let embed_us = t0.elapsed().as_micros() as u64;

        // Step 2: Encrypt query (includes CKKS SEAL encryption when active)
        let t0 = std::time::Instant::now();
        let encrypted_query = pipeline.encrypt_query(&query_embedding)?;
        if let Some(ckks) = &self.ckks_context {
            let ct = ckks.encrypt(&encrypted_query)?;
            ckks.cache_query_ct(ct);
        }
        let encrypt_us = t0.elapsed().as_micros() as u64;

        // Step 3+4: Fan-out to all shards, get merged top-K
        let t0 = std::time::Instant::now();
        let shard_results = self.shard_manager.search_all(&encrypted_query, top_k).await?;
        let search_us = t0.elapsed().as_micros() as u64;
        if let Some(ckks) = &self.ckks_context {
            ckks.clear_query_ct_cache();
        }

        if shard_results.is_empty() {
            return Ok(QueryResult {
                documents: vec![],
                scores: vec![],
                doc_ids: vec![],
                embed_us,
                encrypt_us,
                search_us,
                total_us: t_total.elapsed().as_micros() as u64,
            });
        }

        // Step 5: Reverse-map scrambled IDs → real doc IDs
        // Using a HashMap to preserve score ordering (not doc_id ordering)
        let scrambled_ids: Vec<i64> = shard_results.iter().map(|r| r.scrambled_id).collect();
        let id_map = self
            .index_mapping
            .get_originals_map(&scrambled_ids)
            .await?;

        // Step 6: Fetch docs in score order
        let mut documents = Vec::new();
        let mut scores = Vec::new();
        let mut result_doc_ids = Vec::new();

        for result in &shard_results {
            if let Some(&doc_id) = id_map.get(&result.scrambled_id) {
                match self.doc_queries.get(doc_id).await? {
                    Some(doc) => {
                        documents.push(doc);
                        scores.push(result.score);
                        result_doc_ids.push(doc_id);
                    }
                    None => {
                        tracing::warn!("Document {} not found in Bob's DB (stale index?)", doc_id);
                    }
                }
            }
        }

        // Audit log: SEARCH event (no query text, no Alice identity)
        let shard_ids_hit: Vec<i32> = shard_results.iter().map(|r| r.shard_id).collect();
        let _ = self
            .audit_queries
            .log(
                "SEARCH",
                None,
                Some(serde_json::json!({
                    "shards_queried": shard_ids_hit,
                    "results_returned": result_doc_ids.len(),
                    "top_k_requested": top_k,
                })),
                None,
            )
            .await;

        Ok(QueryResult {
            documents,
            scores,
            doc_ids: result_doc_ids,
            embed_us,
            encrypt_us,
            search_us,
            total_us: t_total.elapsed().as_micros() as u64,
        })
    }

    /// Get full system health status (for MCP get_health tool)
    pub async fn health_status(&self) -> VPKResult<SystemHealthStatus> {
        let doc_count = self.doc_queries.count().await? as usize;
        let total_vectors = self.shard_manager.total_count();
        let shards = self.shard_manager.health_check_all().await;
        let shard_count = self.shard_manager.shard_count();

        Ok(SystemHealthStatus {
            state: format!("{:?}", self.state()),
            documents: doc_count,
            total_vectors,
            shards,
            embedding_dimension: self.config.crypto.vector_dim,
            shard_count,
            active_key_id: self.current_key_id.map(|id| id.to_string()),
        })
    }

    /// Get audit log entries (firewall-style — no plaintext content, no Alice identity)
    pub async fn get_audit_log(&self, limit: i64) -> VPKResult<Vec<JsonValue>> {
        self.audit_queries.get_recent(limit).await
    }

    /// Get audit log entries since a given timestamp
    pub async fn get_audit_log_since(
        &self,
        since: chrono::DateTime<chrono::Utc>,
        limit: i64,
    ) -> VPKResult<Vec<JsonValue>> {
        self.audit_queries.get_since(since, limit).await
    }

    /// Get system statistics
    pub async fn stats(&self) -> VPKResult<SystemStats> {
        let doc_count = self.doc_queries.count().await?;
        let vector_count = self.shard_manager.total_count();
        let (cache_forward, _) = self.index_mapping.cache_stats();

        Ok(SystemStats {
            state: self.state(),
            documents: doc_count as usize,
            vectors: vector_count,
            cache_entries: cache_forward,
            active_key_id: self.current_key_id,
        })
    }

    // ------------------------------------------------------------------
    // Tenant management
    // ------------------------------------------------------------------

    /// Register a new tenant (called by Eve-side API when Bob connects).
    pub async fn create_tenant(
        &self,
        peer_id: &str,
        capacity_limit_mb: i64,
        display_name: Option<&str>,
    ) -> VPKResult<TenantRecord> {
        self.tenant_queries.create(peer_id, capacity_limit_mb, display_name).await
    }

    /// Return all active tenants.
    pub async fn list_tenants(&self) -> VPKResult<Vec<TenantRecord>> {
        self.tenant_queries.list_active().await
    }

    /// Return a single tenant by UUID.
    pub async fn get_tenant(&self, tenant_id: Uuid) -> VPKResult<Option<TenantRecord>> {
        self.tenant_queries.get(tenant_id).await
    }

    /// Soft-delete a tenant (sets status = 'deleted').
    pub async fn delete_tenant(&self, tenant_id: Uuid) -> VPKResult<()> {
        self.tenant_queries.delete(tenant_id).await
    }

    /// Count documents belonging to a tenant.
    pub async fn tenant_document_count(&self, tenant_id: Uuid) -> VPKResult<i64> {
        self.tenant_queries.document_count(tenant_id).await
    }

    /// Count all active tenants (for health endpoint).
    pub async fn tenant_count(&self) -> VPKResult<usize> {
        Ok(self.tenant_queries.list_active().await?.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VPKConfig;

    #[tokio::test]
    #[ignore] // Requires running PostgreSQL database
    async fn test_vpk_state_machine() {
        let config = VPKConfig::default_test();
        let vpk = VPK::new(config).await.unwrap();
        assert_eq!(vpk.state(), VPKState::Uninitialized);
    }

    #[test]
    fn test_state_transitions() {
        use std::sync::{Arc, RwLock};

        let state = Arc::new(RwLock::new(VPKState::Uninitialized));
        {
            let mut s = state.write().unwrap();
            *s = VPKState::Initializing;
        }
        assert_eq!(*state.read().unwrap(), VPKState::Initializing);
        {
            let mut s = state.write().unwrap();
            *s = VPKState::Active;
        }
        assert_eq!(*state.read().unwrap(), VPKState::Active);
        {
            let mut s = state.write().unwrap();
            *s = VPKState::Degraded;
        }
        assert_eq!(*state.read().unwrap(), VPKState::Degraded);
    }

    /// End-to-end integration test with multi-shard
    #[tokio::test]
    #[ignore] // Requires running PostgreSQL database
    async fn test_end_to_end_encrypted_search() {
        let config = VPKConfig::default_test();
        let mut vpk = VPK::new(config).await.unwrap();

        vpk.initialize().await.unwrap();
        assert_eq!(vpk.state(), VPKState::Active);

        let documents = vec![
            "patient with lupus symptoms".to_string(),
            "diabetes treatment plan".to_string(),
            "lupus diagnosis and care".to_string(),
            "heart disease prevention".to_string(),
        ];

        let upload_result = vpk.upload_documents(documents.clone()).await.unwrap();
        assert_eq!(upload_result.documents_inserted, 4);
        assert_eq!(upload_result.vectors_uploaded, 4);
        assert!(upload_result.errors.is_empty());

        // Verify distribution across shards (3 shards, 4 docs → at least 2 shards used)
        assert!(vpk.shard_manager.total_count() == 4);

        let query_result = vpk.query("lupus symptoms", 3).await.unwrap();
        assert_eq!(query_result.documents.len(), 3);

        // Scores should be in descending order
        for i in 1..query_result.scores.len() {
            assert!(query_result.scores[i - 1] >= query_result.scores[i]);
        }

        // Health status should show active shards
        let health = vpk.health_status().await.unwrap();
        assert_eq!(health.state, "Active");
        assert_eq!(health.shard_count, 3);

        println!("Multi-shard end-to-end test passed!");
        println!("  Uploaded {} documents across {} shards", health.documents, health.shard_count);
    }

    #[tokio::test]
    #[ignore]
    async fn test_single_document_upload_via_mcp_method() {
        let config = VPKConfig::default_test();
        let mut vpk = VPK::new(config).await.unwrap();
        vpk.initialize().await.unwrap();

        let doc_id = vpk
            .upload_document(
                "test document content".to_string(),
                Some(serde_json::json!({"source": "test"})),
            )
            .await
            .unwrap();

        assert!(doc_id > 0);
        let doc = vpk.get_document(doc_id).await.unwrap().unwrap();
        assert_eq!(doc.content, "test document content");
    }
}
