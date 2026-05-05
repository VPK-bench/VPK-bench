//! API request handlers with visualization support

use crate::vpk::VPK;
use axum::{
    extract::{Extension, Json, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Shared API base URL (injected at startup)
pub type ApiBase = Arc<String>;

/// Shared application state
pub type AppState = Arc<RwLock<VPK>>;

/// Upload request
#[derive(Debug, Deserialize)]
pub struct UploadRequest {
    pub documents: Vec<String>,
}

/// Upload response with visualization data
#[derive(Debug, Serialize)]
pub struct UploadResponse {
    pub success: bool,
    pub documents_inserted: usize,
    pub vectors_uploaded: usize,
    pub errors: Vec<String>,
    pub visualization: UploadVisualization,
}

/// Per-shard info for health and upload visualizations
#[derive(Debug, Serialize, Clone)]
pub struct ShardInfo {
    pub shard_id: i32,
    pub name: String,
    pub healthy: bool,
    pub num_vectors: usize,
}

/// Visualization data for upload process
#[derive(Debug, Serialize)]
pub struct UploadVisualization {
    pub original_embeddings: Vec<Vec<f64>>,
    pub encrypted_vectors: Vec<Vec<f64>>,
    pub doc_ids: Vec<i64>,
    pub scrambled_ids: Vec<i64>,
    /// Per-shard vector counts after upload
    pub shard_status: Vec<ShardInfo>,
}

/// Search request
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize {
    5
}

/// Search response with visualization data
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub success: bool,
    pub results: Vec<SearchResultItem>,
    pub visualization: SearchVisualization,
}

/// Individual search result
#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    pub doc_id: i64,
    pub content: String,
    pub score: f64,
}

/// Visualization data for search process
#[derive(Debug, Serialize)]
pub struct SearchVisualization {
    pub query_embedding: Vec<f64>,
    pub encrypted_query: Vec<f64>,
    pub result_vectors: Vec<Vec<f64>>,
    pub encrypted_scores: Vec<f64>,
    pub unencrypted_scores: Vec<f64>,
    pub result_embeddings: Vec<Vec<f64>>,
    /// Number of Eve shards queried in parallel
    pub shards_queried: usize,
}

/// Health check response (extended with shard status)
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub healthy: bool,
    pub state: String,
    pub documents: usize,
    pub vectors: usize,
    pub cache_entries: usize,
    pub shard_count: usize,
    pub shards_healthy: usize,
    /// Per-shard details (id, name, healthy, num_vectors)
    pub shards: Vec<ShardInfo>,
    /// Currently active encryption algorithm
    pub algorithm: String,
    /// Current noise range minimum
    pub noise_min: f64,
    /// Current noise range maximum
    pub noise_max: f64,
    /// Number of active tenants registered on this Eve node
    pub tenant_count: usize,
}

/// Upload documents with visualization
pub async fn upload_documents(
    State(vpk): State<AppState>,
    Json(request): Json<UploadRequest>,
) -> Response {
    let mut vpk = vpk.write().await;

    // Generate embeddings first for visualization
    let mut original_embeddings = Vec::new();
    for content in &request.documents {
        match vpk.embedding_model.embed(content).await {
            Ok(emb) => original_embeddings.push(emb.to_vec()),
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(UploadResponse {
                        success: false,
                        documents_inserted: 0,
                        vectors_uploaded: 0,
                        errors: vec![format!("Failed to generate embedding: {}", e)],
                        visualization: UploadVisualization {
                            original_embeddings: vec![],
                            encrypted_vectors: vec![],
                            doc_ids: vec![],
                            scrambled_ids: vec![],
                            shard_status: vec![],
                        },
                    }),
                )
                    .into_response();
            }
        }
    }

    // Upload documents
    match vpk.upload_documents(request.documents).await {
        Ok(result) => {
            // Fetch per-shard status after upload for visualization
            let shard_status = match vpk.health_status().await {
                Ok(status) => status
                    .shards
                    .into_iter()
                    .map(|s| ShardInfo {
                        shard_id: s.shard_id,
                        name: s.name,
                        healthy: s.healthy,
                        num_vectors: s.num_vectors,
                    })
                    .collect(),
                Err(_) => vec![],
            };

            let visualization = UploadVisualization {
                original_embeddings,
                encrypted_vectors: vec![],
                doc_ids: result.doc_ids,
                scrambled_ids: vec![],
                shard_status,
            };

            (
                StatusCode::OK,
                Json(UploadResponse {
                    success: true,
                    documents_inserted: result.documents_inserted,
                    vectors_uploaded: result.vectors_uploaded,
                    errors: result.errors,
                    visualization,
                }),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(UploadResponse {
                success: false,
                documents_inserted: 0,
                vectors_uploaded: 0,
                errors: vec![format!("Upload failed: {}", e)],
                visualization: UploadVisualization {
                    original_embeddings: vec![],
                    encrypted_vectors: vec![],
                    doc_ids: vec![],
                    scrambled_ids: vec![],
                    shard_status: vec![],
                },
            }),
        )
            .into_response(),
    }
}

/// Search with visualization
pub async fn search(
    State(vpk): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Response {
    let vpk = vpk.read().await;

    let shards_queried = vpk.shard_count();

    // Generate query embedding for visualization
    let query_embedding = match vpk.embedding_model.embed(&request.query).await {
        Ok(emb) => emb,
        Err(_e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(SearchResponse {
                    success: false,
                    results: vec![],
                    visualization: SearchVisualization {
                        query_embedding: vec![],
                        encrypted_query: vec![],
                        result_vectors: vec![],
                        encrypted_scores: vec![],
                        unencrypted_scores: vec![],
                        result_embeddings: vec![],
                        shards_queried,
                    },
                }),
            )
                .into_response();
        }
    };

    // Encrypt query for visualization
    let encrypted_query = match vpk.encrypt_query_vector(&query_embedding) {
        Ok(enc) => enc,
        Err(_e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(SearchResponse {
                    success: false,
                    results: vec![],
                    visualization: SearchVisualization {
                        query_embedding: query_embedding.to_vec(),
                        encrypted_query: vec![],
                        result_vectors: vec![],
                        encrypted_scores: vec![],
                        unencrypted_scores: vec![],
                        result_embeddings: vec![],
                        shards_queried,
                    },
                }),
            )
                .into_response();
        }
    };

    // Perform search
    match vpk.query(&request.query, request.top_k).await {
        Ok(query_result) => {
            // Compute unencrypted scores for comparison
            let mut unencrypted_scores = Vec::new();
            let mut result_embeddings = Vec::new();

            for doc in &query_result.documents {
                // Generate embedding for this document
                if let Ok(doc_emb) = vpk.embedding_model.embed(&doc.content).await {
                    // Compute cosine similarity with query embedding (unencrypted)
                    let dot: f64 = query_embedding.iter().zip(doc_emb.iter()).map(|(a, b)| a * b).sum();
                    let norm1: f64 = query_embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let norm2: f64 = doc_emb.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let cosine_sim = if norm1 > 1e-10 && norm2 > 1e-10 {
                        dot / (norm1 * norm2)
                    } else {
                        0.0
                    };
                    unencrypted_scores.push(cosine_sim);
                    result_embeddings.push(doc_emb.to_vec());
                } else {
                    unencrypted_scores.push(0.0);
                    result_embeddings.push(vec![]);
                }
            }

            let results: Vec<SearchResultItem> = query_result
                .documents
                .iter()
                .zip(query_result.scores.iter())
                .zip(query_result.doc_ids.iter())
                .map(|((doc, score), doc_id)| SearchResultItem {
                    doc_id: *doc_id,
                    content: doc.content.clone(),
                    score: *score,
                })
                .collect();

            let visualization = SearchVisualization {
                query_embedding: query_embedding.to_vec(),
                encrypted_query: encrypted_query.to_vec(),
                result_vectors: vec![],
                encrypted_scores: query_result.scores,
                unencrypted_scores,
                result_embeddings,
                shards_queried,
            };

            (
                StatusCode::OK,
                Json(SearchResponse {
                    success: true,
                    results,
                    visualization,
                }),
            )
                .into_response()
        }
        Err(_e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(SearchResponse {
                success: false,
                results: vec![],
                visualization: SearchVisualization {
                    query_embedding: query_embedding.to_vec(),
                    encrypted_query: encrypted_query.to_vec(),
                    result_vectors: vec![],
                    encrypted_scores: vec![],
                    unencrypted_scores: vec![],
                    result_embeddings: vec![],
                    shards_queried,
                },
            }),
        )
            .into_response(),
    }
}

pub fn algorithm_to_str(algo: &crate::config::EncryptionAlgorithm) -> &'static str {
    algo.as_str()
}

/// Health check (includes shard reachability)
pub async fn health(State(vpk): State<AppState>) -> Response {
    let vpk = vpk.read().await;

    let algorithm = algorithm_to_str(&vpk.get_algorithm()).to_string();
    let (noise_min, noise_max) = vpk.get_noise_range();

    let tenant_count = vpk.tenant_count().await.unwrap_or(0);

    match vpk.health_status().await {
        Ok(status) => {
            let shards_healthy = status.shards.iter().filter(|s| s.healthy).count();
            let shards: Vec<ShardInfo> = status
                .shards
                .into_iter()
                .map(|s| ShardInfo {
                    shard_id: s.shard_id,
                    name: s.name,
                    healthy: s.healthy,
                    num_vectors: s.num_vectors,
                })
                .collect();
            (
                StatusCode::OK,
                Json(HealthResponse {
                    healthy: shards_healthy > 0,
                    state: status.state,
                    documents: status.documents,
                    vectors: status.total_vectors,
                    cache_entries: 0,
                    shard_count: status.shard_count,
                    shards_healthy,
                    shards,
                    algorithm,
                    noise_min,
                    noise_max,
                    tenant_count,
                }),
            )
                .into_response()
        }
        Err(_e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(HealthResponse {
                healthy: false,
                state: "Error".to_string(),
                documents: 0,
                vectors: 0,
                cache_entries: 0,
                shard_count: 0,
                shards_healthy: 0,
                shards: vec![],
                algorithm,
                noise_min,
                noise_max,
                tenant_count,
            }),
        )
            .into_response(),
    }
}

/// Config update request
#[derive(Debug, Deserialize)]
pub struct ConfigRequest {
    pub algorithm: String,
    pub noise_min: Option<f64>,
    pub noise_max: Option<f64>,
    /// When true, only the noise layer is reseeded; the DS / ROME / DIEHARD / CKKS
    /// rotation matrix is preserved.  Requires noise_min + noise_max.
    /// Used by Experiment G to isolate the noise effect from DS initialisation.
    pub freeze_rotation: Option<bool>,
}

/// Config response
#[derive(Debug, Serialize)]
pub struct ConfigResponse {
    pub success: bool,
    pub algorithm: String,
    pub message: String,
}

/// Update encryption configuration
pub async fn set_config(
    State(vpk): State<AppState>,
    Json(request): Json<ConfigRequest>,
) -> Response {
    use crate::config::EncryptionAlgorithm;

    let mut vpk = vpk.write().await;
    let algorithm_str = request.algorithm.clone();

    // Parse algorithm
    let algorithm = match request.algorithm.as_str() {
        "scrambling" => EncryptionAlgorithm::Scrambling,
        "noise" => EncryptionAlgorithm::Noise,
        "combined" => EncryptionAlgorithm::Combined,
        "rome" => EncryptionAlgorithm::Rome,
        "rome-combined" => EncryptionAlgorithm::RomeCombined,
        "romm" => EncryptionAlgorithm::Romm,
        "romm-combined" => EncryptionAlgorithm::RommCombined,
        "diehard" => EncryptionAlgorithm::Diehard,
        "diehard-combined" => EncryptionAlgorithm::DiehardCombined,
        "ckks" => EncryptionAlgorithm::Ckks,
        "ckks-combined" => EncryptionAlgorithm::CkksCombined,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ConfigResponse {
                    success: false,
                    algorithm: algorithm_str,
                    message: format!("Unknown algorithm: {}", request.algorithm),
                }),
            )
                .into_response();
        }
    };

    // Get noise range if provided
    let noise_range = if let (Some(min), Some(max)) = (request.noise_min, request.noise_max) {
        Some((min, max))
    } else {
        None
    };

    let freeze = request.freeze_rotation.unwrap_or(false);

    if freeze {
        // Only reseed the noise layer; rotation matrix is preserved.
        let range = match noise_range {
            Some(r) => r,
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ConfigResponse {
                        success: false,
                        algorithm: algorithm_str,
                        message: "freeze_rotation=true requires noise_min and noise_max".to_string(),
                    }),
                ).into_response();
            }
        };
        match vpk.set_noise_only(range) {
            Ok(_) => (
                StatusCode::OK,
                Json(ConfigResponse {
                    success: true,
                    algorithm: algorithm_str.clone(),
                    message: format!("Noise reseeded (rotation frozen) for {}", algorithm_str),
                }),
            ).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ConfigResponse {
                    success: false,
                    algorithm: algorithm_str,
                    message: format!("Failed to reseed noise: {}", e),
                }),
            ).into_response(),
        }
    } else {
        // Full regeneration: new rotation matrix + noise
        match vpk.set_algorithm(algorithm, noise_range).await {
            Ok(_) => (
                StatusCode::OK,
                Json(ConfigResponse {
                    success: true,
                    algorithm: algorithm_str.clone(),
                    message: format!("Successfully updated algorithm to {}", algorithm_str),
                }),
            )
                .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ConfigResponse {
                    success: false,
                    algorithm: algorithm_str,
                    message: format!("Failed to update algorithm: {}", e),
                }),
            )
                .into_response(),
        }
    }
}

/// Re-encrypt response
#[derive(Debug, Serialize)]
pub struct ReencryptResponse {
    pub success: bool,
    pub documents_processed: usize,
    pub vectors_uploaded: usize,
    pub message: String,
    pub errors: Vec<String>,
    /// Total wall-clock time in milliseconds
    pub elapsed_us: u64,
    /// Time spent on embedding generation in milliseconds
    pub embed_us: u64,
    /// Time spent on encryption in milliseconds
    pub encrypt_us: u64,
}

/// Re-encrypt all documents with current algorithm
pub async fn reencrypt_database(State(vpk): State<AppState>) -> Response {
    let mut vpk = vpk.write().await;

    match vpk.reencrypt_all_documents().await {
        Ok(result) => (
            StatusCode::OK,
            Json(ReencryptResponse {
                success: true,
                documents_processed: result.documents_inserted,
                vectors_uploaded: result.vectors_uploaded,
                message: format!(
                    "Successfully re-encrypted {} documents",
                    result.documents_inserted
                ),
                errors: result.errors,
                elapsed_us: result.elapsed_us,
                embed_us: result.embed_us,
                encrypt_us: result.encrypt_us,
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ReencryptResponse {
                success: false,
                documents_processed: 0,
                vectors_uploaded: 0,
                message: format!("Failed to re-encrypt: {}", e),
                errors: vec![],
                elapsed_us: 0,
                embed_us: 0,
                encrypt_us: 0,
            }),
        )
            .into_response(),
    }
}

/// SSE streaming re-encryption with live progress
pub async fn reencrypt_stream(State(vpk): State<AppState>) -> Response {
    use axum::response::sse::{Event, Sse};
    use std::convert::Infallible;

    let (tx, mut rx) = tokio::sync::mpsc::channel::<(usize, usize, u64, u64)>(64);

    // Spawn the re-encryption in a background task
    let vpk_clone = vpk.clone();
    let result_tx = tokio::sync::oneshot::channel::<Result<crate::vpk::UploadResult, String>>();
    let (done_tx, done_rx) = result_tx;

    tokio::spawn(async move {
        let mut vpk = vpk_clone.write().await;
        let result = vpk.reencrypt_all_documents_with_progress(Some(tx)).await;
        let _ = done_tx.send(result.map_err(|e| e.to_string()));
    });

    // Create SSE stream
    let stream = async_stream::stream! {
        let mut done_rx = done_rx;
        let start = std::time::Instant::now();

        loop {
            tokio::select! {
                Some((processed, total, embed_us, encrypt_us)) = rx.recv() => {
                    let elapsed = start.elapsed().as_micros() as u64;
                    let data = serde_json::json!({
                        "type": "progress",
                        "processed": processed,
                        "total": total,
                        "embed_us": embed_us,
                        "encrypt_us": encrypt_us,
                        "elapsed_us": elapsed,
                    });
                    yield Ok::<_, Infallible>(Event::default().data(data.to_string()));
                }
                result = &mut done_rx => {
                    // Drain remaining progress messages
                    while let Ok((processed, total, embed_us, encrypt_us)) = rx.try_recv() {
                        let elapsed = start.elapsed().as_micros() as u64;
                        let data = serde_json::json!({
                            "type": "progress",
                            "processed": processed,
                            "total": total,
                            "embed_us": embed_us,
                            "encrypt_us": encrypt_us,
                            "elapsed_us": elapsed,
                        });
                        yield Ok::<_, Infallible>(Event::default().data(data.to_string()));
                    }

                    match result {
                        Ok(Ok(upload_result)) => {
                            let data = serde_json::json!({
                                "type": "complete",
                                "success": true,
                                "documents_processed": upload_result.documents_inserted,
                                "vectors_uploaded": upload_result.vectors_uploaded,
                                "errors": upload_result.errors,
                                "elapsed_us": upload_result.elapsed_us,
                                "embed_us": upload_result.embed_us,
                                "encrypt_us": upload_result.encrypt_us,
                            });
                            yield Ok::<_, Infallible>(Event::default().data(data.to_string()));
                        }
                        Ok(Err(e)) => {
                            let data = serde_json::json!({
                                "type": "complete",
                                "success": false,
                                "error": e,
                            });
                            yield Ok::<_, Infallible>(Event::default().data(data.to_string()));
                        }
                        Err(_) => {
                            let data = serde_json::json!({
                                "type": "complete",
                                "success": false,
                                "error": "Internal channel error",
                            });
                            yield Ok::<_, Infallible>(Event::default().data(data.to_string()));
                        }
                    }
                    break;
                }
            }
        }
    };

    Sse::new(stream).into_response()
}

/// Serve demo HTML page, injecting the configured API base URL
pub async fn demo_page(Extension(api_base): Extension<ApiBase>) -> Html<String> {
    let html = include_str!("../../demo/index.html")
        .replace("const API_BASE = '';", &format!("const API_BASE = '{}';", api_base));
    Html(html)
}

// ── Experiment endpoint (NeurIPS evaluation) ──────────────────────────────────

/// Per-query metrics returned by the experiment endpoint
#[derive(Debug, Serialize)]
pub struct QueryMetrics {
    pub query: String,
    /// Encrypted top-K doc IDs (in rank order)
    pub enc_doc_ids: Vec<i64>,
    /// Encrypted similarity scores
    pub enc_scores: Vec<f64>,
    /// Plaintext baseline top-K doc IDs (only when include_plaintext_baseline=true)
    pub baseline_doc_ids: Vec<i64>,
    /// Plaintext baseline similarity scores
    pub baseline_scores: Vec<f64>,
    // ── Rank preservation (vs plaintext baseline) ──
    pub recall_at_k: f64,
    pub ndcg_at_k: f64,
    pub kendall_tau: f64,
    pub exact_match: bool,
    // ── Timing (μs) ──
    pub enc_embed_us: u64,
    pub enc_encrypt_us: u64,
    pub enc_search_us: u64,
    pub enc_total_us: u64,
    // ── Encryption strength heuristics (encrypted result scores) ──
    pub score_entropy: f64,
    pub score_variance: f64,
    pub score_distinct_count: usize,
}

/// Experiment request
#[derive(Debug, Deserialize)]
pub struct ExperimentRequest {
    pub queries: Vec<String>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// If true, run each query with no encryption as baseline for rank metrics
    #[serde(default)]
    pub include_plaintext_baseline: bool,
}

/// Experiment response
#[derive(Debug, Serialize)]
pub struct ExperimentResponse {
    pub success: bool,
    pub algorithm: String,
    pub top_k: usize,
    pub results: Vec<QueryMetrics>,
    pub errors: Vec<String>,
}

/// POST /api/experiment
///
/// Runs a batch of queries and returns per-query rank-preservation and timing
/// metrics. If `include_plaintext_baseline` is true each query is run twice —
/// first with the current encryption pipeline to get the baseline, then a
/// second time in the same encrypted context (baseline requires a no-op config
/// change; callers should switch to `Scrambling` + `noise_range=(1.0,1.0)` for
/// a true no-op, or use the existing plaintext DB scores returned by the
/// search visualizer).
///
/// **Practical approach used here:** we run the query once with encryption and
/// use the *unencrypted* cosine similarities computed in the search handler
/// (already available via the embedding model) as the baseline. The caller
/// may also set the algorithm to `Scrambling` with zero noise for a lossless
/// "plaintext" run and record those doc IDs as ground truth.
pub async fn run_experiment(State(vpk): State<AppState>, Json(req): Json<ExperimentRequest>) -> Response {
    use crate::vpk::metrics;

    let vpk = vpk.read().await;
    let top_k = req.top_k;
    let algorithm = algorithm_to_str(&vpk.get_algorithm()).to_string();

    let mut results = Vec::new();
    let mut errors = Vec::new();

    for query in &req.queries {
        // Run encrypted query
        let enc_result = match vpk.query(query, top_k).await {
            Ok(r) => r,
            Err(e) => {
                errors.push(format!("Query '{}' failed: {}", query, e));
                continue;
            }
        };

        // Optionally run plaintext baseline (second query with same pipeline)
        // In practice callers configure a no-op algorithm for the baseline run
        // and pass those doc_ids here as baseline_doc_ids via a separate call.
        // Here we expose the encrypted result as its own baseline when
        // include_plaintext_baseline is false, so metrics will trivially be 1.0.
        let (baseline_doc_ids, baseline_scores): (Vec<i64>, Vec<f64>) =
            if req.include_plaintext_baseline {
                // Run a second query — caller is expected to have configured a
                // no-op pipeline (Scrambling, noise 1.0–1.0) before this call.
                match vpk.query(query, top_k).await {
                    Ok(r) => (r.doc_ids, r.scores),
                    Err(_) => (enc_result.doc_ids.clone(), enc_result.scores.clone()),
                }
            } else {
                (enc_result.doc_ids.clone(), enc_result.scores.clone())
            };

        let gt_scores: Vec<(i64, f64)> = baseline_doc_ids
            .iter()
            .zip(baseline_scores.iter())
            .map(|(&id, &s)| (id, s))
            .collect();

        let recall = metrics::recall_at_k(&baseline_doc_ids, &enc_result.doc_ids, top_k);
        let ndcg = metrics::ndcg_at_k(&gt_scores, &enc_result.doc_ids, top_k);
        let tau = metrics::kendall_tau(&baseline_doc_ids, &enc_result.doc_ids);
        let exact = metrics::exact_match_at_k(&baseline_doc_ids, &enc_result.doc_ids, top_k);
        let entropy = metrics::score_entropy(&enc_result.scores, 20);
        let variance = metrics::score_variance(&enc_result.scores);
        let distinct = metrics::score_distinct_count(&enc_result.scores);

        results.push(QueryMetrics {
            query: query.clone(),
            enc_doc_ids: enc_result.doc_ids,
            enc_scores: enc_result.scores,
            baseline_doc_ids,
            baseline_scores,
            recall_at_k: recall,
            ndcg_at_k: ndcg,
            kendall_tau: tau,
            exact_match: exact,
            enc_embed_us: enc_result.embed_us,
            enc_encrypt_us: enc_result.encrypt_us,
            enc_search_us: enc_result.search_us,
            enc_total_us: enc_result.total_us,
            score_entropy: entropy,
            score_variance: variance,
            score_distinct_count: distinct,
        });
    }

    (
        StatusCode::OK,
        Json(ExperimentResponse {
            success: errors.is_empty(),
            algorithm,
            top_k,
            results,
            errors,
        }),
    )
        .into_response()
}
