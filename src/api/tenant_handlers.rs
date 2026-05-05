//! Tenant-scoped HTTP API handlers (Eve node registration, per-tenant upload/search).
//!
//! Routes:
//!   POST   /api/tenants                       → create_tenant
//!   GET    /api/tenants                       → list_tenants
//!   GET    /api/tenants/:id                   → get_tenant
//!   DELETE /api/tenants/:id                   → delete_tenant
//!   POST   /api/tenants/:id/upload            → upload_for_tenant
//!   POST   /api/tenants/:id/search            → search_for_tenant

use crate::database::queries::TenantRecord;
use super::handlers::{AppState, SearchResultItem};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

/// Body for `POST /api/tenants`
#[derive(Debug, Deserialize)]
pub struct CreateTenantRequest {
    /// Bob's libp2p peer ID (base58).
    pub peer_id: String,
    /// Maximum storage this tenant may use on Eve, in MB.
    #[serde(default = "default_capacity_mb")]
    pub capacity_limit_mb: i64,
    /// Human-readable label for this tenant (optional).
    pub display_name: Option<String>,
}

fn default_capacity_mb() -> i64 {
    1024
}

#[derive(Debug, Serialize)]
pub struct TenantResponse {
    pub success: bool,
    pub tenant: TenantRecord,
}

#[derive(Debug, Serialize)]
pub struct TenantListResponse {
    pub success: bool,
    pub tenants: Vec<TenantRecord>,
    pub count: usize,
}

#[derive(Debug, Serialize)]
pub struct TenantDetailResponse {
    pub success: bool,
    pub tenant: TenantRecord,
    pub document_count: i64,
}

#[derive(Debug, Serialize)]
pub struct DeleteTenantResponse {
    pub success: bool,
    pub message: String,
}

/// Body for `POST /api/tenants/:id/upload`
#[derive(Debug, Deserialize)]
pub struct TenantUploadRequest {
    pub documents: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct TenantUploadResponse {
    pub success: bool,
    pub documents_inserted: usize,
    pub vectors_uploaded: usize,
    pub errors: Vec<String>,
}

/// Body for `POST /api/tenants/:id/search`
#[derive(Debug, Deserialize)]
pub struct TenantSearchRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize {
    5
}

#[derive(Debug, Serialize)]
pub struct TenantSearchResponse {
    pub success: bool,
    pub results: Vec<SearchResultItem>,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /api/tenants — register a new Bob node as a tenant.
pub async fn create_tenant(
    State(vpk): State<AppState>,
    Json(req): Json<CreateTenantRequest>,
) -> Response {
    let vpk = vpk.read().await;

    match vpk
        .create_tenant(&req.peer_id, req.capacity_limit_mb, req.display_name.as_deref())
        .await
    {
        Ok(tenant) => (StatusCode::CREATED, Json(TenantResponse { success: true, tenant }))
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "success": false, "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// GET /api/tenants — list all active tenants.
pub async fn list_tenants(State(vpk): State<AppState>) -> Response {
    let vpk = vpk.read().await;

    match vpk.list_tenants().await {
        Ok(tenants) => {
            let count = tenants.len();
            (StatusCode::OK, Json(TenantListResponse { success: true, tenants, count }))
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "success": false, "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// GET /api/tenants/:id — fetch a single tenant + its document count.
pub async fn get_tenant(State(vpk): State<AppState>, Path(id): Path<Uuid>) -> Response {
    let vpk = vpk.read().await;

    let tenant = match vpk.get_tenant(id).await {
        Ok(Some(t)) => t,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "success": false, "error": "tenant not found" })),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "success": false, "error": e.to_string() })),
            )
                .into_response();
        }
    };

    let document_count = vpk.tenant_document_count(id).await.unwrap_or(0);

    (
        StatusCode::OK,
        Json(TenantDetailResponse { success: true, tenant, document_count }),
    )
        .into_response()
}

/// DELETE /api/tenants/:id — soft-delete a tenant.
pub async fn delete_tenant(State(vpk): State<AppState>, Path(id): Path<Uuid>) -> Response {
    let vpk = vpk.read().await;

    match vpk.delete_tenant(id).await {
        Ok(()) => (
            StatusCode::OK,
            Json(DeleteTenantResponse {
                success: true,
                message: format!("Tenant {} deleted", id),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "success": false, "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// POST /api/tenants/:id/upload — upload documents scoped to a specific tenant.
pub async fn upload_for_tenant(
    State(vpk): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<TenantUploadRequest>,
) -> Response {
    // Verify tenant exists before writing
    {
        let vpk = vpk.read().await;
        match vpk.get_tenant(id).await {
            Ok(None) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({ "success": false, "error": "tenant not found" })),
                )
                    .into_response();
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "success": false, "error": e.to_string() })),
                )
                    .into_response();
            }
            Ok(Some(_)) => {}
        }
    }

    let mut vpk = vpk.write().await;
    let mut documents_inserted = 0usize;
    let mut vectors_uploaded = 0usize;
    let mut errors = Vec::new();

    for text in req.documents {
        match vpk.upload_document_for_tenant(text, None, id).await {
            Ok(_) => {
                documents_inserted += 1;
                vectors_uploaded += 1;
            }
            Err(e) => errors.push(e.to_string()),
        }
    }

    (
        StatusCode::OK,
        Json(TenantUploadResponse {
            success: errors.is_empty(),
            documents_inserted,
            vectors_uploaded,
            errors,
        }),
    )
        .into_response()
}

/// POST /api/tenants/:id/search — search within a specific tenant's documents.
pub async fn search_for_tenant(
    State(vpk): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<TenantSearchRequest>,
) -> Response {
    let vpk = vpk.read().await;

    match vpk.query_for_tenant(&req.query, req.top_k, id).await {
        Ok(result) => {
            let results: Vec<SearchResultItem> = result
                .documents
                .iter()
                .zip(result.scores.iter())
                .zip(result.doc_ids.iter())
                .map(|((doc, score), doc_id)| SearchResultItem {
                    doc_id: *doc_id,
                    content: doc.content.clone(),
                    score: *score,
                })
                .collect();

            (StatusCode::OK, Json(TenantSearchResponse { success: true, results })).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "success": false, "error": e.to_string() })),
        )
            .into_response(),
    }
}
