//! REST API server and handlers

pub mod handlers;
pub mod openai;
pub mod tenant_handlers;

use crate::error::VPKResult;
use crate::vpk::VPK;
use axum::{
    routing::{delete, get, post},
    Extension, Router,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

/// API server configuration
#[derive(Debug, Clone)]
pub struct APIConfig {
    pub bind_address: String,
    pub port: u16,
    /// Base URL the frontend uses to reach the API (empty = relative URLs)
    pub api_base: String,
}

impl Default for APIConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1".to_string(),
            port: 3000,
            api_base: String::new(),
        }
    }
}

/// Create the API router with all routes
pub fn create_router(vpk: VPK, api_base: String) -> Router {
    let state = Arc::new(RwLock::new(vpk));
    let api_base = Arc::new(api_base);

    // CORS layer to allow web frontend
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/", get(handlers::demo_page))
        .route("/api/upload", post(handlers::upload_documents))
        .route("/api/search", post(handlers::search))
        .route("/api/health", get(handlers::health))
        .route("/api/config", post(handlers::set_config))
        .route("/api/reencrypt", post(handlers::reencrypt_database))
        .route("/api/reencrypt/stream", post(handlers::reencrypt_stream))
        .route("/api/experiment", post(handlers::run_experiment))
        // Tenant management (Eve node registration + scoped upload/search)
        .route("/api/tenants", post(tenant_handlers::create_tenant))
        .route("/api/tenants", get(tenant_handlers::list_tenants))
        .route("/api/tenants/:id", get(tenant_handlers::get_tenant))
        .route("/api/tenants/:id", delete(tenant_handlers::delete_tenant))
        .route("/api/tenants/:id/upload", post(tenant_handlers::upload_for_tenant))
        .route("/api/tenants/:id/search", post(tenant_handlers::search_for_tenant))
        // OpenAI-compatible routes for Open Web UI integration
        .route("/v1/models", get(openai::list_models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .layer(Extension(api_base))
        .layer(cors)
        .with_state(state)
}

/// Run the API server
pub async fn run_server(vpk: VPK, config: APIConfig) -> VPKResult<()> {
    let app = create_router(vpk, config.api_base.clone());
    let addr = format!("{}:{}", config.bind_address, config.port);

    println!("🚀 VPK Demo Server starting on http://{}", addr);
    println!("📊 Open http://{} in your browser to see the visualization", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| crate::error::VPKError::Other(format!("Failed to bind: {}", e)))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| crate::error::VPKError::Other(format!("Server error: {}", e)))?;

    Ok(())
}
