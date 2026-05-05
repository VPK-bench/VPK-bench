//! OpenAI-compatible API endpoints for Open Web UI integration

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::handlers::AppState;

// ============= REQUEST TYPES =============

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    #[serde(default)]
    pub stream: bool,

    #[serde(default = "default_temperature")]
    pub temperature: f64,

    #[serde(default)]
    pub max_tokens: Option<usize>,

    // VPK-specific: how many search results to include
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_temperature() -> f64 {
    0.7
}
fn default_top_k() -> usize {
    5
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String, // "user", "assistant", "system"
    pub content: String,
}

// ============= RESPONSE TYPES =============

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageStats,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String, // "stop", "length", "error"
}

#[derive(Debug, Serialize)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ============= MODEL LIST TYPES =============

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String, // "list"
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String, // "model"
    pub created: i64,
    pub owned_by: String,
}

// ============= HANDLERS =============

/// OpenAI-compatible chat completions endpoint
pub async fn chat_completions(
    State(vpk): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    // Step 1: Extract user query from last message
    let user_query = match extract_user_query(&request.messages) {
        Some(query) => query,
        None => {
            return error_response(
                "No user message found in conversation",
                "invalid_request_error",
            );
        }
    };

    // Step 2: Query VPK with encrypted search
    let vpk = vpk.read().await;
    let search_results = match vpk.query(&user_query, request.top_k).await {
        Ok(results) => results,
        Err(e) => {
            return error_response(&format!("Search failed: {}", e), "vpk_search_error");
        }
    };

    // Step 3: Format results as chat response
    let response_content = format_search_results(&search_results, &user_query);

    // Step 4: Calculate token usage (approximate)
    let prompt_tokens = estimate_tokens(&request.messages);
    let completion_tokens = estimate_tokens_from_text(&response_content);

    // Step 5: Build OpenAI-compatible response
    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: Utc::now().timestamp(),
        model: request.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response_content,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: UsageStats {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    (StatusCode::OK, Json(response)).into_response()
}

/// OpenAI-compatible models list endpoint
pub async fn list_models() -> Response {
    let models = ModelListResponse {
        object: "list".to_string(),
        data: vec![
            ModelInfo {
                id: "vpk-encrypted-search".to_string(),
                object: "model".to_string(),
                created: 1704067200, // 2024-01-01
                owned_by: "vpk".to_string(),
            },
            ModelInfo {
                id: "vpk-rome".to_string(),
                object: "model".to_string(),
                created: 1704067200,
                owned_by: "vpk".to_string(),
            },
            ModelInfo {
                id: "vpk-scrambling".to_string(),
                object: "model".to_string(),
                created: 1704067200,
                owned_by: "vpk".to_string(),
            },
        ],
    };

    (StatusCode::OK, Json(models)).into_response()
}

// ============= HELPER FUNCTIONS =============

fn extract_user_query(messages: &[ChatMessage]) -> Option<String> {
    messages
        .iter()
        .rev() // Start from most recent
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
}

fn format_search_results(results: &crate::vpk::QueryResult, query: &str) -> String {
    if results.documents.is_empty() {
        return format!(
            "I searched the encrypted knowledge base for '{}' but found no relevant results.",
            query
        );
    }

    let mut response = format!("Based on encrypted search for '{}':\n\n", query);

    for (idx, (doc, score)) in results
        .documents
        .iter()
        .zip(results.scores.iter())
        .enumerate()
    {
        response.push_str(&format!(
            "{}. [Score: {:.3}] {}\n\n",
            idx + 1,
            score,
            doc.content.trim()
        ));
    }

    response.push_str(&format!(
        "\n🔐 Note: All {} results were retrieved using encrypted vector search. \
        The vector database never saw your query or these documents in plaintext.",
        results.documents.len()
    ));

    response
}

fn estimate_tokens(messages: &[ChatMessage]) -> usize {
    // Rough estimate: ~4 characters per token
    messages.iter().map(|m| m.content.len() / 4).sum()
}

fn estimate_tokens_from_text(text: &str) -> usize {
    text.len() / 4
}

fn error_response(message: &str, error_type: &str) -> Response {
    #[derive(Serialize)]
    struct ErrorResponse {
        error: ErrorDetail,
    }

    #[derive(Serialize)]
    struct ErrorDetail {
        message: String,
        r#type: String,
    }

    let error = ErrorResponse {
        error: ErrorDetail {
            message: message.to_string(),
            r#type: error_type.to_string(),
        },
    };

    (StatusCode::BAD_REQUEST, Json(error)).into_response()
}
