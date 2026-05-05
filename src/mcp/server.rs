//! MCP (Model Context Protocol) server for VPK
//!
//! Implements JSON-RPC 2.0 over stdio — the standard MCP transport.
//! Each message is a single JSON object terminated by `\n`.
//!
//! # Tools exposed
//!
//! | Tool                   | Description                                      |
//! |------------------------|--------------------------------------------------|
//! | search_knowledge_base  | Encrypted fan-out search, returns real doc IDs   |
//! | upload_document        | Embed → encrypt → shard → return doc_id         |
//! | get_health             | Shard status, index size, embedding info         |
//! | get_audit_log          | Firewall-style log (no content, no Alice ID)     |
//!
//! # Trust boundary
//!
//! This server runs inside Bob's trusted zone. Alice's identity has already
//! been stripped by Bob's application layer before reaching VPK. The MCP
//! server is identity-blind.

use crate::error::VPKResult;
use crate::vpk::VPK;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;

// ── JSON-RPC 2.0 wire types ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    /// None for notifications (server must not respond)
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i64,
    message: String,
}

impl JsonRpcResponse {
    fn ok(id: Value, result: Value) -> Self {
        Self { jsonrpc: "2.0".into(), id, result: Some(result), error: None }
    }

    fn err(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(RpcError { code, message: message.into() }),
        }
    }
}

// ── Tool result helpers ────────────────────────────────────────────────────

/// Wrap a serialisable value into an MCP `CallToolResult` content block.
fn text_content(value: &impl Serialize) -> Value {
    json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(value).unwrap_or_default()
        }]
    })
}

// ── Server ─────────────────────────────────────────────────────────────────

/// MCP server wrapping VPK
pub struct McpServer {
    vpk: Arc<RwLock<VPK>>,
}

impl McpServer {
    pub fn new(vpk: Arc<RwLock<VPK>>) -> Self {
        Self { vpk }
    }

    /// Run the MCP server; reads from stdin, writes to stdout.
    /// Returns when stdin reaches EOF.
    pub async fn serve(&self) -> VPKResult<()> {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut writer = stdout;
        let mut line = String::new();

        loop {
            line.clear();
            let n = reader.read_line(&mut line).await.map_err(|e| {
                crate::error::VPKError::Other(format!("Stdin read error: {}", e))
            })?;

            if n == 0 {
                break; // EOF — client disconnected
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            match serde_json::from_str::<JsonRpcRequest>(trimmed) {
                Ok(req) => {
                    if let Some(resp) = self.handle(req).await {
                        let mut out = serde_json::to_string(&resp).unwrap_or_default();
                        out.push('\n');
                        writer.write_all(out.as_bytes()).await.ok();
                        writer.flush().await.ok();
                    }
                }
                Err(e) => {
                    let resp = JsonRpcResponse::err(Value::Null, -32700, format!("Parse error: {}", e));
                    let mut out = serde_json::to_string(&resp).unwrap_or_default();
                    out.push('\n');
                    writer.write_all(out.as_bytes()).await.ok();
                    writer.flush().await.ok();
                }
            }
        }
        Ok(())
    }

    async fn handle(&self, req: JsonRpcRequest) -> Option<JsonRpcResponse> {
        let id = req.id.clone().unwrap_or(Value::Null);
        let is_notification = req.id.is_none();

        let outcome: Option<Result<Value, (i64, String)>> = match req.method.as_str() {
            "initialize" => Some(Ok(self.rpc_initialize())),
            "initialized" => None, // notification — no response
            "ping" => Some(Ok(json!({}))),
            "tools/list" => Some(Ok(self.rpc_tools_list())),
            "tools/call" => Some(self.rpc_tools_call(req.params).await),
            _ => Some(Err((-32601, format!("Method not found: {}", req.method)))),
        };

        match outcome {
            None => None,
            Some(_) if is_notification => None,
            Some(Ok(v)) => Some(JsonRpcResponse::ok(id, v)),
            Some(Err((code, msg))) => Some(JsonRpcResponse::err(id, code, msg)),
        }
    }

    // ── RPC handlers ──────────────────────────────────────────────────────

    fn rpc_initialize(&self) -> Value {
        json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": {
                "name": "vpk-mcp-server",
                "version": "0.1.0"
            }
        })
    }

    fn rpc_tools_list(&self) -> Value {
        json!({
            "tools": [
                {
                    "name": "search_knowledge_base",
                    "description": "Search the encrypted knowledge base. Fan-out across all Eve shards, returns real document IDs and similarity scores. Alice identity is never transmitted or logged.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top results to return (default: 5)",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 100
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "upload_document",
                    "description": "Upload a document to the encrypted knowledge base. VPK embeds, encrypts, and distributes the vector across Eve shards. Returns the real document ID for Bob's document store.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Document text to embed and encrypt"
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata (stored in Bob's DB, never sent to Eve)"
                            }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "get_health",
                    "description": "Get VPK system health: shard reachability, index size, and embedding model info.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "get_audit_log",
                    "description": "Get firewall-style audit log entries. Each entry contains: event_type, timestamp, shard IDs, result count. Never contains query text, document content, or Alice identity.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "since": {
                                "type": "string",
                                "description": "ISO 8601 timestamp — return only events after this time (e.g. '2026-01-01T00:00:00Z')"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum entries to return (default: 100)",
                                "default": 100,
                                "minimum": 1,
                                "maximum": 1000
                            }
                        }
                    }
                }
            ]
        })
    }

    async fn rpc_tools_call(
        &self,
        params: Option<Value>,
    ) -> Result<Value, (i64, String)> {
        let p = params.ok_or_else(|| (-32602i64, "Missing params".to_string()))?;
        let name = p["name"]
            .as_str()
            .ok_or_else(|| (-32602i64, "Missing 'name'".to_string()))?;
        let args = p.get("arguments").cloned().unwrap_or(json!({}));

        match name {
            "search_knowledge_base" => self.tool_search(args).await,
            "upload_document" => self.tool_upload(args).await,
            "get_health" => self.tool_health().await,
            "get_audit_log" => self.tool_audit_log(args).await,
            other => Err((-32602, format!("Unknown tool: {}", other))),
        }
    }

    // ── Tool implementations ───────────────────────────────────────────────

    async fn tool_search(&self, args: Value) -> Result<Value, (i64, String)> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| (-32602i64, "Missing 'query'".to_string()))?
            .to_string();
        let top_k = args["top_k"].as_u64().unwrap_or(5).min(100) as usize;

        let vpk = self.vpk.read().await;
        let result = vpk
            .query(&query, top_k)
            .await
            .map_err(|e| (-32000i64, format!("Search failed: {}", e)))?;

        let items: Vec<Value> = result
            .doc_ids
            .iter()
            .zip(result.scores.iter())
            .map(|(doc_id, score)| json!({ "doc_id": doc_id, "score": score }))
            .collect();

        Ok(text_content(&items))
    }

    async fn tool_upload(&self, args: Value) -> Result<Value, (i64, String)> {
        let text = args["text"]
            .as_str()
            .ok_or_else(|| (-32602i64, "Missing 'text'".to_string()))?
            .to_string();
        let metadata = args.get("metadata").cloned();

        let mut vpk = self.vpk.write().await;
        let doc_id = vpk
            .upload_document(text, metadata)
            .await
            .map_err(|e| (-32000i64, format!("Upload failed: {}", e)))?;

        Ok(text_content(&json!({ "doc_id": doc_id })))
    }

    async fn tool_health(&self) -> Result<Value, (i64, String)> {
        let vpk = self.vpk.read().await;
        let status = vpk
            .health_status()
            .await
            .map_err(|e| (-32000i64, format!("Health check failed: {}", e)))?;
        Ok(text_content(&status))
    }

    async fn tool_audit_log(&self, args: Value) -> Result<Value, (i64, String)> {
        let limit = args["limit"].as_u64().unwrap_or(100).min(1000) as i64;
        let vpk = self.vpk.read().await;

        let entries = if let Some(since_str) = args["since"].as_str() {
            // Parse ISO 8601 timestamp
            let since = since_str
                .parse::<chrono::DateTime<chrono::Utc>>()
                .map_err(|e| (-32602i64, format!("Invalid 'since' timestamp: {}", e)))?;
            vpk.get_audit_log_since(since, limit)
                .await
                .map_err(|e| (-32000i64, format!("Audit log failed: {}", e)))?
        } else {
            vpk.get_audit_log(limit)
                .await
                .map_err(|e| (-32000i64, format!("Audit log failed: {}", e)))?
        };

        Ok(text_content(&entries))
    }
}

/// Convenience function: run MCP server with an already-initialized VPK
pub async fn serve_mcp(vpk: VPK) -> VPKResult<()> {
    let vpk = Arc::new(RwLock::new(vpk));
    let server = McpServer::new(vpk);
    server.serve().await
}
