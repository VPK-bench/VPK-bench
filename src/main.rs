//! VPK Server
//!
//! Modes (controlled by the `VPK_MODE` environment variable):
//!
//! * `VPK_MODE=mcp`  — Run as an MCP server (JSON-RPC 2.0 over stdio).
//!                     Used by Claude and LLM agents inside Bob's trusted zone.
//!
//! * `VPK_MODE=api`  — Run the REST API demo server with visualization (default).
//!
//! Example: `VPK_MODE=mcp ./vpk`

use phe::{api, config::VPKConfig, mcp, vpk::VPK};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI arguments: --config <file>  --api-base <url>  --port <n>
    let args: Vec<String> = std::env::args().collect();
    let mut config_file: Option<String> = None;
    let mut api_base_override: Option<String> = None;
    let mut port_override: Option<u16> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => { i += 1; config_file = args.get(i).cloned(); }
            "--api-base" => { i += 1; api_base_override = args.get(i).cloned(); }
            "--port" => { i += 1; port_override = args.get(i).and_then(|p| p.parse().ok()); }
            _ => {}
        }
        i += 1;
    }

    // Load configuration from file or use defaults
    let mut config = match config_file {
        Some(ref path) => {
            println!("📄 Loading config from: {}", path);
            VPKConfig::from_file(path)?
        }
        None => VPKConfig::default_test(),
    };

    // Update database connection string with credentials
    config.database.connection_string = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://postgres:test@localhost/vpk_test".to_string());

    println!("🔐 VPK - Virtual Private Knowledge");
    println!("==================================\n");

    // Extract api_base before config is moved into VPK
    let config_api_base = config.api.as_ref().map(|a| a.api_base.clone()).unwrap_or_default();

    // Create and initialize VPK
    println!("📦 Creating VPK instance...");
    let mut vpk = VPK::new(config).await?;

    println!("🔧 Initializing VPK system...");
    if let Err(e) = vpk.initialize().await {
        eprintln!("❌ Initialization failed: {:?}", e);
        return Err(e.into());
    }

    println!("✅ VPK initialized successfully\n");
    println!("State: {:?}", vpk.state());

    // The database starts empty. Use the dataset load scripts to populate:
    //   python experiments/load_nfcorpus.py
    //   python experiments/load_nq.py

    // Choose operating mode
    let mode = std::env::var("VPK_MODE").unwrap_or_else(|_| "api".to_string());

    match mode.as_str() {
        "mcp" => {
            eprintln!("VPK MCP server starting (stdio transport)");
            eprintln!("AI clients can now call: search_knowledge_base, upload_document, get_health, get_audit_log");
            mcp::serve_mcp(vpk).await?;
        }
        _ => {
            // REST API demo server (default)
            let mut api_config = api::APIConfig::default();
            // Apply overrides: CLI flags > config file > defaults
            if let Some(port) = port_override {
                api_config.port = port;
            }
            if let Some(base) = api_base_override {
                api_config.api_base = base;
            } else if !config_api_base.is_empty() {
                api_config.api_base = config_api_base;
            }
            if !api_config.api_base.is_empty() {
                println!("🔗 API base URL: {}", api_config.api_base);
            }
            println!("🌐 Starting web server...\n");
            api::run_server(vpk, api_config).await?;
        }
    }

    Ok(())
}
