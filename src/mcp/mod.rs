//! MCP (Model Context Protocol) server
//!
//! Exposes VPK as an MCP server so AI clients (Claude, LLM agents) can
//! call the four VPK tools from within Bob's trusted zone.
//!
//! Transport: JSON-RPC 2.0 over stdio (newline-delimited JSON).
//!
//! # Usage
//!
//! ```text
//! VPK_MODE=mcp ./vpk
//! ```
//!
//! Or in code:
//! ```rust,ignore
//! use phe::mcp::serve_mcp;
//! // vpk is an initialized VPK instance
//! serve_mcp(vpk).await?;
//! ```

pub mod server;

pub use server::{McpServer, serve_mcp};
