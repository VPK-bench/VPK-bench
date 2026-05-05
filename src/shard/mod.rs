//! Multi-shard Eve management
//!
//! Provides the `ShardManager` for horizontal partitioning across N Eve instances.

pub mod manager;

pub use manager::{ShardHealthStatus, ShardManager, ShardedSearchResult};
