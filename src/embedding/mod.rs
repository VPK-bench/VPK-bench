//! Embedding models for converting text to vectors

pub mod model;

pub use model::{DummyEmbeddingModel, EmbeddingModel, HttpEmbeddingModel};
