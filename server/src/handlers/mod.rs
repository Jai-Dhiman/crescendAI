//! API Request Handlers
//!
//! This module contains all HTTP request handlers for the CrescendAI API.

/// Chat system handlers (sessions, streaming chat)
pub mod chat;

/// Performance feedback generation handlers
pub mod feedback;

/// Audio file upload and recording management handlers
pub mod upload;

/// User context management handlers
pub mod context;
