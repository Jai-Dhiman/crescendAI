// CrescendAI Server - API Models & Serialization
// Dedalus API types and other API-level data structures

use serde::{Deserialize, Serialize};

// ============================================================================
// Dedalus API Types (OpenAI-compatible chat completion format)
// ============================================================================

/// Represents a message in a chat conversation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DedalusMessage {
    /// Role of the message author (system, user, assistant, tool)
    pub role: String,

    /// Content of the message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Name of the tool (for tool messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool calls made by the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Tool call ID (for tool response messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl DedalusMessage {
    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message with tool calls
    pub fn assistant_with_tools(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: None,
            name: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// Create a tool response message
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Represents a tool call made by the assistant
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    /// Unique identifier for the tool call
    pub id: String,

    /// Type of the tool call (always "function" for now)
    #[serde(rename = "type")]
    pub call_type: String,

    /// Function call details
    pub function: FunctionCall,
}

/// Function call details within a tool call
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionCall {
    /// Name of the function to call
    pub name: String,

    /// JSON string of function arguments
    pub arguments: String,
}

/// Tool definition for Dedalus function calling
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DedalusTool {
    /// Type of tool (always "function")
    #[serde(rename = "type")]
    pub tool_type: String,

    /// Function definition
    pub function: FunctionDefinition,
}

impl DedalusTool {
    /// Create a new function tool
    pub fn function(name: impl Into<String>, description: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

/// Function definition for a tool
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionDefinition {
    /// Name of the function
    pub name: String,

    /// Description of what the function does
    pub description: String,

    /// JSON Schema for function parameters
    pub parameters: serde_json::Value,
}

/// Request body for Dedalus chat completions
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DedalusChatRequest {
    /// Model identifier (e.g., "openai/gpt-4o-mini", "openai/gpt-5-nano")
    pub model: String,

    /// List of messages in the conversation
    pub messages: Vec<DedalusMessage>,

    /// Available tools for function calling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<DedalusTool>>,

    /// Temperature for sampling (0.0-2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// Response from Dedalus chat completions (non-streaming)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DedalusChatResponse {
    /// Unique identifier for the response
    pub id: String,

    /// Object type (always "chat.completion")
    pub object: String,

    /// Unix timestamp of when the response was created
    pub created: i64,

    /// Model used for the completion
    pub model: String,

    /// List of completion choices
    pub choices: Vec<Choice>,

    /// Usage statistics
    pub usage: Usage,
}

/// A single completion choice
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Choice {
    /// Index of this choice
    pub index: u32,

    /// The message generated by the model
    pub message: DedalusMessage,

    /// Reason why the model stopped generating
    pub finish_reason: String,
}

/// Token usage statistics
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,

    /// Number of tokens in the completion
    pub completion_tokens: u32,

    /// Total tokens used
    pub total_tokens: u32,
}

/// Server-sent event chunk from streaming response
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StreamChunk {
    /// Unique identifier for the chat completion
    pub id: String,

    /// Object type (always "chat.completion.chunk")
    pub object: String,

    /// Unix timestamp
    pub created: i64,

    /// Model used
    pub model: String,

    /// Array of choices
    pub choices: Vec<StreamChoice>,
}

/// A choice in a streaming response
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StreamChoice {
    /// Index of this choice
    pub index: u32,

    /// Delta content
    pub delta: Delta,

    /// Finish reason (null until final chunk)
    pub finish_reason: Option<String>,
}

/// Delta content in streaming response
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Delta {
    /// Role (only in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Content chunk
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Tool calls (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

// ============================================================================
// API Request/Response Types
// ============================================================================

/// Request to start a chat session
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateChatSessionRequest {
    /// Optional recording ID to associate with the session
    pub recording_id: Option<String>,
}

/// Request to send a chat message
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatMessageRequest {
    /// Session ID
    pub session_id: String,

    /// User message content
    pub message: String,
}

/// Request to update user context
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateUserContextRequest {
    /// User's learning goals
    pub goals: Option<Vec<String>>,

    /// User's constraints (time, physical, etc.)
    pub constraints: Option<Vec<String>>,

    /// Current repertoire
    pub repertoire: Option<Vec<String>>,
}

// ============================================================================
// Error Response Type
// ============================================================================

/// Standard error response format
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ErrorResponse {
    /// Error type
    pub error: String,

    /// Error message
    pub message: String,

    /// Optional error details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl ErrorResponse {
    /// Create a new error response
    pub fn new(error: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            message: message.into(),
            details: None,
        }
    }

    /// Create an error response with details
    pub fn with_details(error: impl Into<String>, message: impl Into<String>, details: serde_json::Value) -> Self {
        Self {
            error: error.into(),
            message: message.into(),
            details: Some(details),
        }
    }
}
