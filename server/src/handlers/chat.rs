// CrescendAI Server - Chat Handler
// Implements streaming chat with Dedalus integration and message persistence

use worker::*;
use serde_json::json;
use crate::db::{sessions, messages};
use crate::dedalus_client::DedalusClient;
use crate::tool_executor;
use crate::models::{DedalusMessage, DedalusChatRequest, ChatMessageRequest, CreateChatSessionRequest};
use crate::rag_tools;
use crate::security::{validate_uuid, validate_body_size, SecurityError};

// ============================================================================
// Session Management Endpoints
// ============================================================================

/// Create a new chat session
/// POST /api/chat/sessions
pub async fn create_session(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    const MAX_JSON_SIZE: usize = 1024; // 1KB limit

    // Validate body size
    if let Ok(Some(content_length)) = req.headers().get("Content-Length") {
        if let Ok(size) = content_length.parse::<usize>() {
            validate_body_size(size, MAX_JSON_SIZE)?;
        }
    }

    let body: CreateChatSessionRequest = req.json().await?;

    // Validate recording_id if provided
    if let Some(ref recording_id) = body.recording_id {
        validate_uuid(recording_id)?;
    }

    // TODO: Extract user_id from authentication (for now use "default_user")
    let user_id = "default_user";

    // Create session in D1
    let session = sessions::create_session(
        &ctx.env,
        user_id,
        body.recording_id.as_deref(),
        None, // title will be auto-generated from first message
    ).await
        .map_err(|e| worker::Error::RustError(format!("Failed to create session: {}", e)))?;

    Response::from_json(&json!({
        "session_id": session.id,
        "user_id": session.user_id,
        "recording_id": session.recording_id,
        "created_at": session.created_at,
        "message": "Chat session created successfully"
    }))
}

/// Get a chat session with its message history
/// GET /api/chat/sessions/:id
pub async fn get_session(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let session_id = ctx.param("id")
        .ok_or_else(|| worker::Error::from(SecurityError::InvalidInput(
            "Missing session ID parameter".to_string()
        )))?;

    // Validate UUID format
    validate_uuid(session_id)?;

    // Get session from D1
    let session = sessions::get_session(&ctx.env, session_id).await
        .map_err(|e| worker::Error::RustError(format!("Failed to get session: {}", e)))?;

    // Get all messages for this session
    let session_messages = messages::get_messages_by_session(&ctx.env, session_id).await
        .map_err(|e| worker::Error::RustError(format!("Failed to get messages: {}", e)))?;

    Response::from_json(&json!({
        "session": {
            "id": session.id,
            "user_id": session.user_id,
            "recording_id": session.recording_id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        },
        "messages": session_messages.iter().map(|m| json!({
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "tool_calls": m.tool_calls,
            "tool_call_id": m.tool_call_id,
            "created_at": m.created_at,
        })).collect::<Vec<_>>(),
        "message_count": session_messages.len()
    }))
}

/// List all sessions for the current user
/// GET /api/chat/sessions
pub async fn list_sessions(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // TODO: Extract user_id from authentication
    let user_id = "default_user";

    // Get limit from query params (default 50, max 100)
    let url = req.url()?;
    let limit = url
        .query_pairs()
        .find(|(k, _)| k == "limit")
        .and_then(|(_, v)| v.parse::<u32>().ok())
        .unwrap_or(50)
        .min(100);

    // List sessions from D1
    let sessions_list = sessions::list_sessions_by_user(&ctx.env, user_id, Some(limit)).await
        .map_err(|e| worker::Error::RustError(format!("Failed to list sessions: {}", e)))?;

    // Get message count for each session
    let mut sessions_with_counts = Vec::new();
    for session in sessions_list {
        let count = messages::count_messages(&ctx.env, &session.id).await.unwrap_or(0);
        sessions_with_counts.push(json!({
            "id": session.id,
            "user_id": session.user_id,
            "recording_id": session.recording_id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "message_count": count,
        }));
    }

    Response::from_json(&json!({
        "sessions": sessions_with_counts,
        "total": sessions_with_counts.len()
    }))
}

/// Delete a chat session
/// DELETE /api/chat/sessions/:id
pub async fn delete_session(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let session_id = ctx.param("id")
        .ok_or_else(|| worker::Error::from(SecurityError::InvalidInput(
            "Missing session ID parameter".to_string()
        )))?;

    // Validate UUID format
    validate_uuid(session_id)?;

    // Verify session exists and belongs to user
    let session = sessions::get_session(&ctx.env, session_id).await
        .map_err(|e| worker::Error::RustError(format!("Session not found: {}", e)))?;

    // TODO: Verify user_id matches authenticated user
    // For now, anyone can delete any session

    // Delete session (cascade will delete messages)
    sessions::delete_session(&ctx.env, session_id).await
        .map_err(|e| worker::Error::RustError(format!("Failed to delete session: {}", e)))?;

    Response::from_json(&json!({
        "message": "Session deleted successfully",
        "session_id": session_id
    }))
}

// ============================================================================
// Streaming Chat Endpoint
// ============================================================================

/// Stream a chat response with Dedalus integration
/// POST /api/chat
pub async fn stream_chat(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    const MAX_JSON_SIZE: usize = 50 * 1024; // 50KB limit for chat requests

    // Validate body size
    if let Ok(Some(content_length)) = req.headers().get("Content-Length") {
        if let Ok(size) = content_length.parse::<usize>() {
            validate_body_size(size, MAX_JSON_SIZE)?;
        }
    }

    let body: ChatMessageRequest = req.json().await?;

    // Validate session_id
    validate_uuid(&body.session_id)?;

    // Verify session exists
    let session = sessions::get_session(&ctx.env, &body.session_id).await
        .map_err(|e| worker::Error::RustError(format!("Session not found: {}", e)))?;

    console_log!("Chat request for session {}: {}", body.session_id, body.message);

    // Store user message in D1
    let user_message = messages::insert_message(
        &ctx.env,
        &body.session_id,
        "user",
        &body.message,
        None,
        None,
        None,
    ).await
        .map_err(|e| worker::Error::RustError(format!("Failed to store user message: {}", e)))?;

    console_log!("Stored user message: {}", user_message.id);

    // Get conversation history (last 20 messages for context)
    let history = messages::get_last_n_messages(&ctx.env, &body.session_id, 20).await
        .map_err(|e| worker::Error::RustError(format!("Failed to get message history: {}", e)))?;

    // Build messages array for Dedalus
    let mut dedalus_messages = Vec::new();

    // Add system message
    let system_prompt = build_system_prompt(&session, &ctx.env).await;
    dedalus_messages.push(DedalusMessage::system(system_prompt.clone()));

    // Add conversation history (excluding the just-added user message since it's already in history)
    for msg in &history {
        match msg.role.as_str() {
            "user" => {
                dedalus_messages.push(DedalusMessage::user(&msg.content));
            }
            "assistant" => {
                if let Some(ref tool_calls_json) = msg.tool_calls {
                    // Parse tool calls and add assistant message with tool calls
                    if let Ok(tool_calls) = serde_json::from_str(tool_calls_json) {
                        dedalus_messages.push(DedalusMessage::assistant_with_tools(tool_calls));
                    } else {
                        dedalus_messages.push(DedalusMessage::assistant(&msg.content));
                    }
                } else {
                    dedalus_messages.push(DedalusMessage::assistant(&msg.content));
                }
            }
            "tool" => {
                if let Some(ref tool_call_id) = msg.tool_call_id {
                    dedalus_messages.push(DedalusMessage::tool(tool_call_id, &msg.content));
                }
            }
            _ => {}
        }
    }

    // Get RAG tools
    let tools = rag_tools::get_all_rag_tools();

    // Prepare Dedalus request
    let dedalus_request = DedalusChatRequest {
        model: "openai/gpt-4o-mini".to_string(), // TODO: Make configurable
        messages: dedalus_messages,
        tools: Some(tools),
        temperature: Some(0.7),
        max_tokens: Some(2000),
        stream: Some(true),
    };

    // Create Dedalus client from service binding
    let client = DedalusClient::from_env(&ctx.env)?;

    // Call Dedalus with non-streaming for now (TODO: implement SSE streaming)
    console_log!("Starting Dedalus request...");

    // Use non-streaming completion
    let mut dedalus_request_sync = dedalus_request.clone();
    dedalus_request_sync.stream = Some(false);

    match client.chat_completion(dedalus_request_sync).await {
        Ok(response) => {
            console_log!("Dedalus request completed successfully");

            // Extract content and tool calls from response
            let choice = response.choices.first()
                .ok_or_else(|| worker::Error::RustError("No choices in response".to_string()))?;

            let final_content = choice.message.content.clone().unwrap_or_default();
            let tool_calls_result = choice.message.tool_calls.clone();

            console_log!("Stream completed. Content length: {}, Tool calls: {}",
                final_content.len(),
                tool_calls_result.as_ref().map(|v| v.len()).unwrap_or(0)
            );

            // Handle tool calls if present
            let mut assistant_content = final_content.clone();
            let mut tool_calls_json = None;

            if let Some(tool_calls) = tool_calls_result {
                console_log!("Processing {} tool calls", tool_calls.len());

                // Store assistant message with tool calls
                tool_calls_json = Some(serde_json::to_string(&tool_calls)
                    .map_err(|e| worker::Error::RustError(format!("Failed to serialize tool calls: {}", e)))?);

                let assistant_msg = messages::insert_message(
                    &ctx.env,
                    &body.session_id,
                    "assistant",
                    "",
                    tool_calls_json.as_deref(),
                    None,
                    None,
                ).await
                    .map_err(|e| worker::Error::RustError(format!("Failed to store assistant message: {}", e)))?;

                console_log!("Stored assistant message with tool calls: {}", assistant_msg.id);

                // Execute each tool call and store results
                let executor = tool_executor::ToolExecutor::new(&ctx.env);
                for tool_call in &tool_calls {
                    let tool_name = &tool_call.function.name;
                    console_log!("Executing tool: {}", tool_name);

                    let tool_result = executor.execute_tool(
                        tool_name,
                        &tool_call.function.arguments,
                    ).await
                        .map_err(|e| worker::Error::RustError(format!("Tool execution failed: {}", e)))?;

                    console_log!("Tool {} executed successfully", tool_name);

                    // Store tool result message
                    messages::insert_message(
                        &ctx.env,
                        &body.session_id,
                        "tool",
                        &tool_result,
                        None,
                        Some(&tool_call.id),
                        None,
                    ).await
                        .map_err(|e| worker::Error::RustError(format!("Failed to store tool message: {}", e)))?;
                }

                // Make a second call to Dedalus with tool results
                console_log!("Making follow-up request with tool results");

                // Rebuild message history with tool results
                let updated_history = messages::get_last_n_messages(&ctx.env, &body.session_id, 25).await
                    .map_err(|e| worker::Error::RustError(format!("Failed to get updated history: {}", e)))?;

                let mut updated_messages = vec![DedalusMessage::system(system_prompt)];
                for msg in &updated_history {
                    match msg.role.as_str() {
                        "user" => updated_messages.push(DedalusMessage::user(&msg.content)),
                        "assistant" => {
                            if let Some(ref tc_json) = msg.tool_calls {
                                if let Ok(tc) = serde_json::from_str(tc_json) {
                                    updated_messages.push(DedalusMessage::assistant_with_tools(tc));
                                } else if !msg.content.is_empty() {
                                    updated_messages.push(DedalusMessage::assistant(&msg.content));
                                }
                            } else if !msg.content.is_empty() {
                                updated_messages.push(DedalusMessage::assistant(&msg.content));
                            }
                        }
                        "tool" => {
                            if let Some(ref tc_id) = msg.tool_call_id {
                                updated_messages.push(DedalusMessage::tool(tc_id, &msg.content));
                            }
                        }
                        _ => {}
                    }
                }

                let follow_up_request = DedalusChatRequest {
                    model: "openai/gpt-4o-mini".to_string(),
                    messages: updated_messages,
                    tools: Some(rag_tools::get_all_rag_tools()),
                    temperature: Some(0.7),
                    max_tokens: Some(2000),
                    stream: Some(false),
                };

                let follow_up_response = client.chat_completion(follow_up_request).await?;
                let follow_up_content = follow_up_response.choices.first()
                    .and_then(|c| c.message.content.clone())
                    .unwrap_or_default();

                assistant_content = follow_up_content;
                console_log!("Follow-up response completed. Length: {}", assistant_content.len());
            }

            // Store final assistant message
            if !assistant_content.is_empty() {
                let final_msg = messages::insert_message(
                    &ctx.env,
                    &body.session_id,
                    "assistant",
                    &assistant_content,
                    None,
                    None,
                    None,
                ).await
                    .map_err(|e| worker::Error::RustError(format!("Failed to store final message: {}", e)))?;

                console_log!("Stored final assistant message: {}", final_msg.id);
            }

            // Return non-streaming response (for now, can be changed to SSE later)
            Response::from_json(&json!({
                "session_id": body.session_id,
                "message": assistant_content,
                "role": "assistant",
                "tool_calls": tool_calls_json,
            }))
        }
        Err(e) => {
            console_log!("Dedalus streaming failed: {:?}", e);
            Err(worker::Error::RustError(format!("Dedalus API error: {}", e)))
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Build system prompt with context
async fn build_system_prompt(session: &crate::db::sessions::ChatSession, env: &Env) -> String {
    let mut prompt = String::from(
        "You are CrescendAI, an expert piano pedagogy assistant. \
        You help pianists improve their performance by analyzing their playing and providing \
        evidence-based feedback grounded in piano pedagogy research.\n\n"
    );

    // Add recording context if available
    if let Some(ref recording_id) = session.recording_id {
        prompt.push_str(&format!(
            "The user has shared a recording (ID: {}). You can retrieve analysis data using the \
            get_performance_analysis tool and search pedagogical knowledge using the \
            search_knowledge_base tool.\n\n",
            recording_id
        ));
    }

    prompt.push_str(
        "Guidelines:\n\
        - Provide specific, actionable advice based on performance data and pedagogy research\n\
        - Use the search_knowledge_base tool to ground your advice in authoritative sources\n\
        - Be encouraging and supportive while being honest about areas for improvement\n\
        - Consider the user's context and goals when providing recommendations\n\
        - Cite sources when referencing specific pedagogical techniques or concepts"
    );

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_prompt_format() {
        // Basic test to ensure system prompt is non-empty and contains key terms
        let prompt = "You are CrescendAI, an expert piano pedagogy assistant.";
        assert!(!prompt.is_empty());
        assert!(prompt.contains("CrescendAI"));
        assert!(prompt.contains("piano"));
    }
}
