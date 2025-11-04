// CrescendAI Server - Tool Execution Handlers
// Handlers that execute tools when Dedalus requests them

use crate::db;
use crate::rag_tools::*;
use serde_json;
use worker::*;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone)]
pub enum ToolExecutionError {
    /// Invalid tool name
    UnknownTool(String),

    /// Invalid tool arguments
    InvalidArguments(String),

    /// Tool execution failed
    ExecutionFailed(String),

    /// Database error
    DatabaseError(String),

    /// Knowledge base search error
    SearchError(String),
}

impl std::fmt::Display for ToolExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolExecutionError::UnknownTool(name) => write!(f, "Unknown tool: {}", name),
            ToolExecutionError::InvalidArguments(msg) => write!(f, "Invalid arguments: {}", msg),
            ToolExecutionError::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
            ToolExecutionError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            ToolExecutionError::SearchError(msg) => write!(f, "Search error: {}", msg),
        }
    }
}

impl std::error::Error for ToolExecutionError {}

pub type ToolExecutionResult<T> = std::result::Result<T, ToolExecutionError>;

// ============================================================================
// Tool Executor
// ============================================================================

/// Main tool executor that dispatches tool calls to appropriate handlers
pub struct ToolExecutor<'a> {
    env: &'a Env,
}

impl<'a> ToolExecutor<'a> {
    /// Create a new tool executor
    pub fn new(env: &'a Env) -> Self {
        Self { env }
    }

    /// Execute a tool by name with JSON arguments
    pub async fn execute_tool(
        &self,
        tool_name: &str,
        arguments_json: &str,
    ) -> ToolExecutionResult<String> {
        worker::console_log!(
            "Executing tool: {} with args: {}",
            tool_name,
            arguments_json
        );

        match tool_name {
            "search_knowledge_base" => self.handle_search_knowledge_base(arguments_json).await,
            "get_performance_analysis" => {
                self.handle_get_performance_analysis(arguments_json).await
            }
            "get_user_context" => self.handle_get_user_context(arguments_json).await,
            _ => Err(ToolExecutionError::UnknownTool(tool_name.to_string())),
        }
    }

    /// Handle search_knowledge_base tool
    async fn handle_search_knowledge_base(&self, args_json: &str) -> ToolExecutionResult<String> {
        // Parse arguments
        let request: SearchKnowledgeBaseRequest =
            parse_tool_arguments(args_json).map_err(|e| ToolExecutionError::InvalidArguments(e))?;

        // Validate query
        validate_search_query(&request.query)
            .map_err(|e| ToolExecutionError::InvalidArguments(e))?;

        // Get D1 database
        let d1 = self.env.d1("DB").map_err(|e| {
            ToolExecutionError::DatabaseError(format!("Failed to get D1 binding: {}", e))
        })?;

        // Perform search using knowledge base module
        match self.search_knowledge_base_internal(&d1, &request).await {
            Ok(response) => {
                // Format response as string for LLM
                let formatted = format_search_response(&response);
                Ok(formatted)
            }
            Err(e) => Err(ToolExecutionError::SearchError(format!(
                "Search failed: {}",
                e
            ))),
        }
    }

    /// Internal knowledge base search implementation
    async fn search_knowledge_base_internal(
        &self,
        _d1: &D1Database,
        request: &SearchKnowledgeBaseRequest,
    ) -> Result<SearchKnowledgeBaseResponse> {
        // Use the new hybrid search from knowledge_base module
        let chunks = crate::knowledge_base::hybrid_search(
            self.env,
            &request.query,
            request.top_k,
            true, // use cache
        )
        .await
        .map_err(|e| worker::Error::RustError(format!("Hybrid search failed: {}", e)))?;

        // Format results with citations
        let formatted = crate::knowledge_base::format_search_results(self.env, chunks.clone())
            .await
            .map_err(|e| worker::Error::RustError(format!("Failed to format results: {}", e)))?;

        // Convert to SearchResult format for compatibility with existing code
        let mut search_results = Vec::new();

        for result in formatted.results {
            search_results.push(SearchResult {
                id: result.chunk.id.clone(),
                score: result.relevance_score,
                content: result.chunk.content.clone(),
                source: SourceMetadata {
                    title: result.document_title.clone(),
                    author: None, // Can be enhanced later
                    page: Some(result.chunk.chunk_index as u32),
                    section: None, // Can be enhanced later
                },
                highlights: None,
            });
        }

        Ok(SearchKnowledgeBaseResponse {
            total_results: search_results.len(),
            results: search_results,
            query: request.query.clone(),
        })
    }

    /// Handle get_performance_analysis tool
    async fn handle_get_performance_analysis(
        &self,
        args_json: &str,
    ) -> ToolExecutionResult<String> {
        // Parse arguments
        let request: GetPerformanceAnalysisRequest =
            parse_tool_arguments(args_json).map_err(|e| ToolExecutionError::InvalidArguments(e))?;

        // Validate recording ID
        validate_recording_id(&request.recording_id)
            .map_err(|e| ToolExecutionError::InvalidArguments(e))?;

        // Query recording from database
        match db::get_recording(self.env, &request.recording_id).await {
            Ok(recording) => {
                // Check if analysis exists
                if recording.status != "analyzed" {
                    return Ok(format!(
                        "Analysis for recording {} is not yet complete (status: {}). Please try again later.",
                        request.recording_id,
                        recording.status
                    ));
                }

                // For now, return a placeholder response
                // TODO: Retrieve actual analysis results from analysis_results table
                let response = GetPerformanceAnalysisResponse {
                    recording_id: request.recording_id.clone(),
                    analysis: self.get_mock_analysis(),
                    temporal_analysis: if request.include_temporal {
                        Some(self.get_mock_temporal_analysis())
                    } else {
                        None
                    },
                };

                Ok(format_analysis_response(&response))
            }
            Err(db::DbError::NotFound(_)) => Ok(format!(
                "Recording {} not found. Please upload a recording first.",
                request.recording_id
            )),
            Err(e) => Err(ToolExecutionError::DatabaseError(format!(
                "Failed to retrieve recording: {}",
                e
            ))),
        }
    }

    /// Handle get_user_context tool
    async fn handle_get_user_context(&self, args_json: &str) -> ToolExecutionResult<String> {
        // Parse arguments
        let request: GetUserContextRequest =
            parse_tool_arguments(args_json).map_err(|e| ToolExecutionError::InvalidArguments(e))?;

        // Query session from database
        match db::get_session(self.env, &request.session_id).await {
            Ok(session) => {
                // Get user context (for now, use mock data)
                // TODO: Implement actual user_contexts table query
                let response = GetUserContextResponse {
                    session_id: request.session_id.clone(),
                    user_context: self.get_mock_user_context(),
                };

                Ok(format_user_context_response(&response))
            }
            Err(db::DbError::NotFound(_)) => {
                Ok(format!("Session {} not found.", request.session_id))
            }
            Err(e) => Err(ToolExecutionError::DatabaseError(format!(
                "Failed to retrieve session: {}",
                e
            ))),
        }
    }

    // ========================================================================
    // Mock Data Helpers (TODO: Replace with actual data)
    // ========================================================================

    fn get_mock_analysis(&self) -> PerformanceAnalysis {
        PerformanceAnalysis {
            timing_stable_unstable: 0.35,
            articulation_short_long: 0.62,
            articulation_soft_hard: 0.48,
            pedal_sparse_saturated: 0.55,
            pedal_clean_blurred: 0.42,
            timbre_even_colorful: 0.58,
            timbre_shallow_rich: 0.51,
            timbre_bright_dark: 0.45,
            timbre_soft_loud: 0.53,
            dynamic_sophisticated_raw: 0.47,
            dynamic_range_little_large: 0.61,
            music_making_fast_slow: 0.52,
            music_making_flat_spacious: 0.56,
            music_making_disproportioned_balanced: 0.49,
            music_making_pure_dramatic: 0.54,
            emotion_mood_optimistic_dark: 0.48,
            emotion_mood_low_high_energy: 0.57,
            emotion_mood_honest_imaginative: 0.50,
            interpretation_unsatisfactory_convincing: 0.65,
        }
    }

    fn get_mock_temporal_analysis(&self) -> Vec<TemporalSegment> {
        vec![
            TemporalSegment {
                start_time: 0.0,
                end_time: 15.0,
                timestamp: "0:00-0:15".to_string(),
                scores: self.get_mock_analysis(),
                notable_features: vec![
                    "Strong opening with clear articulation".to_string(),
                    "Good dynamic control".to_string(),
                ],
            },
            TemporalSegment {
                start_time: 15.0,
                end_time: 30.0,
                timestamp: "0:15-0:30".to_string(),
                scores: self.get_mock_analysis(),
                notable_features: vec!["Slight tempo instability in middle section".to_string()],
            },
        ]
    }

    fn get_mock_user_context(&self) -> UserContext {
        UserContext {
            goals: vec![
                "Improve sight-reading skills".to_string(),
                "Master advanced pedaling techniques".to_string(),
            ],
            constraints: vec![
                "30 minutes practice per day".to_string(),
                "No access to teacher weekly".to_string(),
            ],
            repertoire: vec![
                "Bach: Prelude in C Major (WTC Book I)".to_string(),
                "Chopin: Nocturne in E-flat Major, Op. 9 No. 2".to_string(),
            ],
            skill_level: Some("Intermediate".to_string()),
            learning_style: Some("Visual and kinesthetic learner".to_string()),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_execution_error_display() {
        let err = ToolExecutionError::UnknownTool("invalid_tool".to_string());
        assert_eq!(err.to_string(), "Unknown tool: invalid_tool");

        let err = ToolExecutionError::InvalidArguments("missing field".to_string());
        assert_eq!(err.to_string(), "Invalid arguments: missing field");
    }

    #[test]
    fn test_parse_search_request() {
        let json = r#"{"query": "pedaling", "top_k": 3, "context": "user question"}"#;
        let req: SearchKnowledgeBaseRequest = parse_tool_arguments(json).unwrap();
        assert_eq!(req.query, "pedaling");
        assert_eq!(req.top_k, 3);
        assert_eq!(req.context, Some("user question".to_string()));
    }

    #[test]
    fn test_parse_analysis_request() {
        let json = r#"{"recording_id": "550e8400-e29b-41d4-a716-446655440000", "include_temporal": false}"#;
        let req: GetPerformanceAnalysisRequest = parse_tool_arguments(json).unwrap();
        assert_eq!(req.recording_id, "550e8400-e29b-41d4-a716-446655440000");
        assert_eq!(req.include_temporal, false);
    }

    #[test]
    fn test_parse_user_context_request() {
        let json = r#"{"session_id": "session123"}"#;
        let req: GetUserContextRequest = parse_tool_arguments(json).unwrap();
        assert_eq!(req.session_id, "session123");
    }
}
