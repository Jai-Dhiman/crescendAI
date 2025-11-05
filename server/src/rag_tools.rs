// CrescendAI Server - RAG Tool Definitions
// Tool schemas and definitions for Dedalus function calling

use serde::{Deserialize, Serialize};
use serde_json::json;
use crate::models::{DedalusTool, FunctionDefinition};

// ============================================================================
// Tool Schemas
// ============================================================================

/// Get the search_knowledge_base tool definition
pub fn search_knowledge_base_tool() -> DedalusTool {
    DedalusTool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "search_knowledge_base".to_string(),
            description: "Search the piano pedagogy knowledge base for relevant information. Use this to find teaching methods, practice techniques, performance advice, and musical concepts.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific and include relevant keywords (e.g., 'staccato articulation techniques', 'pedaling in Chopin nocturnes', 'hand independence exercises')."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (1-10). Default is 5.",
                        "minimum": 1,
                        "maximum": 10
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context about why this search is being performed (e.g., 'user asked about pedaling', 'analyzing staccato performance'). Helps refine results."
                    }
                },
                "required": ["query"]
            }),
        }
    }
}

/// Get the get_performance_analysis tool definition
pub fn get_performance_analysis_tool() -> DedalusTool {
    DedalusTool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_performance_analysis".to_string(),
            description: "Retrieve detailed performance analysis results for a recording, including 16-dimensional AST model scores and temporal analysis. Use this when discussing a specific recording or performance.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "recording_id": {
                        "type": "string",
                        "description": "The unique identifier for the recording to analyze."
                    },
                    "include_temporal": {
                        "type": "boolean",
                        "description": "Whether to include temporal (time-based) analysis. Default is true.",
                        "default": true
                    }
                },
                "required": ["recording_id"]
            }),
        }
    }
}

/// Get the get_user_context tool definition
pub fn get_user_context_tool() -> DedalusTool {
    DedalusTool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_user_context".to_string(),
            description: "Retrieve the user's learning context including their goals, constraints, current repertoire, and skill level. Use this to personalize advice and recommendations.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The chat session ID to get user context for."
                    }
                },
                "required": ["session_id"]
            }),
        }
    }
}

/// Get all available RAG tools
pub fn get_all_rag_tools() -> Vec<DedalusTool> {
    vec![
        search_knowledge_base_tool(),
        get_performance_analysis_tool(),
        get_user_context_tool(),
    ]
}

// ============================================================================
// Tool Request/Response Types
// ============================================================================

/// Request for search_knowledge_base tool
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchKnowledgeBaseRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    pub context: Option<String>,
}

fn default_top_k() -> u32 {
    5
}

/// Response from search_knowledge_base tool
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchKnowledgeBaseResponse {
    pub results: Vec<SearchResult>,
    pub total_results: usize,
    pub query: String,
}

/// A single search result
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchResult {
    /// Chunk ID
    pub id: String,

    /// Relevance score (0.0-1.0)
    pub score: f32,

    /// Chunk content
    pub content: String,

    /// Source document metadata
    pub source: SourceMetadata,

    /// Highlighted snippets (optional)
    pub highlights: Option<Vec<String>>,
}

/// Source document metadata
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SourceMetadata {
    /// Document title
    pub title: String,

    /// Author(s)
    pub author: Option<String>,

    /// Page number
    pub page: Option<u32>,

    /// Section/chapter
    pub section: Option<String>,
}

/// Request for get_performance_analysis tool
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetPerformanceAnalysisRequest {
    pub recording_id: String,
    #[serde(default = "default_include_temporal")]
    pub include_temporal: bool,
}

fn default_include_temporal() -> bool {
    true
}

/// Response from get_performance_analysis tool
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetPerformanceAnalysisResponse {
    pub recording_id: String,
    pub analysis: PerformanceAnalysis,
    pub temporal_analysis: Option<Vec<TemporalSegment>>,
}

/// Performance analysis with 16D scores
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PerformanceAnalysis {
    // Timing
    pub timing_stable_unstable: f32,

    // Articulation
    pub articulation_short_long: f32,
    pub articulation_soft_hard: f32,

    // Pedal
    pub pedal_sparse_saturated: f32,
    pub pedal_clean_blurred: f32,

    // Timbre
    pub timbre_even_colorful: f32,
    pub timbre_shallow_rich: f32,
    pub timbre_bright_dark: f32,
    pub timbre_soft_loud: f32,

    // Dynamics
    pub dynamic_sophisticated_raw: f32,
    pub dynamic_range_little_large: f32,

    // Music Making
    pub music_making_fast_slow: f32,
    pub music_making_flat_spacious: f32,
    pub music_making_disproportioned_balanced: f32,
    pub music_making_pure_dramatic: f32,

    // Emotion/Mood
    pub emotion_mood_optimistic_dark: f32,
    pub emotion_mood_low_high_energy: f32,
    pub emotion_mood_honest_imaginative: f32,

    // Interpretation
    pub interpretation_unsatisfactory_convincing: f32,
}

/// Temporal segment analysis
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TemporalSegment {
    pub start_time: f32,
    pub end_time: f32,
    pub timestamp: String, // e.g., "0:00-0:15"
    pub scores: PerformanceAnalysis,
    pub notable_features: Vec<String>,
}

/// Request for get_user_context tool
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetUserContextRequest {
    pub session_id: String,
}

/// Response from get_user_context tool
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetUserContextResponse {
    pub session_id: String,
    pub user_context: UserContext,
}

/// User learning context
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UserContext {
    /// User's learning goals
    pub goals: Vec<String>,

    /// User's constraints (time, physical, etc.)
    pub constraints: Vec<String>,

    /// Current repertoire
    pub repertoire: Vec<String>,

    /// Estimated skill level (beginner, intermediate, advanced, professional)
    pub skill_level: Option<String>,

    /// Preferred learning style
    pub learning_style: Option<String>,
}

// ============================================================================
// Tool Response Formatting
// ============================================================================

/// Format a SearchKnowledgeBaseResponse as a tool result string
pub fn format_search_response(response: &SearchKnowledgeBaseResponse) -> String {
    if response.results.is_empty() {
        return format!("No results found for query: '{}'", response.query);
    }

    let mut output = format!(
        "Found {} results for '{}'\n\n",
        response.total_results,
        response.query
    );

    for (idx, result) in response.results.iter().enumerate() {
        output.push_str(&format!(
            "{}. [Score: {:.2}] {}\n",
            idx + 1,
            result.score,
            result.content
        ));

        // Add source metadata
        output.push_str(&format!("   Source: {}", result.source.title));
        if let Some(author) = &result.source.author {
            output.push_str(&format!(" by {}", author));
        }
        if let Some(page) = result.source.page {
            output.push_str(&format!(", p. {}", page));
        }
        output.push_str("\n\n");
    }

    output
}

/// Format a GetPerformanceAnalysisResponse as a tool result string
pub fn format_analysis_response(response: &GetPerformanceAnalysisResponse) -> String {
    let mut output = format!("Performance Analysis for Recording {}\n\n", response.recording_id);

    // Overall scores
    output.push_str("Overall Scores (0.0-1.0 scale):\n");
    let analysis = &response.analysis;

    output.push_str(&format!("  Timing: {:.2} (stable ← → unstable)\n", analysis.timing_stable_unstable));
    output.push_str(&format!("  Articulation: {:.2}/{:.2} (short/long, soft/hard)\n",
        analysis.articulation_short_long, analysis.articulation_soft_hard));
    output.push_str(&format!("  Pedal: {:.2}/{:.2} (sparse/saturated, clean/blurred)\n",
        analysis.pedal_sparse_saturated, analysis.pedal_clean_blurred));
    output.push_str(&format!("  Dynamics: {:.2}/{:.2} (sophisticated/raw, little/large range)\n",
        analysis.dynamic_sophisticated_raw, analysis.dynamic_range_little_large));
    output.push_str(&format!("  Interpretation: {:.2} (unsatisfactory ← → convincing)\n",
        analysis.interpretation_unsatisfactory_convincing));

    // Temporal analysis if available
    if let Some(temporal) = &response.temporal_analysis {
        output.push_str(&format!("\nTemporal Analysis ({} segments):\n", temporal.len()));
        for segment in temporal {
            output.push_str(&format!("  [{}]: ", segment.timestamp));
            if !segment.notable_features.is_empty() {
                output.push_str(&segment.notable_features.join(", "));
            } else {
                output.push_str("No notable features");
            }
            output.push_str("\n");
        }
    }

    output
}

/// Format a GetUserContextResponse as a tool result string
pub fn format_user_context_response(response: &GetUserContextResponse) -> String {
    let mut output = format!("User Context for Session {}\n\n", response.session_id);
    let context = &response.user_context;

    if !context.goals.is_empty() {
        output.push_str("Goals:\n");
        for goal in &context.goals {
            output.push_str(&format!("  - {}\n", goal));
        }
        output.push_str("\n");
    }

    if !context.constraints.is_empty() {
        output.push_str("Constraints:\n");
        for constraint in &context.constraints {
            output.push_str(&format!("  - {}\n", constraint));
        }
        output.push_str("\n");
    }

    if !context.repertoire.is_empty() {
        output.push_str("Current Repertoire:\n");
        for piece in &context.repertoire {
            output.push_str(&format!("  - {}\n", piece));
        }
        output.push_str("\n");
    }

    if let Some(skill_level) = &context.skill_level {
        output.push_str(&format!("Skill Level: {}\n", skill_level));
    }

    if let Some(learning_style) = &context.learning_style {
        output.push_str(&format!("Learning Style: {}\n", learning_style));
    }

    output
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse tool arguments from JSON string
pub fn parse_tool_arguments<T: for<'de> Deserialize<'de>>(args_json: &str) -> Result<T, String> {
    serde_json::from_str(args_json)
        .map_err(|e| format!("Failed to parse tool arguments: {}", e))
}

/// Validate search query
pub fn validate_search_query(query: &str) -> Result<(), String> {
    if query.trim().is_empty() {
        return Err("Search query cannot be empty".to_string());
    }

    if query.len() > 500 {
        return Err("Search query too long (max 500 characters)".to_string());
    }

    Ok(())
}

/// Validate recording ID format
pub fn validate_recording_id(recording_id: &str) -> Result<(), String> {
    if recording_id.trim().is_empty() {
        return Err("Recording ID cannot be empty".to_string());
    }

    // Basic UUID format validation (optional, can be more strict)
    if recording_id.len() != 36 && recording_id.len() != 32 {
        return Err("Invalid recording ID format".to_string());
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_knowledge_base_tool_schema() {
        let tool = search_knowledge_base_tool();
        assert_eq!(tool.function.name, "search_knowledge_base");
        assert!(tool.function.description.contains("piano pedagogy"));

        // Verify parameters schema
        let params = tool.function.parameters;
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["query"].is_object());
        assert!(params["required"].as_array().unwrap().contains(&json!("query")));
    }

    #[test]
    fn test_get_performance_analysis_tool_schema() {
        let tool = get_performance_analysis_tool();
        assert_eq!(tool.function.name, "get_performance_analysis");
        assert!(tool.function.description.contains("16-dimensional"));
    }

    #[test]
    fn test_get_user_context_tool_schema() {
        let tool = get_user_context_tool();
        assert_eq!(tool.function.name, "get_user_context");
        assert!(tool.function.description.contains("learning context"));
    }

    #[test]
    fn test_get_all_rag_tools() {
        let tools = get_all_rag_tools();
        assert_eq!(tools.len(), 3);
        assert_eq!(tools[0].function.name, "search_knowledge_base");
        assert_eq!(tools[1].function.name, "get_performance_analysis");
        assert_eq!(tools[2].function.name, "get_user_context");
    }

    #[test]
    fn test_parse_search_request() {
        let json = r#"{"query": "staccato techniques", "top_k": 3}"#;
        let req: SearchKnowledgeBaseRequest = parse_tool_arguments(json).unwrap();
        assert_eq!(req.query, "staccato techniques");
        assert_eq!(req.top_k, 3);
    }

    #[test]
    fn test_validate_search_query() {
        assert!(validate_search_query("pedaling techniques").is_ok());
        assert!(validate_search_query("").is_err());
        assert!(validate_search_query("   ").is_err());

        let long_query = "a".repeat(501);
        assert!(validate_search_query(&long_query).is_err());
    }

    #[test]
    fn test_validate_recording_id() {
        assert!(validate_recording_id("550e8400-e29b-41d4-a716-446655440000").is_ok());
        assert!(validate_recording_id("550e8400e29b41d4a716446655440000").is_ok());
        assert!(validate_recording_id("").is_err());
        assert!(validate_recording_id("invalid").is_err());
    }

    #[test]
    fn test_format_search_response() {
        let response = SearchKnowledgeBaseResponse {
            results: vec![
                SearchResult {
                    id: "1".to_string(),
                    score: 0.95,
                    content: "Staccato articulation requires precise finger control...".to_string(),
                    source: SourceMetadata {
                        title: "Piano Technique".to_string(),
                        author: Some("Smith, J.".to_string()),
                        page: Some(42),
                        section: None,
                    },
                    highlights: None,
                }
            ],
            total_results: 1,
            query: "staccato techniques".to_string(),
        };

        let formatted = format_search_response(&response);
        assert!(formatted.contains("staccato techniques"));
        assert!(formatted.contains("Piano Technique"));
        assert!(formatted.contains("Smith, J."));
        assert!(formatted.contains("p. 42"));
    }

    #[test]
    fn test_format_empty_search_response() {
        let response = SearchKnowledgeBaseResponse {
            results: vec![],
            total_results: 0,
            query: "nonexistent query".to_string(),
        };

        let formatted = format_search_response(&response);
        assert!(formatted.contains("No results found"));
    }
}
