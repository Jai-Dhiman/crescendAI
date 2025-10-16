use serde::{Deserialize, Serialize};
use worker::*;

use crate::knowledge_base::{query_top_k, KBChunk};
use crate::tutor::{call_llm, UserContext};
use crate::AnalysisData;

use super::ace_framework::{
    AceAgent, AceError, FeedbackContext, InitialFeedback, PianoPlaybook,
    extract_weakest_dimensions, generate_context_tags,
};

/// Input for the Generator agent
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GeneratorInput {
    pub feedback_context: FeedbackContext,
    pub playbook: PianoPlaybook,
    pub retrieval_k: usize,
}

/// The Generator agent that produces initial feedback
pub struct FeedbackGenerator {
    pub temperature: f32,
    pub max_tokens: u32,
}

impl FeedbackGenerator {
    pub fn new() -> Self {
        Self {
            temperature: 0.3,
            max_tokens: 600,
        }
    }

    pub fn with_config(temperature: f32, max_tokens: u32) -> Self {
        Self {
            temperature,
            max_tokens,
        }
    }

    /// Build retrieval query from performance analysis and user context
    fn build_retrieval_query(&self, analysis: &AnalysisData, user_context: &UserContext) -> String {
        let weakest = extract_weakest_dimensions(analysis, 4);
        let weak_dimensions: Vec<String> = weakest.into_iter().map(|(name, _)| name).collect();
        
        let mut query_parts = vec![
            format!("Piano technique problems: {}", weak_dimensions.join(", "))
        ];

        // Add user goals
        if !user_context.goals.is_empty() {
            query_parts.push(format!("Goals: {}", user_context.goals.join(", ")));
        }

        // Add practice time constraints
        if user_context.practice_time_per_day_minutes < 30 {
            query_parts.push("Short practice sessions, efficient exercises".to_string());
        } else if user_context.practice_time_per_day_minutes > 120 {
            query_parts.push("Extended practice, detailed technical work".to_string());
        }

        // Add repertoire context
        if let Some(rep) = &user_context.repertoire_info {
            if !rep.composer.is_empty() || !rep.piece.is_empty() {
                query_parts.push(format!("Repertoire: {} {}", rep.composer, rep.piece));
            }
        }

        query_parts.join("; ")
    }

    /// Get relevant playbook bullets based on context
    fn get_relevant_playbook_content(&self, 
        playbook: &PianoPlaybook, 
        analysis: &AnalysisData,
        user_context: &UserContext
    ) -> Vec<String> {
        let tags = generate_context_tags(analysis, user_context);
        
        // Get bullets from different sections
        let mut relevant_content = Vec::new();
        
        // Techniques and strategies
        let techniques = playbook.get_relevant_bullets(&tags, Some("techniques"));
        let strategies = playbook.get_relevant_bullets(&tags, Some("strategies"));
        let patterns = playbook.get_relevant_bullets(&tags, Some("patterns"));
        
        // Format playbook bullets for the prompt
        for bullet in techniques.into_iter().take(3) {
            relevant_content.push(format!("[Technique] {}", bullet.content));
        }
        
        for bullet in strategies.into_iter().take(3) {
            relevant_content.push(format!("[Strategy] {}", bullet.content));
        }
        
        for bullet in patterns.into_iter().take(2) {
            relevant_content.push(format!("[Pattern] {}", bullet.content));
        }
        
        relevant_content
    }

    /// Build the system prompt for the Generator
    fn build_system_prompt(&self) -> String {
        r#"You are the Generator agent in an ACE (Agentic Context Engineering) piano tutoring system.

Your role is to produce initial feedback recommendations based on:
1. Performance analysis scores (16 dimensions, 0-1 scale where lower = weaker)
2. User context (goals, practice time, repertoire)
3. Retrieved knowledge from piano pedagogy sources
4. Evolving playbook of proven strategies

Return ONLY valid JSON with this schema:
{
  "recommendations": ["string", ...],
  "technique_focus": ["string", ...], 
  "practice_suggestions": ["string", ...],
  "expected_timeline": "string",
  "citations": ["string", ...],
  "reasoning_trace": "string"
}

Guidelines:
- Focus on the 3-4 weakest performance dimensions
- Make recommendations specific and actionable within 24-48 hours
- Include practice time estimates appropriate for user's available time
- Reference retrieval sources and playbook strategies when relevant
- Keep reasoning_trace concise but show your decision process
- Limit to 3-5 recommendations total for focus"#.to_string()
    }

    /// Build the user prompt with all context
    fn build_user_prompt(&self,
        analysis: &AnalysisData,
        user_context: &UserContext,
        kb_chunks: &[KBChunk],
        playbook_content: &[String]
    ) -> String {
        let weakest = extract_weakest_dimensions(analysis, 4);
        
        // Format performance scores compactly
        let performance_summary: Vec<String> = weakest.into_iter()
            .map(|(dim, score)| format!("{}: {:.2}", dim, score))
            .collect();

        // Format knowledge base chunks
        let kb_content: Vec<String> = kb_chunks.iter()
            .map(|chunk| format!("[{}] {} ({})", chunk.source, chunk.title, 
                &chunk.text.chars().take(150).collect::<String>()))
            .collect();

        // Build the complete prompt
        serde_json::json!({
            "performance_analysis": {
                "weakest_dimensions": performance_summary,
                "overall_pattern": self.analyze_performance_pattern(analysis)
            },
            "user_context": {
                "goals": user_context.goals,
                "daily_practice_minutes": user_context.practice_time_per_day_minutes,
                "constraints": user_context.constraints,
                "repertoire": user_context.repertoire_info
            },
            "knowledge_base": kb_content,
            "playbook_strategies": playbook_content,
            "task": "Generate focused, actionable practice recommendations"
        }).to_string()
    }

    /// Analyze overall performance patterns
    fn analyze_performance_pattern(&self, analysis: &AnalysisData) -> String {
        let timing = analysis.timing_stable_unstable;
        let articulation_avg = (analysis.articulation_short_long + analysis.articulation_soft_hard) / 2.0;
        let pedal_avg = (analysis.pedal_sparse_saturated + analysis.pedal_clean_blurred) / 2.0;
        let dynamic_avg = (analysis.dynamic_sophisticated_raw + analysis.dynamic_range_little_large) / 2.0;
        
        if timing < 0.4 {
            "Primary issue: timing instability affecting overall performance".to_string()
        } else if articulation_avg < 0.4 {
            "Primary issue: articulation control needs development".to_string()
        } else if pedal_avg < 0.4 {
            "Primary issue: pedal technique requires attention".to_string()
        } else if dynamic_avg < 0.4 {
            "Primary issue: dynamic control and expression".to_string()
        } else {
            "Good technical foundation, focus on musical refinement".to_string()
        }
    }

    /// Validate and enhance generated feedback
    fn validate_feedback(&self, feedback: &mut InitialFeedback) -> Result<()> {
        // Ensure recommendations are not empty
        if feedback.recommendations.is_empty() {
            feedback.recommendations.push("Focus on slow, deliberate practice".to_string());
        }

        // Ensure technique focus is specified
        if feedback.technique_focus.is_empty() {
            feedback.technique_focus.push("fundamental_technique".to_string());
        }

        // Ensure practice suggestions exist
        if feedback.practice_suggestions.is_empty() {
            feedback.practice_suggestions.push("Practice 15-20 minutes daily with focused attention".to_string());
        }

        // Validate timeline format
        if feedback.expected_timeline.is_empty() {
            feedback.expected_timeline = "1-2 weeks with consistent practice".to_string();
        }

        // Limit lengths for practical use
        feedback.recommendations.truncate(5);
        feedback.technique_focus.truncate(4);
        feedback.practice_suggestions.truncate(6);

        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl AceAgent for FeedbackGenerator {
    type Input = GeneratorInput;
    type Output = InitialFeedback;

    async fn process(&self, env: &Env, input: GeneratorInput) -> Result<InitialFeedback> {
        let context = &input.feedback_context;
        let analysis = &context.performance_scores;
        let user_context = &context.user_context;

        // Build retrieval query
        let retrieval_query = self.build_retrieval_query(analysis, user_context);
        
        // Retrieve knowledge base chunks (with fallback if not configured)
        let kb_chunks = if env.var("CF_ACCOUNT_ID").is_ok() && env.secret("CF_API_TOKEN").is_ok() {
            match query_top_k(env, &retrieval_query, input.retrieval_k).await {
                Ok(chunks) => chunks,
                Err(e) => {
                    console_log!("KB retrieval failed: {:?}", e);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        // Get relevant playbook content
        let playbook_content = self.get_relevant_playbook_content(
            &input.playbook, 
            analysis, 
            user_context
        );

        // Build prompts
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(analysis, user_context, &kb_chunks, &playbook_content);

        // Call LLM to generate feedback
        let llm_response = call_llm(env, &system_prompt, &user_prompt, self.temperature, self.max_tokens)
            .await
            .map_err(|e| AceError::LlmError(e.to_string()))?;

        // Parse JSON response
        let mut feedback: InitialFeedback = serde_json::from_str(&llm_response)
            .map_err(|e| {
                // Try one repair attempt
                let repair_prompt = format!("Return only valid JSON matching the schema. Fix this: {}", llm_response);
                AceError::ParseError(format!("JSON parse failed: {}. Response: {}", e, llm_response))
            })?;

        // If parsing failed, attempt repair
        if feedback.recommendations.is_empty() && feedback.technique_focus.is_empty() {
            let repair_prompt = format!("Return only valid JSON matching the schema. Fix this response: {}", llm_response);
            let repaired_response = call_llm(env, &system_prompt, &repair_prompt, 0.0, self.max_tokens)
                .await
                .map_err(|e| AceError::LlmError(format!("Repair attempt failed: {}", e)))?;
            
            feedback = serde_json::from_str(&repaired_response)
                .map_err(|e| AceError::ParseError(format!("Repair parse failed: {}", e)))?;
        }

        // Validate and enhance feedback
        self.validate_feedback(&mut feedback)
            .map_err(|e| AceError::ValidationError(e.to_string()))?;

        Ok(feedback)
    }

    fn agent_name(&self) -> &'static str {
        "ACE_Generator"
    }
}

impl Default for FeedbackGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tutor::RepertoireInfo;

    #[test]
    fn test_generator_creation() {
        let generator = FeedbackGenerator::new();
        assert_eq!(generator.temperature, 0.3);
        assert_eq!(generator.max_tokens, 600);
        assert_eq!(generator.agent_name(), "ACE_Generator");
    }

    #[test]
    fn test_build_retrieval_query() {
        let generator = FeedbackGenerator::new();
        
        let analysis = AnalysisData {
            timing_stable_unstable: 0.2,
            articulation_short_long: 0.3,
            // Fill other fields with higher values
            articulation_soft_hard: 0.6,
            pedal_sparse_saturated: 0.7,
            pedal_clean_blurred: 0.8,
            timbre_even_colorful: 0.5,
            timbre_shallow_rich: 0.5,
            timbre_bright_dark: 0.5,
            timbre_soft_loud: 0.5,
            dynamic_sophisticated_raw: 0.5,
            dynamic_range_little_large: 0.5,
            music_making_fast_slow: 0.5,
            music_making_flat_spacious: 0.5,
            music_making_disproportioned_balanced: 0.5,
            music_making_pure_dramatic: 0.5,
            emotion_mood_optimistic_dark: 0.5,
            emotion_mood_low_high_energy: 0.5,
            emotion_mood_honest_imaginative: 0.5,
            interpretation_unsatisfactory_convincing: 0.5,
        };

        let user_context = UserContext {
            goals: vec!["Improve timing".to_string()],
            practice_time_per_day_minutes: 30,
            constraints: vec![],
            repertoire_info: Some(RepertoireInfo {
                composer: "Bach".to_string(),
                piece: "Invention No. 1".to_string(),
                difficulty: Some(4),
            }),
        };

        let query = generator.build_retrieval_query(&analysis, &user_context);
        assert!(query.contains("timing_stability"));
        assert!(query.contains("articulation_length"));
        assert!(query.contains("Improve timing"));
        assert!(query.contains("Bach"));
    }

    #[test]
    fn test_analyze_performance_pattern() {
        let generator = FeedbackGenerator::new();
        
        let timing_issue = AnalysisData {
            timing_stable_unstable: 0.2,
            articulation_short_long: 0.6,
            articulation_soft_hard: 0.6,
            pedal_sparse_saturated: 0.6,
            pedal_clean_blurred: 0.6,
            // Fill other fields
            timbre_even_colorful: 0.5,
            timbre_shallow_rich: 0.5,
            timbre_bright_dark: 0.5,
            timbre_soft_loud: 0.5,
            dynamic_sophisticated_raw: 0.5,
            dynamic_range_little_large: 0.5,
            music_making_fast_slow: 0.5,
            music_making_flat_spacious: 0.5,
            music_making_disproportioned_balanced: 0.5,
            music_making_pure_dramatic: 0.5,
            emotion_mood_optimistic_dark: 0.5,
            emotion_mood_low_high_energy: 0.5,
            emotion_mood_honest_imaginative: 0.5,
            interpretation_unsatisfactory_convincing: 0.5,
        };

        let pattern = generator.analyze_performance_pattern(&timing_issue);
        assert!(pattern.contains("timing"));
    }

    #[test]
    fn test_validate_feedback() {
        let generator = FeedbackGenerator::new();
        
        let mut empty_feedback = InitialFeedback {
            recommendations: vec![],
            technique_focus: vec![],
            practice_suggestions: vec![],
            expected_timeline: "".to_string(),
            citations: vec![],
            reasoning_trace: "".to_string(),
        };

        generator.validate_feedback(&mut empty_feedback).unwrap();
        
        assert!(!empty_feedback.recommendations.is_empty());
        assert!(!empty_feedback.technique_focus.is_empty());
        assert!(!empty_feedback.practice_suggestions.is_empty());
        assert!(!empty_feedback.expected_timeline.is_empty());
    }
}