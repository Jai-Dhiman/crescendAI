use serde::{Deserialize, Serialize};
use worker::*;

use crate::tutor::call_llm;

use super::ace_framework::{
    AceAgent, AceError, ContextDelta, FeedbackContext, FeedbackReflection, 
    InitialFeedback, Operation, QualityAssessment,
};

/// Input for the Reflector agent
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReflectorInput {
    pub initial_feedback: InitialFeedback,
    pub feedback_context: FeedbackContext,
    pub previous_reflections: Vec<String>, // IDs of previous reflections for learning
}

/// The Reflector agent that critiques and improves feedback
pub struct FeedbackReflector {
    pub temperature: f32,
    pub max_tokens: u32,
}

impl FeedbackReflector {
    pub fn new() -> Self {
        Self {
            temperature: 0.2, // Lower temperature for more consistent analysis
            max_tokens: 800,
        }
    }

    pub fn with_config(temperature: f32, max_tokens: u32) -> Self {
        Self {
            temperature,
            max_tokens,
        }
    }

    /// Build the system prompt for reflection analysis
    fn build_system_prompt(&self) -> String {
        r#"You are the Reflector agent in an ACE (Agentic Context Engineering) piano tutoring system.

Your role is to critically analyze initial feedback and extract insights for improving future recommendations.

Evaluate the feedback on these dimensions:
1. ACTIONABILITY: How specific and implementable are the recommendations?
2. CITATION_ACCURACY: Do the citations support the claims made?
3. USER_ALIGNMENT: Are recommendations appropriate for the user's context (time, goals, level)?
4. COMPLETENESS: Does it address the main performance issues identified?

Return ONLY valid JSON with this schema:
{
  "quality_assessment": {
    "actionability_score": 0.0-1.0,
    "citation_accuracy_score": 0.0-1.0, 
    "user_alignment_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "overall_score": 0.0-1.0
  },
  "extracted_insights": ["insight1", "insight2", ...],
  "improvement_suggestions": ["suggestion1", "suggestion2", ...],
  "confidence_score": 0.0-1.0,
  "deltas": [
    {
      "operation": "Add|Update|Remove",
      "bullet_id": "optional_string",
      "content": "string",
      "section": "strategies|techniques|patterns|failure_modes",
      "tags": ["tag1", "tag2", ...],
      "confidence": 0.0-1.0,
      "reasoning": "string"
    }
  ]
}

Guidelines:
- Be critical but constructive
- Focus on patterns that could improve future feedback
- Generate deltas that capture learnable insights
- Consider both successes and failures
- Keep confidence scores realistic (0.6-0.9 range)"#.to_string()
    }

    /// Build the user prompt with feedback to analyze
    fn build_user_prompt(&self, input: &ReflectorInput) -> String {
        let feedback = &input.initial_feedback;
        let context = &input.feedback_context;

        serde_json::json!({
            "feedback_to_analyze": {
                "recommendations": feedback.recommendations,
                "technique_focus": feedback.technique_focus,
                "practice_suggestions": feedback.practice_suggestions,
                "expected_timeline": feedback.expected_timeline,
                "citations": feedback.citations,
                "reasoning_trace": feedback.reasoning_trace
            },
            "user_context": {
                "goals": context.user_context.goals,
                "daily_practice_minutes": context.user_context.practice_time_per_day_minutes,
                "constraints": context.user_context.constraints,
                "repertoire": context.user_context.repertoire_info
            },
            "performance_issues": self.extract_main_issues(&context.performance_scores),
            "analysis_questions": [
                "Are the recommendations specific enough to act on within 24-48 hours?",
                "Do the practice time estimates align with user's available time?",
                "Are the weakest performance dimensions adequately addressed?",
                "Do citations actually support the claims made?",
                "What patterns could be learned for similar future cases?"
            ]
        }).to_string()
    }

    /// Extract main performance issues for reflection context
    fn extract_main_issues(&self, analysis: &crate::AnalysisData) -> Vec<String> {
        let mut issues = Vec::new();

        // Check each major category
        if analysis.timing_stable_unstable < 0.5 {
            issues.push(format!("Timing instability (score: {:.2})", analysis.timing_stable_unstable));
        }

        let articulation_avg = (analysis.articulation_short_long + analysis.articulation_soft_hard) / 2.0;
        if articulation_avg < 0.5 {
            issues.push(format!("Articulation control (avg score: {:.2})", articulation_avg));
        }

        let pedal_avg = (analysis.pedal_sparse_saturated + analysis.pedal_clean_blurred) / 2.0;
        if pedal_avg < 0.5 {
            issues.push(format!("Pedal technique (avg score: {:.2})", pedal_avg));
        }

        let dynamic_avg = (analysis.dynamic_sophisticated_raw + analysis.dynamic_range_little_large) / 2.0;
        if dynamic_avg < 0.5 {
            issues.push(format!("Dynamic control (avg score: {:.2})", dynamic_avg));
        }

        let musical_avg = (analysis.music_making_fast_slow + analysis.music_making_flat_spacious + 
                          analysis.music_making_disproportioned_balanced + analysis.music_making_pure_dramatic) / 4.0;
        if musical_avg < 0.5 {
            issues.push(format!("Musical expression (avg score: {:.2})", musical_avg));
        }

        issues
    }

    /// Validate and enhance the reflection output
    fn validate_reflection(&self, reflection: &mut FeedbackReflection) -> Result<()> {
        let assessment = &mut reflection.quality_assessment;

        // Ensure scores are within valid range
        assessment.actionability_score = assessment.actionability_score.clamp(0.0, 1.0);
        assessment.citation_accuracy_score = assessment.citation_accuracy_score.clamp(0.0, 1.0);
        assessment.user_alignment_score = assessment.user_alignment_score.clamp(0.0, 1.0);
        assessment.completeness_score = assessment.completeness_score.clamp(0.0, 1.0);

        // Calculate overall score as weighted average
        assessment.overall_score = (
            assessment.actionability_score * 0.3 +
            assessment.citation_accuracy_score * 0.2 +
            assessment.user_alignment_score * 0.3 +
            assessment.completeness_score * 0.2
        ).clamp(0.0, 1.0);

        // Ensure confidence is reasonable
        reflection.confidence_score = reflection.confidence_score.clamp(0.4, 0.95);

        // Validate deltas
        for delta in &mut reflection.deltas {
            // Ensure section is valid
            if !matches!(delta.section.as_str(), "strategies" | "techniques" | "patterns" | "failure_modes") {
                delta.section = "strategies".to_string();
            }

            // Ensure confidence is reasonable
            delta.confidence = delta.confidence.clamp(0.3, 0.9);

            // Ensure content is not empty
            if delta.content.trim().is_empty() {
                delta.content = "General practice improvement needed".to_string();
            }

            // Limit tag count
            delta.tags.truncate(5);
        }

        // Limit number of insights and suggestions
        reflection.extracted_insights.truncate(8);
        reflection.improvement_suggestions.truncate(6);

        Ok(())
    }

    /// Generate fallback reflection if LLM fails
    fn generate_fallback_reflection(&self, input: &ReflectorInput) -> FeedbackReflection {
        let feedback = &input.initial_feedback;
        
        // Basic quality assessment
        let actionability_score = if feedback.practice_suggestions.is_empty() { 0.3 } else { 0.6 };
        let citation_accuracy_score = if feedback.citations.is_empty() { 0.4 } else { 0.7 };
        let user_alignment_score = 0.6; // Default reasonable score
        let completeness_score = if feedback.recommendations.is_empty() { 0.2 } else { 0.6 };

        let quality_assessment = QualityAssessment {
            actionability_score,
            citation_accuracy_score,
            user_alignment_score,
            completeness_score,
            overall_score: (actionability_score + citation_accuracy_score + 
                           user_alignment_score + completeness_score) / 4.0,
        };

        // Generate basic insights
        let mut extracted_insights = vec![
            "Feedback generation system is operational".to_string(),
        ];

        if feedback.recommendations.is_empty() {
            extracted_insights.push("Recommendation generation needs improvement".to_string());
        }

        if feedback.citations.is_empty() {
            extracted_insights.push("Citation system needs attention".to_string());
        }

        // Generate improvement suggestions
        let improvement_suggestions = vec![
            "Ensure recommendations are specific and actionable".to_string(),
            "Include relevant citations for credibility".to_string(),
            "Align practice time with user constraints".to_string(),
        ];

        // Create a learning delta
        let delta = ContextDelta {
            operation: Operation::Add,
            bullet_id: None,
            content: format!(
                "System generated {} recommendations with {} citations for user with {} minutes daily practice",
                feedback.recommendations.len(),
                feedback.citations.len(),
                input.feedback_context.user_context.practice_time_per_day_minutes
            ),
            section: "patterns".to_string(),
            tags: vec!["system_performance".to_string(), "fallback_reflection".to_string()],
            confidence: 0.5,
            reasoning: "Fallback reflection generated when LLM analysis failed".to_string(),
        };

        FeedbackReflection {
            quality_assessment,
            extracted_insights,
            improvement_suggestions,
            confidence_score: 0.5,
            deltas: vec![delta],
        }
    }
}

#[async_trait::async_trait(?Send)]
impl AceAgent for FeedbackReflector {
    type Input = ReflectorInput;
    type Output = FeedbackReflection;

    async fn process(&self, env: &Env, input: ReflectorInput) -> Result<FeedbackReflection> {
        // Build prompts
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(&input);

        // Call LLM for reflection analysis
        let llm_response = match call_llm(env, &system_prompt, &user_prompt, self.temperature, self.max_tokens).await {
            Ok(response) => response,
            Err(e) => {
                console_log!("Reflector LLM call failed: {:?}, using fallback", e);
                return Ok(self.generate_fallback_reflection(&input));
            }
        };

        // Parse JSON response
        let mut reflection: FeedbackReflection = match serde_json::from_str(&llm_response) {
            Ok(reflection) => reflection,
            Err(e) => {
                // Try one repair attempt
                let repair_prompt = format!(
                    "Return only valid JSON matching the schema. Fix this response: {}", 
                    llm_response
                );
                
                let repaired_response = match call_llm(env, &system_prompt, &repair_prompt, 0.0, self.max_tokens).await {
                    Ok(response) => response,
                    Err(_) => {
                        console_log!("Reflector repair failed, using fallback");
                        return Ok(self.generate_fallback_reflection(&input));
                    }
                };
                
                match serde_json::from_str::<FeedbackReflection>(&repaired_response) {
                    Ok(reflection) => reflection,
                    Err(_) => {
                        console_log!("Reflector repair parse failed, using fallback");
                        return Ok(self.generate_fallback_reflection(&input));
                    }
                }
            }
        };

        // Validate and enhance reflection
        self.validate_reflection(&mut reflection)
            .map_err(|e| AceError::ValidationError(e.to_string()))?;

        Ok(reflection)
    }

    fn agent_name(&self) -> &'static str {
        "ACE_Reflector"
    }
}

impl Default for FeedbackReflector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tutor::{UserContext, RepertoireInfo};
    use crate::AnalysisData;

    fn create_test_input() -> ReflectorInput {
        let feedback_context = FeedbackContext {
            performance_scores: AnalysisData {
                timing_stable_unstable: 0.3,
                articulation_short_long: 0.4,
                articulation_soft_hard: 0.6,
                pedal_sparse_saturated: 0.7,
                pedal_clean_blurred: 0.5,
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
            },
            user_context: UserContext {
                goals: vec!["Improve timing".to_string()],
                practice_time_per_day_minutes: 30,
                constraints: vec![],
                repertoire_info: Some(RepertoireInfo {
                    composer: "Bach".to_string(),
                    piece: "Invention No. 1".to_string(),
                    difficulty: Some(4),
                }),
            },
            previous_sessions: vec![],
            session_id: "test_session".to_string(),
        };

        let initial_feedback = InitialFeedback {
            recommendations: vec![
                "Practice with metronome at slow tempo".to_string(),
                "Focus on steady beat in left hand".to_string(),
            ],
            technique_focus: vec!["timing_stability".to_string()],
            practice_suggestions: vec![
                "Start at 60 BPM, increase by 5 BPM when steady".to_string(),
            ],
            expected_timeline: "2-3 weeks with daily practice".to_string(),
            citations: vec!["metronome_practice_guide".to_string()],
            reasoning_trace: "Timing issues identified, metronome practice recommended".to_string(),
        };

        ReflectorInput {
            initial_feedback,
            feedback_context,
            previous_reflections: vec![],
        }
    }

    #[test]
    fn test_reflector_creation() {
        let reflector = FeedbackReflector::new();
        assert_eq!(reflector.temperature, 0.2);
        assert_eq!(reflector.max_tokens, 800);
        assert_eq!(reflector.agent_name(), "ACE_Reflector");
    }

    #[test]
    fn test_extract_main_issues() {
        let reflector = FeedbackReflector::new();
        let analysis = AnalysisData {
            timing_stable_unstable: 0.3,
            articulation_short_long: 0.2,
            articulation_soft_hard: 0.3,
            // Fill other fields with reasonable values
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

        let issues = reflector.extract_main_issues(&analysis);
        assert!(!issues.is_empty());
        
        // Should identify timing and articulation issues
        assert!(issues.iter().any(|issue| issue.contains("Timing")));
        assert!(issues.iter().any(|issue| issue.contains("Articulation")));
    }

    #[test]
    fn test_generate_fallback_reflection() {
        let reflector = FeedbackReflector::new();
        let input = create_test_input();

        let reflection = reflector.generate_fallback_reflection(&input);

        assert!(!reflection.extracted_insights.is_empty());
        assert!(!reflection.improvement_suggestions.is_empty());
        assert!(!reflection.deltas.is_empty());
        assert!(reflection.confidence_score >= 0.4);
        assert!(reflection.confidence_score <= 0.95);
        assert!(reflection.quality_assessment.overall_score >= 0.0);
        assert!(reflection.quality_assessment.overall_score <= 1.0);
    }

    #[test]
    fn test_validate_reflection() {
        let reflector = FeedbackReflector::new();
        
        let mut reflection = FeedbackReflection {
            quality_assessment: QualityAssessment {
                actionability_score: 1.5, // Invalid - too high
                citation_accuracy_score: -0.1, // Invalid - negative
                user_alignment_score: 0.7,
                completeness_score: 0.8,
                overall_score: 0.0, // Will be recalculated
            },
            extracted_insights: vec!["Test insight".to_string()],
            improvement_suggestions: vec!["Test suggestion".to_string()],
            confidence_score: 1.2, // Invalid - too high
            deltas: vec![ContextDelta {
                operation: Operation::Add,
                bullet_id: None,
                content: "".to_string(), // Invalid - empty
                section: "invalid_section".to_string(), // Invalid section
                tags: vec!["test".to_string()],
                confidence: 1.5, // Invalid - too high
                reasoning: "Test reasoning".to_string(),
            }],
        };

        reflector.validate_reflection(&mut reflection).unwrap();

        // Check that invalid values were corrected
        assert!(reflection.quality_assessment.actionability_score <= 1.0);
        assert!(reflection.quality_assessment.citation_accuracy_score >= 0.0);
        assert!(reflection.confidence_score <= 0.95);
        assert!(!reflection.deltas[0].content.is_empty());
        assert_eq!(reflection.deltas[0].section, "strategies");
        assert!(reflection.deltas[0].confidence <= 0.9);
    }
}