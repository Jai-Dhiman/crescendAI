use serde::{Deserialize, Serialize};
use uuid::Uuid;
use worker::*;

use crate::tutor::{TutorCitation, TutorFeedback, TutorRecommendation, UserContext};
use crate::AnalysisData;

use super::ace_curator::{CuratorInput, SessionOutcome, TutorCurator};
use super::ace_framework::{
    extract_weakest_dimensions, generate_context_tags, AceAgent, AceConfig, FeedbackContext,
    PianoPlaybook,
};
use super::ace_generator::{FeedbackGenerator, GeneratorInput};
use super::ace_reflector::{FeedbackReflector, ReflectorInput};

/// Complete ACE pipeline output
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AcePipelineOutput {
    pub feedback: TutorFeedback,
    pub session_id: String,
    pub pipeline_metadata: PipelineMetadata,
}

/// Metadata about the ACE pipeline execution
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PipelineMetadata {
    pub generator_confidence: f32,
    pub reflector_confidence: f32,
    pub quality_scores: serde_json::Value,
    pub playbook_version: u32,
    pub applied_deltas_count: usize,
    pub processing_time_ms: u64,
    pub cache_hit: bool,
}

/// The main ACE pipeline orchestrator
pub struct AcePipeline {
    generator: FeedbackGenerator,
    reflector: FeedbackReflector,
    curator: TutorCurator,
    config: AceConfig,
}

impl AcePipeline {
    pub fn new() -> Self {
        Self {
            generator: FeedbackGenerator::new(),
            reflector: FeedbackReflector::new(),
            curator: TutorCurator::new(),
            config: AceConfig::default(),
        }
    }

    pub fn with_config(config: AceConfig) -> Self {
        Self {
            generator: FeedbackGenerator::with_config(0.3, 600),
            reflector: FeedbackReflector::with_config(0.2, 800),
            curator: TutorCurator::with_config(
                config.confidence_threshold,
                config.max_playbook_size,
            ),
            config,
        }
    }

    /// Main entry point for the ACE pipeline
    pub async fn generate_ace_feedback(
        &self,
        env: &Env,
        analysis: &AnalysisData,
        user_context: &UserContext,
        retrieval_k: Option<usize>,
        previous_session_outcome: Option<SessionOutcome>,
    ) -> Result<AcePipelineOutput> {
        let start_time = js_sys::Date::now() as u64;
        let session_id = Uuid::new_v4().to_string();

        // Build feedback context
        let feedback_context = FeedbackContext {
            performance_scores: analysis.clone(),
            user_context: user_context.clone(),
            previous_sessions: vec![], // Could be populated from storage
            session_id: session_id.clone(),
        };

        // Check cache first
        let cache_key = self.build_cache_key(analysis, user_context, retrieval_k.unwrap_or(3));
        let cached_result = self.check_cache(env, &cache_key).await;

        if let Some(cached) = cached_result {
            console_log!("ACE pipeline cache hit for session {}", session_id);
            return Ok(cached);
        }

        // Load or initialize playbook
        let playbook = self.load_playbook(env).await.unwrap_or_else(|_| {
            console_log!("Initializing new playbook");
            PianoPlaybook::new()
        });

        console_log!(
            "Starting ACE pipeline for session {} with playbook version {}",
            session_id,
            playbook.version
        );

        // Step 1: Generator - produce initial feedback
        let generator_input = GeneratorInput {
            feedback_context: feedback_context.clone(),
            playbook: playbook.clone(),
            retrieval_k: retrieval_k.unwrap_or(3),
        };

        let initial_feedback = self
            .generator
            .process(env, generator_input)
            .await
            .map_err(|e| worker::Error::RustError(format!("Generator failed: {}", e)))?;

        console_log!(
            "Generator produced {} recommendations",
            initial_feedback.recommendations.len()
        );

        // Step 2: Reflector - analyze and critique feedback
        let reflector_input = ReflectorInput {
            initial_feedback: initial_feedback.clone(),
            feedback_context: feedback_context.clone(),
            previous_reflections: vec![], // Could be populated from storage
        };

        let reflection = self
            .reflector
            .process(env, reflector_input)
            .await
            .map_err(|e| worker::Error::RustError(format!("Reflector failed: {}", e)))?;

        console_log!(
            "Reflector generated {} deltas with confidence {:.2}",
            reflection.deltas.len(),
            reflection.confidence_score
        );

        // Step 3: Curator - update playbook (if enabled)
        let curator_output = if self.config.curator_enabled {
            let curator_input = CuratorInput {
                reflection: reflection.clone(),
                current_playbook: playbook,
                session_outcome: previous_session_outcome,
            };

            let output = self
                .curator
                .process(env, curator_input)
                .await
                .map_err(|e| worker::Error::RustError(format!("Curator failed: {}", e)))?;

            console_log!(
                "Curator applied {} deltas, pruned {} bullets",
                output.applied_deltas.len(),
                output.pruned_bullets.len()
            );

            // Save updated playbook
            if let Err(e) = self.save_playbook(env, &output.updated_playbook).await {
                console_log!("Failed to save playbook: {:?}", e);
            }

            Some(output)
        } else {
            None
        };

        // Convert to TutorFeedback format
        let tutor_feedback =
            self.convert_to_tutor_feedback(&initial_feedback, &reflection, user_context);

        // Build pipeline metadata
        let processing_time_ms = js_sys::Date::now() as u64 - start_time;
        let pipeline_metadata = PipelineMetadata {
            generator_confidence: self.calculate_generator_confidence(&initial_feedback),
            reflector_confidence: reflection.confidence_score,
            quality_scores: serde_json::json!(reflection.quality_assessment),
            playbook_version: curator_output
                .as_ref()
                .map(|c| c.updated_playbook.version)
                .unwrap_or(0),
            applied_deltas_count: curator_output
                .as_ref()
                .map(|c| c.applied_deltas.len())
                .unwrap_or(0),
            processing_time_ms,
            cache_hit: false,
        };

        let output = AcePipelineOutput {
            feedback: tutor_feedback,
            session_id,
            pipeline_metadata,
        };

        // Cache the result
        if let Err(e) = self.cache_result(env, &cache_key, &output).await {
            console_log!("Failed to cache ACE result: {:?}", e);
        }

        console_log!("ACE pipeline completed in {}ms", processing_time_ms);
        Ok(output)
    }

    /// Convert ACE output to TutorFeedback format for API compatibility
    fn convert_to_tutor_feedback(
        &self,
        initial_feedback: &super::ace_framework::InitialFeedback,
        reflection: &super::ace_framework::FeedbackReflection,
        user_context: &UserContext,
    ) -> TutorFeedback {
        // Build recommendations from initial feedback
        let recommendations: Vec<TutorRecommendation> = initial_feedback
            .recommendations
            .iter()
            .enumerate()
            .map(|(i, rec)| {
                let technique_focus = if i < initial_feedback.technique_focus.len() {
                    vec![initial_feedback.technique_focus[i].clone()]
                } else {
                    vec!["general_improvement".to_string()]
                };

                let practice_plan = if i < initial_feedback.practice_suggestions.len() {
                    vec![initial_feedback.practice_suggestions[i].clone()]
                } else {
                    vec!["Practice this recommendation consistently".to_string()]
                };

                // Estimate time based on user's available practice time
                let estimated_time_minutes = std::cmp::min(
                    user_context.practice_time_per_day_minutes / 3, // About 1/3 of practice time
                    30,                                             // Cap at 30 minutes
                )
                .max(5); // Minimum 5 minutes

                TutorRecommendation {
                    title: rec.clone(),
                    detail: format!("Focus on: {}", technique_focus.join(", ")),
                    applies_to: technique_focus,
                    practice_plan,
                    estimated_time_minutes,
                    citations: initial_feedback.citations.clone(),
                }
            })
            .collect();

        // Build citations from initial feedback
        let citations: Vec<TutorCitation> = initial_feedback
            .citations
            .iter()
            .map(|citation_id| TutorCitation {
                id: citation_id.clone(),
                title: format!("Piano Pedagogy Reference: {}", citation_id),
                source: "Knowledge Base".to_string(),
                url: None,
                sections: vec!["practice_techniques".to_string()],
            })
            .collect();

        TutorFeedback {
            recommendations,
            citations,
        }
    }

    /// Calculate generator confidence based on output quality
    fn calculate_generator_confidence(
        &self,
        feedback: &super::ace_framework::InitialFeedback,
    ) -> f32 {
        let mut confidence: f32 = 0.5; // Base confidence

        // Boost confidence if recommendations are present and specific
        if !feedback.recommendations.is_empty() {
            confidence += 0.2;

            // Check for specificity (presence of numbers, specific instructions)
            let has_specific_guidance = feedback.recommendations.iter().any(|rec| {
                rec.chars().any(|c| c.is_ascii_digit())
                    || rec.to_lowercase().contains("bpm")
                    || rec.to_lowercase().contains("minutes")
                    || rec.to_lowercase().contains("times")
            });

            if has_specific_guidance {
                confidence += 0.1;
            }
        }

        // Boost confidence if citations are present
        if !feedback.citations.is_empty() {
            confidence += 0.1;
        }

        // Boost confidence if reasoning trace is detailed
        if feedback.reasoning_trace.len() > 50 {
            confidence += 0.1;
        }

        confidence.min(0.95)
    }

    /// Build cache key for ACE results
    fn build_cache_key(
        &self,
        analysis: &AnalysisData,
        user_context: &UserContext,
        k: usize,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash the key components
        format!("{:.2}", analysis.timing_stable_unstable).hash(&mut hasher);
        format!("{:.2}", analysis.articulation_short_long).hash(&mut hasher);
        format!("{:.2}", analysis.pedal_clean_blurred).hash(&mut hasher);
        user_context.practice_time_per_day_minutes.hash(&mut hasher);
        user_context.goals.join(",").hash(&mut hasher);
        k.hash(&mut hasher);

        format!("ace_pipeline:{:x}", hasher.finish())
    }

    /// Check cache for existing ACE result
    async fn check_cache(&self, env: &Env, cache_key: &str) -> Option<AcePipelineOutput> {
        if let Ok(kv) = env.kv("CRESCENDAI_METADATA") {
            if let Ok(Some(cached_json)) = kv.get(cache_key).text().await {
                if let Ok(mut cached) = serde_json::from_str::<AcePipelineOutput>(&cached_json) {
                    cached.pipeline_metadata.cache_hit = true;
                    return Some(cached);
                }
            }
        }
        None
    }

    /// Cache ACE pipeline result
    async fn cache_result(
        &self,
        env: &Env,
        cache_key: &str,
        output: &AcePipelineOutput,
    ) -> Result<()> {
        if let Ok(kv) = env.kv("CRESCENDAI_METADATA") {
            if let Ok(json) = serde_json::to_string(output) {
                let ttl_seconds = self.config.cache_ttl_hours * 3600;
                let _ = kv
                    .put(cache_key, &json)?
                    .expiration_ttl(ttl_seconds as u64)
                    .execute()
                    .await;
            }
        }
        Ok(())
    }

    /// Load playbook from persistent storage
    async fn load_playbook(&self, env: &Env) -> Result<PianoPlaybook> {
        if let Ok(kv) = env.kv("CRESCENDAI_METADATA") {
            if let Ok(Some(playbook_json)) = kv.get("ace_playbook").text().await {
                if let Ok(playbook) = serde_json::from_str::<PianoPlaybook>(&playbook_json) {
                    console_log!(
                        "Loaded playbook version {} with {} bullets",
                        playbook.version,
                        playbook.bullets.len()
                    );
                    return Ok(playbook);
                }
            }
        }
        Err(worker::Error::RustError("Playbook not found".to_string()))
    }

    /// Save playbook to persistent storage
    async fn save_playbook(&self, env: &Env, playbook: &PianoPlaybook) -> Result<()> {
        if let Ok(kv) = env.kv("CRESCENDAI_METADATA") {
            if let Ok(json) = serde_json::to_string(playbook) {
                let _ = kv.put("ace_playbook", &json)?.execute().await;
                console_log!(
                    "Saved playbook version {} with {} bullets",
                    playbook.version,
                    playbook.bullets.len()
                );
            }
        }
        Ok(())
    }

    /// Record session outcome for future learning
    pub async fn record_session_outcome(
        &self,
        env: &Env,
        session_outcome: SessionOutcome,
    ) -> Result<()> {
        // Store the session outcome
        if let Ok(kv) = env.kv("CRESCENDAI_METADATA") {
            let outcome_key = format!("session_outcome:{}", session_outcome.session_id);
            if let Ok(json) = serde_json::to_string(&session_outcome) {
                let _ = kv
                    .put(&outcome_key, &json)?
                    .expiration_ttl(7 * 24 * 3600) // Keep for 7 days
                    .execute()
                    .await;
            }
        }

        // Trigger playbook update with this outcome
        if self.config.curator_enabled {
            // Load current playbook
            if let Ok(playbook) = self.load_playbook(env).await {
                // Create a minimal reflection for the curator
                let minimal_reflection = super::ace_framework::FeedbackReflection {
                    quality_assessment: super::ace_framework::QualityAssessment {
                        actionability_score: 0.7,
                        citation_accuracy_score: 0.7,
                        user_alignment_score: 0.7,
                        completeness_score: 0.7,
                        overall_score: 0.7,
                    },
                    extracted_insights: vec![format!(
                        "Session outcome recorded: improvement={:.2}, satisfaction={:.2}",
                        session_outcome.user_improvement_score,
                        session_outcome.user_satisfaction_score
                    )],
                    improvement_suggestions: vec![],
                    confidence_score: 0.6,
                    deltas: vec![], // No new deltas, just updating statistics
                };

                let curator_input = CuratorInput {
                    reflection: minimal_reflection,
                    current_playbook: playbook,
                    session_outcome: Some(session_outcome),
                };

                if let Ok(output) = self.curator.process(env, curator_input).await {
                    let _ = self.save_playbook(env, &output.updated_playbook).await;
                    console_log!("Updated playbook with session outcome feedback");
                }
            }
        }

        Ok(())
    }
}

impl Default for AcePipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tutor::RepertoireInfo;

    fn create_test_analysis() -> AnalysisData {
        AnalysisData {
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
        }
    }

    fn create_test_user_context() -> UserContext {
        UserContext {
            goals: vec![
                "Improve timing".to_string(),
                "Better articulation".to_string(),
            ],
            practice_time_per_day_minutes: 45,
            constraints: vec!["Limited practice space".to_string()],
            repertoire_info: Some(RepertoireInfo {
                composer: "Bach".to_string(),
                piece: "Invention No. 1".to_string(),
                difficulty: Some(4),
            }),
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = AcePipeline::new();
        assert_eq!(pipeline.generator.agent_name(), "ACE_Generator");
        assert_eq!(pipeline.reflector.agent_name(), "ACE_Reflector");
        assert_eq!(pipeline.curator.agent_name(), "ACE_Curator");
    }

    #[test]
    fn test_cache_key_generation() {
        let pipeline = AcePipeline::new();
        let analysis = create_test_analysis();
        let user_context = create_test_user_context();

        let key1 = pipeline.build_cache_key(&analysis, &user_context, 3);
        let key2 = pipeline.build_cache_key(&analysis, &user_context, 3);
        let key3 = pipeline.build_cache_key(&analysis, &user_context, 5);

        // Same inputs should produce same key
        assert_eq!(key1, key2);

        // Different k should produce different key
        assert_ne!(key1, key3);

        assert!(key1.starts_with("ace_pipeline:"));
    }

    #[test]
    fn test_calculate_generator_confidence() {
        let pipeline = AcePipeline::new();

        // Test with minimal feedback
        let minimal_feedback = super::ace_framework::InitialFeedback {
            recommendations: vec![],
            technique_focus: vec![],
            practice_suggestions: vec![],
            expected_timeline: "".to_string(),
            citations: vec![],
            reasoning_trace: "".to_string(),
        };

        let confidence1 = pipeline.calculate_generator_confidence(&minimal_feedback);
        assert_eq!(confidence1, 0.5); // Base confidence

        // Test with comprehensive feedback
        let comprehensive_feedback = super::ace_framework::InitialFeedback {
            recommendations: vec![
                "Practice at 60 BPM for 10 minutes daily".to_string(),
                "Focus on left hand timing".to_string(),
            ],
            technique_focus: vec!["timing".to_string()],
            practice_suggestions: vec!["Use metronome".to_string()],
            expected_timeline: "2-3 weeks".to_string(),
            citations: vec!["metronome_guide".to_string()],
            reasoning_trace: "Timing issues identified from low scores in timing stability dimension, recommending structured metronome practice".to_string(),
        };

        let confidence2 = pipeline.calculate_generator_confidence(&comprehensive_feedback);
        assert!(confidence2 > 0.8); // Should be high confidence
    }

    #[test]
    fn test_convert_to_tutor_feedback() {
        let pipeline = AcePipeline::new();
        let user_context = create_test_user_context();

        let initial_feedback = super::ace_framework::InitialFeedback {
            recommendations: vec![
                "Practice scales with metronome".to_string(),
                "Focus on finger independence".to_string(),
            ],
            technique_focus: vec!["timing".to_string(), "articulation".to_string()],
            practice_suggestions: vec![
                "Start at 60 BPM".to_string(),
                "Practice hands separately first".to_string(),
            ],
            expected_timeline: "2-3 weeks".to_string(),
            citations: vec![
                "scales_guide".to_string(),
                "articulation_exercises".to_string(),
            ],
            reasoning_trace: "Analysis complete".to_string(),
        };

        let reflection = super::ace_framework::FeedbackReflection {
            quality_assessment: super::ace_framework::QualityAssessment {
                actionability_score: 0.8,
                citation_accuracy_score: 0.7,
                user_alignment_score: 0.9,
                completeness_score: 0.8,
                overall_score: 0.8,
            },
            extracted_insights: vec!["Good feedback".to_string()],
            improvement_suggestions: vec![],
            confidence_score: 0.8,
            deltas: vec![],
        };

        let tutor_feedback =
            pipeline.convert_to_tutor_feedback(&initial_feedback, &reflection, &user_context);

        assert_eq!(tutor_feedback.recommendations.len(), 2);
        assert_eq!(tutor_feedback.citations.len(), 2);

        // Check that practice time is reasonable
        for rec in &tutor_feedback.recommendations {
            assert!(rec.estimated_time_minutes >= 5);
            assert!(rec.estimated_time_minutes <= 30);
        }
    }
}
