use serde::{Deserialize, Serialize};
use worker::*;
use chrono::Utc;

use super::ace_framework::{
    AceAgent, AceError, ContextDelta, FeedbackReflection, PianoPlaybook,
    Operation, PlaybookBullet,
};

/// Input for the Curator agent
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CuratorInput {
    pub reflection: FeedbackReflection,
    pub current_playbook: PianoPlaybook,
    pub session_outcome: Option<SessionOutcome>, // Feedback on how well the advice worked
}

/// Outcome data for a feedback session
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SessionOutcome {
    pub session_id: String,
    pub user_improvement_score: f32, // 0-1, how much user improved
    pub user_satisfaction_score: f32, // 0-1, how satisfied user was
    pub followed_recommendations: Vec<String>, // Which recommendations were followed
    pub success_indicators: Vec<String>, // What worked well
    pub failure_indicators: Vec<String>, // What didn't work
}

/// Output from the Curator agent
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CuratorOutput {
    pub updated_playbook: PianoPlaybook,
    pub applied_deltas: Vec<ContextDelta>,
    pub pruned_bullets: Vec<String>, // IDs of bullets that were removed
    pub refinement_summary: String,
}

/// The Curator agent that manages playbook evolution
pub struct TutorCurator {
    pub confidence_threshold: f32,
    pub max_playbook_size: usize,
    pub deduplication_threshold: f32, // Similarity threshold for deduplication
}

impl TutorCurator {
    pub fn new() -> Self {
        Self {
            confidence_threshold: 0.5,
            max_playbook_size: 500,
            deduplication_threshold: 0.85,
        }
    }

    pub fn with_config(confidence_threshold: f32, max_playbook_size: usize) -> Self {
        Self {
            confidence_threshold,
            max_playbook_size,
            deduplication_threshold: 0.85,
        }
    }

    /// Apply deltas to the playbook
    fn apply_deltas(&self, playbook: &mut PianoPlaybook, deltas: &[ContextDelta]) -> Result<Vec<ContextDelta>> {
        let mut applied_deltas = Vec::new();

        for delta in deltas {
            // Skip deltas with very low confidence
            if delta.confidence < self.confidence_threshold {
                console_log!("Skipping low-confidence delta: {} (confidence: {:.2})", 
                           delta.content, delta.confidence);
                continue;
            }

            match delta.operation {
                Operation::Add => {
                    let bullet_id = playbook.add_bullet(delta.clone());
                    console_log!("Added bullet {}: {}", bullet_id, delta.content);
                    applied_deltas.push(delta.clone());
                }
                Operation::Update => {
                    if let Some(bullet_id) = &delta.bullet_id {
                        match playbook.update_bullet(bullet_id, delta.clone()) {
                            Ok(_) => {
                                console_log!("Updated bullet {}: {}", bullet_id, delta.content);
                                applied_deltas.push(delta.clone());
                            }
                            Err(e) => {
                                console_log!("Failed to update bullet {}: {}", bullet_id, e);
                            }
                        }
                    }
                }
                Operation::Remove => {
                    if let Some(bullet_id) = &delta.bullet_id {
                        if playbook.remove_bullet(bullet_id) {
                            console_log!("Removed bullet {}", bullet_id);
                            applied_deltas.push(delta.clone());
                        } else {
                            console_log!("Failed to remove bullet {}: not found", bullet_id);
                        }
                    }
                }
            }
        }

        playbook.total_feedback_count += 1;
        Ok(applied_deltas)
    }

    /// Update bullet statistics based on session outcome
    fn update_bullet_statistics(&self, 
        playbook: &mut PianoPlaybook, 
        outcome: &SessionOutcome
    ) -> Result<()> {
        // Find bullets that were referenced in followed recommendations
        for recommendation in &outcome.followed_recommendations {
            // Simple keyword matching - could be improved with embeddings
            let matching_bullets: Vec<String> = playbook.bullets
                .iter()
                .filter(|(_, bullet)| {
                    recommendation.to_lowercase().contains(&bullet.content.to_lowercase()) ||
                    bullet.content.to_lowercase().contains(&recommendation.to_lowercase())
                })
                .map(|(id, _)| id.clone())
                .collect();

            for bullet_id in matching_bullets {
                if let Some(bullet) = playbook.bullets.get_mut(&bullet_id) {
                    // Update statistics based on user improvement
                    if outcome.user_improvement_score > 0.6 {
                        bullet.helpful_count += 1;
                        // Increase confidence for successful bullets
                        bullet.confidence = (bullet.confidence + 0.1).min(0.95);
                    } else if outcome.user_improvement_score < 0.4 {
                        bullet.harmful_count += 1;
                        // Decrease confidence for unsuccessful bullets
                        bullet.confidence = (bullet.confidence - 0.05).max(0.3);
                    }
                    
                    bullet.last_used = Utc::now();
                    
                    console_log!("Updated bullet {} stats: helpful={}, harmful={}, confidence={:.2}", 
                               bullet_id, bullet.helpful_count, bullet.harmful_count, bullet.confidence);
                }
            }
        }

        Ok(())
    }

    /// Remove low-confidence and redundant bullets
    fn prune_playbook(&self, playbook: &mut PianoPlaybook) -> Vec<String> {
        let mut pruned_ids = Vec::new();

        // Collect bullets to remove (low confidence, stale, or redundant)
        let mut bullets_to_remove = Vec::new();

        for (id, bullet) in &playbook.bullets {
            // Remove bullets with very low confidence
            if bullet.confidence < 0.3 {
                bullets_to_remove.push(id.clone());
                continue;
            }

            // Remove bullets that have been consistently harmful
            if bullet.harmful_count > bullet.helpful_count + 2 && bullet.harmful_count > 3 {
                bullets_to_remove.push(id.clone());
                continue;
            }

            // Remove very old bullets that haven't been used recently
            let days_since_used = (Utc::now() - bullet.last_used).num_days();
            if days_since_used > 90 && bullet.helpful_count == 0 {
                bullets_to_remove.push(id.clone());
                continue;
            }
        }

        // Remove identified bullets
        for id in bullets_to_remove {
            if playbook.remove_bullet(&id) {
                pruned_ids.push(id);
            }
        }

        // If still too large, remove least useful bullets
        if playbook.bullets.len() > self.max_playbook_size {
            let mut bullet_scores: Vec<(String, f32)> = playbook.bullets
                .iter()
                .map(|(id, bullet)| {
                    let usage_score = bullet.helpful_count as f32 / (bullet.helpful_count + bullet.harmful_count + 1) as f32;
                    let recency_score = 1.0 / (1.0 + (Utc::now() - bullet.last_used).num_days() as f32 / 30.0);
                    let overall_score = bullet.confidence * 0.4 + usage_score * 0.4 + recency_score * 0.2;
                    (id.clone(), overall_score)
                })
                .collect();

            // Sort by score (ascending) and remove lowest scoring bullets
            bullet_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            
            let bullets_to_remove = playbook.bullets.len() - self.max_playbook_size;
            for (id, _) in bullet_scores.into_iter().take(bullets_to_remove) {
                if playbook.remove_bullet(&id) {
                    pruned_ids.push(id);
                }
            }
        }

        console_log!("Pruned {} bullets from playbook", pruned_ids.len());
        pruned_ids
    }

    /// Basic deduplication using content similarity
    fn deduplicate_bullets(&self, playbook: &mut PianoPlaybook) -> Vec<String> {
        let mut removed_ids = Vec::new();
        
        // Simple approach: find bullets with very similar content
        let bullet_list: Vec<(String, String)> = playbook.bullets
            .iter()
            .map(|(id, bullet)| (id.clone(), bullet.content.to_lowercase()))
            .collect();

        let mut to_remove = Vec::new();

        for i in 0..bullet_list.len() {
            for j in i+1..bullet_list.len() {
                let (id1, content1) = &bullet_list[i];
                let (id2, content2) = &bullet_list[j];
                
                // Simple similarity check - could be improved with proper text similarity
                let similarity = self.simple_text_similarity(content1, content2);
                
                if similarity > self.deduplication_threshold {
                    // Keep the bullet with higher confidence
                    let bullet1 = playbook.bullets.get(id1).unwrap();
                    let bullet2 = playbook.bullets.get(id2).unwrap();
                    
                    let id_to_remove = if bullet1.confidence >= bullet2.confidence {
                        id2.clone()
                    } else {
                        id1.clone()
                    };
                    
                    if !to_remove.contains(&id_to_remove) {
                        to_remove.push(id_to_remove);
                    }
                }
            }
        }

        // Remove duplicates
        for id in to_remove {
            if playbook.remove_bullet(&id) {
                removed_ids.push(id);
            }
        }

        console_log!("Deduplicated {} bullets from playbook", removed_ids.len());
        removed_ids
    }

    /// Simple text similarity based on word overlap
    fn simple_text_similarity(&self, text1: &str, text2: &str) -> f32 {
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();
        
        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Generate a summary of refinement actions taken
    fn generate_refinement_summary(&self,
        applied_deltas: &[ContextDelta],
        pruned_bullets: &[String],
        deduplication_count: usize
    ) -> String {
        let adds = applied_deltas.iter().filter(|d| matches!(d.operation, Operation::Add)).count();
        let updates = applied_deltas.iter().filter(|d| matches!(d.operation, Operation::Update)).count();
        let removes = applied_deltas.iter().filter(|d| matches!(d.operation, Operation::Remove)).count();

        format!(
            "Playbook refinement: {} new bullets, {} updated, {} removed via deltas. \
             {} bullets pruned for low performance. {} duplicates removed.",
            adds, updates, removes, pruned_bullets.len(), deduplication_count
        )
    }
}

#[async_trait::async_trait(?Send)]
impl AceAgent for TutorCurator {
    type Input = CuratorInput;
    type Output = CuratorOutput;

    async fn process(&self, _env: &Env, input: CuratorInput) -> Result<CuratorOutput> {
        let mut playbook = input.current_playbook;
        let reflection = input.reflection;

        // Apply deltas from reflection
        let applied_deltas = self.apply_deltas(&mut playbook, &reflection.deltas)
            .map_err(|e| AceError::ValidationError(e.to_string()))?;

        // Update bullet statistics if we have session outcome data
        if let Some(outcome) = &input.session_outcome {
            self.update_bullet_statistics(&mut playbook, outcome)
                .map_err(|e| AceError::ValidationError(e.to_string()))?;
        }

        // Perform playbook maintenance
        let deduplication_removals = self.deduplicate_bullets(&mut playbook);
        let deduplication_count = deduplication_removals.len();
        
        let pruned_bullets = self.prune_playbook(&mut playbook);
        
        // Combine all removed bullets
        let mut all_removed = pruned_bullets.clone();
        all_removed.extend(deduplication_removals);

        // Generate refinement summary
        let refinement_summary = self.generate_refinement_summary(
            &applied_deltas, 
            &all_removed, 
            deduplication_count
        );

        console_log!("{}", refinement_summary);

        Ok(CuratorOutput {
            updated_playbook: playbook,
            applied_deltas,
            pruned_bullets: all_removed,
            refinement_summary,
        })
    }

    fn agent_name(&self) -> &'static str {
        "ACE_Curator"
    }
}

impl Default for TutorCurator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tutor::ace_framework::{QualityAssessment, generate_context_tags};
    use chrono::{Duration};

    fn create_test_playbook() -> PianoPlaybook {
        let mut playbook = PianoPlaybook::new();
        
        // Add some test bullets
        let delta1 = ContextDelta {
            operation: Operation::Add,
            bullet_id: None,
            content: "Practice scales with metronome for timing".to_string(),
            section: "techniques".to_string(),
            tags: vec!["timing".to_string(), "scales".to_string()],
            confidence: 0.8,
            reasoning: "Timing improvement strategy".to_string(),
        };

        let delta2 = ContextDelta {
            operation: Operation::Add,
            bullet_id: None,
            content: "Use slow practice for difficult passages".to_string(),
            section: "strategies".to_string(),
            tags: vec!["difficult_passages".to_string(), "slow_practice".to_string()],
            confidence: 0.7,
            reasoning: "Proven practice strategy".to_string(),
        };

        playbook.add_bullet(delta1);
        playbook.add_bullet(delta2);
        
        playbook
    }

    fn create_test_reflection() -> FeedbackReflection {
        FeedbackReflection {
            quality_assessment: QualityAssessment {
                actionability_score: 0.8,
                citation_accuracy_score: 0.7,
                user_alignment_score: 0.9,
                completeness_score: 0.6,
                overall_score: 0.75,
            },
            extracted_insights: vec![
                "User responded well to metronome practice".to_string(),
                "Timing improvements noted after 1 week".to_string(),
            ],
            improvement_suggestions: vec![
                "Consider adding more specific tempo markings".to_string(),
            ],
            confidence_score: 0.8,
            deltas: vec![
                ContextDelta {
                    operation: Operation::Add,
                    bullet_id: None,
                    content: "For timing issues, start metronome at 50% tempo".to_string(),
                    section: "techniques".to_string(),
                    tags: vec!["timing".to_string(), "metronome".to_string()],
                    confidence: 0.85,
                    reasoning: "Specific tempo guidance improves success rate".to_string(),
                }
            ],
        }
    }

    #[test]
    fn test_curator_creation() {
        let curator = TutorCurator::new();
        assert_eq!(curator.confidence_threshold, 0.5);
        assert_eq!(curator.max_playbook_size, 500);
        assert_eq!(curator.agent_name(), "ACE_Curator");
    }

    #[test]
    fn test_apply_deltas() {
        let curator = TutorCurator::new();
        let mut playbook = create_test_playbook();
        let initial_count = playbook.bullets.len();
        
        let deltas = vec![
            ContextDelta {
                operation: Operation::Add,
                bullet_id: None,
                content: "New practice technique".to_string(),
                section: "techniques".to_string(),
                tags: vec!["new".to_string()],
                confidence: 0.7,
                reasoning: "Testing delta application".to_string(),
            }
        ];

        let applied = curator.apply_deltas(&mut playbook, &deltas).unwrap();
        assert_eq!(applied.len(), 1);
        assert_eq!(playbook.bullets.len(), initial_count + 1);
    }

    #[test]
    fn test_prune_low_confidence_bullets() {
        let curator = TutorCurator::new();
        let mut playbook = PianoPlaybook::new();
        
        // Add a low confidence bullet
        let low_conf_delta = ContextDelta {
            operation: Operation::Add,
            bullet_id: None,
            content: "Low confidence advice".to_string(),
            section: "techniques".to_string(),
            tags: vec![],
            confidence: 0.2, // Very low confidence
            reasoning: "Test".to_string(),
        };
        
        let bullet_id = playbook.add_bullet(low_conf_delta);
        assert_eq!(playbook.bullets.len(), 1);

        // Manually set confidence even lower to trigger pruning
        if let Some(bullet) = playbook.bullets.get_mut(&bullet_id) {
            bullet.confidence = 0.2;
        }

        let pruned = curator.prune_playbook(&mut playbook);
        assert_eq!(pruned.len(), 1);
        assert_eq!(playbook.bullets.len(), 0);
    }

    #[test]
    fn test_simple_text_similarity() {
        let curator = TutorCurator::new();
        
        // Identical texts
        let sim1 = curator.simple_text_similarity("practice scales daily", "practice scales daily");
        assert_eq!(sim1, 1.0);
        
        // Completely different texts
        let sim2 = curator.simple_text_similarity("practice scales", "use metronome");
        assert!(sim2 < 0.5);
        
        // Partially similar texts
        let sim3 = curator.simple_text_similarity("practice scales with metronome", "practice scales daily");
        assert!(sim3 > 0.0 && sim3 < 1.0);
    }

    #[test]
    fn test_update_bullet_statistics() {
        let curator = TutorCurator::new();
        let mut playbook = create_test_playbook();
        
        let outcome = SessionOutcome {
            session_id: "test".to_string(),
            user_improvement_score: 0.8, // Good improvement
            user_satisfaction_score: 0.9,
            followed_recommendations: vec!["practice scales with metronome".to_string()],
            success_indicators: vec!["Better timing".to_string()],
            failure_indicators: vec![],
        };

        // Get initial bullet stats
        let bullet = playbook.bullets.values().next().unwrap().clone();
        let initial_helpful = bullet.helpful_count;
        let initial_confidence = bullet.confidence;

        curator.update_bullet_statistics(&mut playbook, &outcome).unwrap();

        // Check that matching bullet was updated
        let updated_bullet = playbook.bullets.get(&bullet.id).unwrap();
        assert!(updated_bullet.helpful_count > initial_helpful);
        assert!(updated_bullet.confidence >= initial_confidence);
    }

    #[test]
    fn test_generate_refinement_summary() {
        let curator = TutorCurator::new();
        
        let deltas = vec![
            ContextDelta {
                operation: Operation::Add,
                bullet_id: None,
                content: "New bullet".to_string(),
                section: "techniques".to_string(),
                tags: vec![],
                confidence: 0.7,
                reasoning: "Test".to_string(),
            }
        ];
        
        let pruned = vec!["bullet_1".to_string(), "bullet_2".to_string()];
        
        let summary = curator.generate_refinement_summary(&deltas, &pruned, 1);
        assert!(summary.contains("1 new bullets"));
        assert!(summary.contains("2 bullets pruned"));
        assert!(summary.contains("1 duplicates removed"));
    }
}