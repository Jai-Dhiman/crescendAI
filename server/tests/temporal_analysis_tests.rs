// Temporal Analysis Tests for CrescendAI Backend
// Tests for chunked audio analysis with natural language insights

#[cfg(test)]
mod temporal_analysis_tests {
    use crescendai_backend::*;
    
    // Test audio chunking functionality
    mod test_audio_chunking {
        use super::*;
        
        #[test]
        fn test_chunk_audio_basic() {
            // Create test audio data (5 seconds at 44100Hz)
            let sample_rate = 44100u32;
            let duration = 5.0;
            let num_samples = (sample_rate as f32 * duration) as usize;
            
            let audio_data = crescendai_backend::audio_dsp::AudioData {
                samples: vec![0.0; num_samples],
                sample_rate,
                channels: 1,
                duration_seconds: duration,
            };
            
            // Chunk with 3s duration and 1s overlap
            let result = crescendai_backend::audio_dsp::chunk_audio_with_overlap(
                &audio_data,
                3.0,
                1.0
            );
            
            assert!(result.is_ok(), "Chunking should succeed");
            let chunks = result.unwrap();
            
            // 5 seconds with 3s chunks and 1s overlap should produce 2 chunks
            // Chunk 0: 0-3s, Chunk 1: 2-5s
            assert_eq!(chunks.len(), 2, "Should create 2 chunks for 5s audio");
            
            // Verify first chunk
            assert_eq!(chunks[0].chunk_index, 0);
            assert!((chunks[0].start_time_secs - 0.0).abs() < 0.01);
            assert!((chunks[0].end_time_secs - 3.0).abs() < 0.01);
            assert_eq!(chunks[0].timestamp, "0:00-0:03");
            
            // Verify second chunk
            assert_eq!(chunks[1].chunk_index, 1);
            assert!((chunks[1].start_time_secs - 2.0).abs() < 0.01);
            assert!((chunks[1].end_time_secs - 5.0).abs() < 0.01);
            assert_eq!(chunks[1].timestamp, "0:02-0:05");
        }
        
        #[test]
        fn test_chunk_audio_10_seconds() {
            let sample_rate = 44100u32;
            let duration = 10.0;
            let num_samples = (sample_rate as f32 * duration) as usize;
            
            let audio_data = crescendai_backend::audio_dsp::AudioData {
                samples: vec![0.0; num_samples],
                sample_rate,
                channels: 1,
                duration_seconds: duration,
            };
            
            let result = crescendai_backend::audio_dsp::chunk_audio_with_overlap(
                &audio_data,
                3.0,
                1.0
            );
            
            assert!(result.is_ok());
            let chunks = result.unwrap();
            
            // 10 seconds with 3s chunks and 1s overlap (2s hop) should produce 5 chunks
            // Chunks at: 0-3, 2-5, 4-7, 6-9, 8-11 (but last one would exceed, so 4 chunks)
            assert!(chunks.len() >= 4 && chunks.len() <= 5, 
                "Should create 4-5 chunks for 10s audio, got {}", chunks.len());
        }
        
        #[test]
        fn test_chunk_audio_30_seconds() {
            let sample_rate = 44100u32;
            let duration = 30.0;
            let num_samples = (sample_rate as f32 * duration) as usize;
            
            let audio_data = crescendai_backend::audio_dsp::AudioData {
                samples: vec![0.0; num_samples],
                sample_rate,
                channels: 1,
                duration_seconds: duration,
            };
            
            let result = crescendai_backend::audio_dsp::chunk_audio_with_overlap(
                &audio_data,
                3.0,
                1.0
            );
            
            assert!(result.is_ok());
            let chunks = result.unwrap();
            
            // 30 seconds should produce approximately 15 chunks
            assert!(chunks.len() >= 14 && chunks.len() <= 15, 
                "Should create ~15 chunks for 30s audio, got {}", chunks.len());
            
            // Verify chunks don't overlap incorrectly
            for i in 1..chunks.len() {
                let prev = &chunks[i - 1];
                let curr = &chunks[i];
                
                // Current chunk should start before previous ends (overlap)
                assert!(curr.start_time_secs < prev.end_time_secs,
                    "Chunks should overlap");
                
                // Overlap should be approximately 1 second
                let overlap = prev.end_time_secs - curr.start_time_secs;
                assert!((overlap - 1.0).abs() < 0.1,
                    "Overlap should be ~1s, got {:.2}s", overlap);
            }
        }
        
        #[test]
        fn test_timestamp_formatting() {
            // Test AudioChunk timestamp creation
            let timestamp1 = crescendai_backend::audio_dsp::AudioChunk::create_timestamp(0.0, 3.0);
            assert_eq!(timestamp1, "0:00-0:03");
            
            let timestamp2 = crescendai_backend::audio_dsp::AudioChunk::create_timestamp(62.5, 65.5);
            assert_eq!(timestamp2, "1:02-1:05");
            
            let timestamp3 = crescendai_backend::audio_dsp::AudioChunk::create_timestamp(125.0, 128.0);
            assert_eq!(timestamp3, "2:05-2:08");
        }
    }
    
    // Test edge cases and error handling
    mod test_edge_cases {
        use super::*;
        
        #[test]
        fn test_audio_too_short() {
            // 2 second audio (less than 3s chunk requirement)
            let sample_rate = 44100u32;
            let duration = 2.0;
            let num_samples = (sample_rate as f32 * duration) as usize;
            
            let audio_data = crescendai_backend::audio_dsp::AudioData {
                samples: vec![0.0; num_samples],
                sample_rate,
                channels: 1,
                duration_seconds: duration,
            };
            
            let result = crescendai_backend::audio_dsp::chunk_audio_with_overlap(
                &audio_data,
                3.0,
                1.0
            );
            
            assert!(result.is_err(), "Should fail with audio shorter than chunk size");
            
            if let Err(e) = result {
                let err_msg = format!("{:?}", e);
                assert!(err_msg.contains("too short") || err_msg.contains("Audio too short"),
                    "Error should mention audio is too short: {}", err_msg);
            }
        }
        
        #[test]
        fn test_invalid_chunk_duration() {
            let sample_rate = 44100u32;
            let duration = 10.0;
            let num_samples = (sample_rate as f32 * duration) as usize;
            
            let audio_data = crescendai_backend::audio_dsp::AudioData {
                samples: vec![0.0; num_samples],
                sample_rate,
                channels: 1,
                duration_seconds: duration,
            };
            
            // Test with zero chunk duration
            let result = crescendai_backend::audio_dsp::chunk_audio_with_overlap(
                &audio_data,
                0.0,
                1.0
            );
            
            assert!(result.is_err(), "Should fail with zero chunk duration");
            
            // Test with negative chunk duration
            let result = crescendai_backend::audio_dsp::chunk_audio_with_overlap(
                &audio_data,
                -1.0,
                1.0
            );
            
            assert!(result.is_err(), "Should fail with negative chunk duration");
        }
        
        #[test]
        fn test_invalid_overlap() {
            let sample_rate = 44100u32;
            let duration = 10.0;
            let num_samples = (sample_rate as f32 * duration) as usize;
            
            let audio_data = crescendai_backend::audio_dsp::AudioData {
                samples: vec![0.0; num_samples],
                sample_rate,
                channels: 1,
                duration_seconds: duration,
            };
            
            // Test with overlap equal to chunk duration
            let result = crescendai_backend::audio_dsp::chunk_audio_with_overlap(
                &audio_data,
                3.0,
                3.0
            );
            
            assert!(result.is_err(), "Should fail with overlap equal to chunk duration");
            
            // Test with overlap greater than chunk duration
            let result = crescendai_backend::audio_dsp::chunk_audio_with_overlap(
                &audio_data,
                3.0,
                4.0
            );
            
            assert!(result.is_err(), "Should fail with overlap greater than chunk duration");
            
            // Test with negative overlap
            let result = crescendai_backend::audio_dsp::chunk_audio_with_overlap(
                &audio_data,
                3.0,
                -1.0
            );
            
            assert!(result.is_err(), "Should fail with negative overlap");
        }
        
        #[test]
        fn test_empty_audio() {
            let audio_data = crescendai_backend::audio_dsp::AudioData {
                samples: vec![],
                sample_rate: 44100,
                channels: 1,
                duration_seconds: 0.0,
            };
            
            let result = crescendai_backend::audio_dsp::chunk_audio_with_overlap(
                &audio_data,
                3.0,
                1.0
            );
            
            assert!(result.is_err(), "Should fail with empty audio");
        }
        
        #[test]
        fn test_invalid_sample_rate() {
            let audio_data = crescendai_backend::audio_dsp::AudioData {
                samples: vec![0.0; 44100],
                sample_rate: 0, // Invalid sample rate
                channels: 1,
                duration_seconds: 1.0,
            };
            
            let result = crescendai_backend::audio_dsp::chunk_audio_with_overlap(
                &audio_data,
                3.0,
                1.0
            );
            
            assert!(result.is_err(), "Should fail with invalid sample rate");
        }
        
        #[test]
        fn test_long_audio_300_seconds() {
            // Test with 5 minute audio to ensure no overflow or performance issues
            let sample_rate = 44100u32;
            let duration = 300.0; // 5 minutes
            let num_samples = (sample_rate as f32 * duration) as usize;
            
            let audio_data = crescendai_backend::audio_dsp::AudioData {
                samples: vec![0.0; num_samples],
                sample_rate,
                channels: 1,
                duration_seconds: duration,
            };
            
            let result = crescendai_backend::audio_dsp::chunk_audio_with_overlap(
                &audio_data,
                3.0,
                1.0
            );
            
            assert!(result.is_ok(), "Should handle long audio");
            let chunks = result.unwrap();
            
            // 300 seconds should produce approximately 150 chunks
            assert!(chunks.len() >= 145 && chunks.len() <= 155, 
                "Should create ~150 chunks for 300s audio, got {}", chunks.len());
        }
    }
    
    // Test data structure serialization
    mod test_data_structures {
        use super::*;
        use serde_json;
        
        #[test]
        fn test_temporal_analysis_result_serialization() {
            let result = TemporalAnalysisResult {
                id: "test-id".to_string(),
                status: "completed".to_string(),
                file_id: "test-file-id".to_string(),
                overall_assessment: OverallAssessment {
                    strengths: vec!["Excellent timing".to_string()],
                    priority_areas: vec!["Work on dynamics".to_string()],
                    performance_character: "A focused performance".to_string(),
                },
                temporal_feedback: vec![
                    TemporalFeedbackItem {
                        timestamp: "0:00-0:03".to_string(),
                        insights: vec![
                            AnalysisInsight {
                                category: "Technical".to_string(),
                                observation: "Stable timing".to_string(),
                                actionable_advice: "Continue practice".to_string(),
                                score_reference: "timing: 0.85".to_string(),
                            }
                        ],
                        practice_focus: "Maintain consistency".to_string(),
                    }
                ],
                practice_recommendations: PracticeRecommendations {
                    immediate_priorities: vec![
                        ImmediatePriority {
                            skill_area: "Timing".to_string(),
                            specific_exercise: "Metronome practice".to_string(),
                            expected_outcome: "Improved consistency".to_string(),
                        }
                    ],
                    long_term_development: vec![
                        LongTermDevelopment {
                            musical_aspect: "Expression".to_string(),
                            development_approach: "Study varied repertoire".to_string(),
                            repertoire_suggestions: "Romantic era pieces".to_string(),
                        }
                    ],
                },
                encouragement: "Great work!".to_string(),
                created_at: "2024-01-01T00:00:00Z".to_string(),
                processing_time: Some(12.5),
            };
            
            // Test serialization
            let json = serde_json::to_string(&result);
            assert!(json.is_ok(), "Should serialize successfully");
            
            // Test deserialization
            let json_str = json.unwrap();
            let deserialized: Result<TemporalAnalysisResult, _> = serde_json::from_str(&json_str);
            assert!(deserialized.is_ok(), "Should deserialize successfully");
            
            let result2 = deserialized.unwrap();
            assert_eq!(result.id, result2.id);
            assert_eq!(result.temporal_feedback.len(), result2.temporal_feedback.len());
            assert_eq!(result.overall_assessment.strengths, result2.overall_assessment.strengths);
        }
        
        #[test]
        fn test_analysis_insight_structure() {
            let insight = AnalysisInsight {
                category: "Musical".to_string(),
                observation: "Expressive phrasing".to_string(),
                actionable_advice: "Explore more dynamic contrast".to_string(),
                score_reference: "expression: 0.72".to_string(),
            };
            
            // Verify all fields are non-empty
            assert!(!insight.category.is_empty());
            assert!(!insight.observation.is_empty());
            assert!(!insight.actionable_advice.is_empty());
            assert!(!insight.score_reference.is_empty());
            
            // Verify category is valid
            assert!(
                insight.category == "Technical" 
                || insight.category == "Musical" 
                || insight.category == "Interpretive"
            );
        }
        
        #[test]
        fn test_temporal_feedback_item_structure() {
            let feedback = TemporalFeedbackItem {
                timestamp: "0:00-0:03".to_string(),
                insights: vec![
                    AnalysisInsight {
                        category: "Technical".to_string(),
                        observation: "Test".to_string(),
                        actionable_advice: "Test advice".to_string(),
                        score_reference: "score: 0.5".to_string(),
                    }
                ],
                practice_focus: "Focus on timing".to_string(),
            };
            
            // Verify timestamp format
            assert!(feedback.timestamp.contains('-'));
            assert!(feedback.timestamp.contains(':'));
            
            // Verify insights exist
            assert!(!feedback.insights.is_empty());
            
            // Verify practice focus is non-empty
            assert!(!feedback.practice_focus.is_empty());
        }
        
        #[test]
        fn test_overall_assessment_structure() {
            let assessment = OverallAssessment {
                strengths: vec![
                    "Strong rhythmic control".to_string(),
                    "Clear articulation".to_string(),
                    "Good pedaling technique".to_string(),
                ],
                priority_areas: vec![
                    "Dynamic range expansion needed".to_string(),
                    "Work on tempo stability".to_string(),
                ],
                performance_character: "A technically solid performance with room for emotional depth.".to_string(),
            };
            
            // Verify strengths count (2-5 items)
            assert!(assessment.strengths.len() >= 2 && assessment.strengths.len() <= 5,
                "Should have 2-5 strengths");
            
            // Verify priority areas count (2-4 items)
            assert!(assessment.priority_areas.len() >= 2 && assessment.priority_areas.len() <= 4,
                "Should have 2-4 priority areas");
            
            // Verify performance character is substantial
            assert!(assessment.performance_character.len() > 20,
                "Performance character should be substantial");
        }
        
        #[test]
        fn test_practice_recommendations_structure() {
            let recommendations = PracticeRecommendations {
                immediate_priorities: vec![
                    ImmediatePriority {
                        skill_area: "Timing".to_string(),
                        specific_exercise: "Practice with metronome".to_string(),
                        expected_outcome: "Improved rhythmic accuracy".to_string(),
                    },
                    ImmediatePriority {
                        skill_area: "Dynamics".to_string(),
                        specific_exercise: "Play passages at extreme dynamics".to_string(),
                        expected_outcome: "Greater dynamic range".to_string(),
                    },
                    ImmediatePriority {
                        skill_area: "Articulation".to_string(),
                        specific_exercise: "Staccato scales".to_string(),
                        expected_outcome: "Clearer note separation".to_string(),
                    },
                ],
                long_term_development: vec![
                    LongTermDevelopment {
                        musical_aspect: "Expression".to_string(),
                        development_approach: "Study recordings of master pianists".to_string(),
                        repertoire_suggestions: "Chopin Nocturnes".to_string(),
                    },
                    LongTermDevelopment {
                        musical_aspect: "Interpretation".to_string(),
                        development_approach: "Analyze scores for structural understanding".to_string(),
                        repertoire_suggestions: "Bach Well-Tempered Clavier".to_string(),
                    },
                ],
            };
            
            // Verify immediate priorities count (3-5 items)
            assert!(recommendations.immediate_priorities.len() >= 3 
                && recommendations.immediate_priorities.len() <= 5,
                "Should have 3-5 immediate priorities");
            
            // Verify long-term development count (2-3 items)
            assert!(recommendations.long_term_development.len() >= 2 
                && recommendations.long_term_development.len() <= 3,
                "Should have 2-3 long-term development items");
            
            // Verify all fields are non-empty
            for priority in &recommendations.immediate_priorities {
                assert!(!priority.skill_area.is_empty());
                assert!(!priority.specific_exercise.is_empty());
                assert!(!priority.expected_outcome.is_empty());
            }
            
            for development in &recommendations.long_term_development {
                assert!(!development.musical_aspect.is_empty());
                assert!(!development.development_approach.is_empty());
                assert!(!development.repertoire_suggestions.is_empty());
            }
        }
    }
}
