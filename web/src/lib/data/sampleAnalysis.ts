import type { AnalysisResult } from '$lib/types/analysis';

export const SAMPLE_ANALYSIS: AnalysisResult = {
  "overall_assessment": {
    "strengths": [
      "Excellent pedal control with sophisticated, clean technique (scores 0.76-0.51)",
      "Strong dynamic sophistication showing refined musical control (0.90 overall)",
      "Effective use of bright timbral qualities creating engaging sound character"
    ],
    "priority_areas": [
      "Rhythmic stability needs attention - timing inconsistencies throughout (0.28 overall)",
      "Develop more connected, legato articulation (very low scores 0.06 overall)",
      "Expand dynamic range for greater expressive contrast"
    ],
    "performance_character": "A performance with sophisticated pedal work and refined dynamics, but challenged by rhythmic instability and overly detached articulation. The interpretation leans toward bright, energetic character with moments of high emotional intensity."
  },
  "temporal_feedback": [
    {
      "timestamp": "0:00-0:03",
      "insights": [
        {
          "category": "Technical",
          "observation": "Opening shows very bright, optimistic character but with extremely light touch",
          "actionable_advice": "Work on establishing a more substantial, grounded opening. Practice this passage with firm finger action, feeling the key bed on each note.",
          "score_reference": "timbre_bright_dark: 1.0, articulation_soft_cushioned_hard_solid: 0.09"
        }
      ],
      "practice_focus": "Establishing solid finger technique and rhythmic foundation from the opening"
    },
    {
      "timestamp": "0:02-0:05",
      "insights": [
        {
          "category": "Technical",
          "observation": "Improved articulation strength and some pedal engagement beginning to develop",
          "actionable_advice": "Build on this improvement by maintaining consistent finger strength throughout. Practice scales with this level of articulation.",
          "score_reference": "articulation_soft_cushioned_hard_solid: 0.49, dynamic_sophisticated_mellow_raw_crude: 0.66"
        },
        {
          "category": "Musical",
          "observation": "Musical pacing begins to emerge with more sophisticated dynamics",
          "actionable_advice": "Continue developing this musical momentum while maintaining the dynamic sophistication shown here.",
          "score_reference": "music_making_fast_paced_slow_paced: 0.26"
        }
      ],
      "practice_focus": "Sustaining improved articulation while developing musical flow"
    },
    {
      "timestamp": "0:04-0:07",
      "insights": [
        {
          "category": "Technical",
          "observation": "Excellent pedal work emerges - both saturation and clarity are optimal",
          "actionable_advice": "This is exemplary pedal technique. Study this passage to understand what you're doing right and apply this approach throughout.",
          "score_reference": "pedal_sparse_dry_saturated_wet: 1.0, pedal_clean_blurred: 1.0"
        },
        {
          "category": "Musical",
          "observation": "Timing becomes more stable, supporting the musical architecture",
          "actionable_advice": "Practice this section slowly with metronome to internalize this improved rhythmic stability.",
          "score_reference": "timing_stable_unstable: 0.50"
        }
      ],
      "practice_focus": "Replicating this excellent pedal technique and rhythmic stability"
    },
    {
      "timestamp": "0:06-0:09",
      "insights": [
        {
          "category": "Musical",
          "observation": "Peak emotional intensity with very high energy and excellent dynamic sophistication",
          "actionable_advice": "This represents your strongest expressive moment. Analyze what musical choices create this effect and use as a model for other climactic passages.",
          "score_reference": "emotion_mood_low_energy_high_energy: 0.91, dynamic_sophisticated_mellow_raw_crude: 1.0"
        },
        {
          "category": "Technical",
          "observation": "Strong articulation control balances the intensity effectively",
          "actionable_advice": "Practice maintaining this level of finger control during forte passages in other repertoire.",
          "score_reference": "articulation_soft_cushioned_hard_solid: 0.67"
        }
      ],
      "practice_focus": "Sustaining technical control during emotional peaks"
    },
    {
      "timestamp": "0:08-0:11",
      "insights": [
        {
          "category": "Technical",
          "observation": "Best rhythmic stability in the performance, showing your potential",
          "actionable_advice": "This timing quality should be your benchmark. Practice other sections at this tempo until timing matches this stability.",
          "score_reference": "timing_stable_unstable: 0.84"
        },
        {
          "category": "Interpretive",
          "observation": "More musical pacing emerges, but articulation becomes too detached",
          "actionable_advice": "Work on connecting notes while maintaining this musical flow. Practice with overlapping fingers to achieve legato.",
          "score_reference": "music_making_fast_paced_slow_paced: 0.36"
        }
      ],
      "practice_focus": "Combining rhythmic stability with more connected playing"
    }
  ],
  "practice_recommendations": {
    "immediate_priorities": [
      {
        "skill_area": "Rhythmic Stability",
        "specific_exercise": "Practice with metronome at 60% tempo, focusing on consistent subdivision. Use Czerny Op. 299 studies for rhythmic precision.",
        "expected_outcome": "More consistent timing throughout performance, reducing the current instability"
      },
      {
        "skill_area": "Legato Articulation",
        "specific_exercise": "Five-finger patterns with finger overlapping technique. Practice scales focusing on finger connection rather than finger independence.",
        "expected_outcome": "More connected, singing line quality instead of detached articulation"
      },
      {
        "skill_area": "Touch Control",
        "specific_exercise": "Practice pp-ff scales with consistent finger strength, feeling key resistance. Work on graduated dynamics within phrases.",
        "expected_outcome": "More substantial tone quality and greater dynamic range"
      }
    ],
    "long_term_development": [
      {
        "musical_aspect": "Interpretive Consistency",
        "development_approach": "Study recordings of professional pianists, focusing on how they maintain musical character while varying technical approaches",
        "repertoire_suggestions": "Chopin Nocturnes for legato and pedal work, Bach Inventions for rhythmic stability and articulation control"
      },
      {
        "musical_aspect": "Structural Balance",
        "development_approach": "Learn to identify and practice musical high points, ensuring technical execution supports rather than undermines musical peaks",
        "repertoire_suggestions": "Beethoven Sonatas for structural development, Debussy for advanced pedal technique"
      }
    ]
  },
  "encouragement": "Your pedal technique and dynamic sophistication are truly impressive - these are advanced skills that show real musical maturity. The moments where your timing stabilizes reveal your true potential as a pianist. With focused practice on rhythmic consistency and legato technique, you have all the tools to become a compelling performer. Your musical instincts are strong; now it's about giving them the technical foundation they deserve.",
  "audioUrl": "",
  "fileName": "sample_recording.mp3"
};

