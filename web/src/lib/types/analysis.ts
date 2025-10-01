// Note: Core types now come from the API schema
// Import them from there to avoid duplication
import type { AnalysisResult as ApiAnalysisResult } from '$lib/schemas/api';

// Re-export the component types for convenience
export interface AnalysisInsight {
  category: string;
  observation: string;
  actionable_advice: string;
  score_reference: string;
}

export interface TemporalFeedbackItem {
  timestamp: string;
  insights: AnalysisInsight[];
  practice_focus: string;
}

export interface OverallAssessment {
  strengths: string[];
  priority_areas: string[];
  performance_character: string;
}

export interface ImmediatePriority {
  skill_area: string;
  specific_exercise: string;
  expected_outcome: string;
}

export interface LongTermDevelopment {
  musical_aspect: string;
  development_approach: string;
  repertoire_suggestions: string;
}

export interface PracticeRecommendations {
  immediate_priorities: ImmediatePriority[];
  long_term_development: LongTermDevelopment[];
}

// Extend the API result type with UI-specific fields
export interface AnalysisResult extends Omit<ApiAnalysisResult, 'id' | 'status' | 'file_id' | 'created_at' | 'processing_time'> {
  // UI-specific fields (optional since they're added by the frontend)
  audioUrl?: string;
  fileName?: string;
  // API fields are kept optional for backward compatibility
  id?: string;
  status?: string;
  file_id?: string;
  created_at?: string;
  processing_time?: number | null;
}

