import { z } from 'zod';

// Base API response schemas
export const ApiErrorSchema = z.object({
  error: z.string().nullable().optional(),
  message: z.string(),
  code: z.number().optional(),
});

// Upload response schema
export const UploadResponseSchema = z.object({
  id: z.string().uuid(),
  status: z.literal('uploaded'),
  message: z.string(),
  original_filename: z.string().optional(),
});

// Analysis request schema
export const AnalysisRequestSchema = z.object({
  id: z.string().uuid(),
});

// Analysis response schema
export const AnalysisResponseSchema = z.object({
  job_id: z.string().uuid(),
  status: z.literal('processing'),
  message: z.string(),
});

// Job status schema
export const JobStatusSchema = z.object({
  job_id: z.string().uuid(),
  status: z.enum(['pending', 'processing', 'completed', 'failed']),
  progress: z.number().min(0).max(100).optional(),
  created_at: z.string().datetime().optional(),
  updated_at: z.string().datetime().optional(),
  error: z.string().nullable().optional(),
});

// Temporal Analysis Schemas
// These schemas match the TemporalAnalysisResult structure from the server

// Individual insight for a temporal segment
export const AnalysisInsightSchema = z.object({
  category: z.string(),
  observation: z.string(),
  actionable_advice: z.string(),
  score_reference: z.string(),
});

// Temporal feedback for a specific time segment
export const TemporalFeedbackItemSchema = z.object({
  timestamp: z.string(),
  insights: z.array(AnalysisInsightSchema),
  practice_focus: z.string(),
});

// Overall assessment of the entire performance
export const OverallAssessmentSchema = z.object({
  strengths: z.array(z.string()),
  priority_areas: z.array(z.string()),
  performance_character: z.string(),
});

// Immediate practice priority
export const ImmediatePrioritySchema = z.object({
  skill_area: z.string(),
  specific_exercise: z.string(),
  expected_outcome: z.string(),
});

// Long-term musical development goal
export const LongTermDevelopmentSchema = z.object({
  musical_aspect: z.string(),
  development_approach: z.string(),
  repertoire_suggestions: z.string(),
});

// Practice recommendations structure
export const PracticeRecommendationsSchema = z.object({
  immediate_priorities: z.array(ImmediatePrioritySchema),
  long_term_development: z.array(LongTermDevelopmentSchema),
});

// Analysis result schema - Temporal analysis format
export const AnalysisResultSchema = z.object({
  id: z.string().uuid(),
  status: z.string(),
  file_id: z.string().uuid(),
  overall_assessment: OverallAssessmentSchema,
  temporal_feedback: z.array(TemporalFeedbackItemSchema),
  practice_recommendations: PracticeRecommendationsSchema,
  encouragement: z.string(),
  created_at: z.string(),
  processing_time: z.number().nullable().optional(),
});

// Health check schema
export const HealthResponseSchema = z.object({
  status: z.literal('healthy'),
  message: z.string(),
});

// Export types from schemas
export type ApiError = z.infer<typeof ApiErrorSchema>;
export type UploadResponse = z.infer<typeof UploadResponseSchema>;
export type AnalysisRequest = z.infer<typeof AnalysisRequestSchema>;
export type AnalysisResponse = z.infer<typeof AnalysisResponseSchema>;
export type JobStatus = z.infer<typeof JobStatusSchema>;
export type AnalysisResult = z.infer<typeof AnalysisResultSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;

// Validation helpers
export const validateUploadResponse = (data: unknown): UploadResponse => 
  UploadResponseSchema.parse(data);

export const validateAnalysisResponse = (data: unknown): AnalysisResponse => 
  AnalysisResponseSchema.parse(data);

export const validateJobStatus = (data: unknown): JobStatus => 
  JobStatusSchema.parse(data);

export const validateAnalysisResult = (data: unknown): AnalysisResult => 
  AnalysisResultSchema.parse(data);

export const validateHealthResponse = (data: unknown): HealthResponse => 
  HealthResponseSchema.parse(data);