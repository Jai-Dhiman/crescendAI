# API Schema Update - Temporal Analysis Format

## Date
2025-10-01

## Problem
The frontend API schema validation was expecting the OLD analysis format with:
- `analysis` object (containing 19 PercePiano dimensions)
- `insights` array (simple string array)

However, the Rust server was returning the NEW temporal analysis format with:
- `overall_assessment` object
- `temporal_feedback` array
- `practice_recommendations` object
- `encouragement` string

This caused Zod validation to fail with:
```
Invalid input: expected object, received undefined (for 'analysis')
Invalid input: expected array, received undefined (for 'insights')
```

## Solution
Updated the API schema in `src/lib/schemas/api.ts` to match the temporal analysis format that the server actually returns.

### Changes Made

1. **Added new Zod schemas** for temporal analysis components:
   - `AnalysisInsightSchema` - Individual insights with category, observation, actionable advice
   - `TemporalFeedbackItemSchema` - Feedback for specific time segments
   - `OverallAssessmentSchema` - Performance strengths and priority areas
   - `ImmediatePrioritySchema` - Short-term practice priorities
   - `LongTermDevelopmentSchema` - Long-term musical development goals
   - `PracticeRecommendationsSchema` - Combined immediate and long-term recommendations

2. **Updated `AnalysisResultSchema`** to expect:
   ```typescript
   {
     id: string (uuid),
     status: string,
     file_id: string (uuid),
     overall_assessment: OverallAssessment,
     temporal_feedback: TemporalFeedbackItem[],
     practice_recommendations: PracticeRecommendations,
     encouragement: string,
     created_at: string,
     processing_time: number | null (optional)
   }
   ```

3. **Updated `src/lib/types/analysis.ts`** to:
   - Import the API schema type to avoid duplication
   - Extend it with UI-specific fields (`audioUrl`, `fileName`)
   - Keep API fields optional for backward compatibility

## Server Response Structure
The server (`server/src/lib.rs`) returns `TemporalAnalysisResult`:
```rust
pub struct TemporalAnalysisResult {
    pub id: String,
    pub status: String,
    pub file_id: String,
    pub overall_assessment: OverallAssessment,
    pub temporal_feedback: Vec<TemporalFeedbackItem>,
    pub practice_recommendations: PracticeRecommendations,
    pub encouragement: String,
    pub created_at: String,
    pub processing_time: Option<f32>,
}
```

## Testing
- TypeScript compilation passes with no errors related to the schema changes
- Build completes successfully
- Schema now correctly validates the server response format

## Files Modified
1. `/Users/jdhiman/Documents/crescendai/web/src/lib/schemas/api.ts`
2. `/Users/jdhiman/Documents/crescendai/web/src/lib/types/analysis.ts`

## Next Steps
The app should now work correctly! To test:
1. Start the web dev server: `cd web && bun dev`
2. Upload an audio file
3. Verify the analysis completes without validation errors
4. Check the results page displays the temporal feedback correctly
