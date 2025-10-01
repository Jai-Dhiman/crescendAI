# Temporal Analysis Testing Checklist

## Overview
This document provides a comprehensive testing guide for the temporal analysis feature in CrescendAI backend.

## Test Categories

### 1. Unit Tests - Audio Chunking
Tests for the core audio chunking functionality.

#### Test Cases:
- [ ] **test_chunk_audio_basic**: 5-second audio produces 2 chunks
  - Expected: 2 chunks with timestamps "0:00-0:03" and "0:02-0:05"
  - Status: ✓ Implemented

- [ ] **test_chunk_audio_10_seconds**: 10-second audio produces 4-5 chunks
  - Expected: 4-5 chunks with proper overlap
  - Status: ✓ Implemented

- [ ] **test_chunk_audio_30_seconds**: 30-second audio produces ~15 chunks
  - Expected: 14-15 chunks with 1-second overlap validation
  - Status: ✓ Implemented

- [ ] **test_timestamp_formatting**: Timestamp strings are correctly formatted
  - Expected: "MM:SS-MM:SS" format
  - Status: ✓ Implemented

### 2. Unit Tests - Edge Cases
Tests for error handling and boundary conditions.

#### Test Cases:
- [ ] **test_audio_too_short**: Audio < 3 seconds fails gracefully
  - Expected: Error with message about minimum duration
  - Status: ✓ Implemented

- [ ] **test_invalid_chunk_duration**: Zero or negative chunk duration fails
  - Expected: Error for invalid parameters
  - Status: ✓ Implemented

- [ ] **test_invalid_overlap**: Overlap >= chunk duration fails
  - Expected: Error for invalid overlap
  - Status: ✓ Implemented

- [ ] **test_empty_audio**: Empty audio data fails gracefully
  - Expected: Error for insufficient samples
  - Status: ✓ Implemented

- [ ] **test_invalid_sample_rate**: Sample rate of 0 fails
  - Expected: Error for invalid sample rate
  - Status: ✓ Implemented

- [ ] **test_long_audio_300_seconds**: 5-minute audio handled correctly
  - Expected: ~150 chunks without performance issues
  - Status: ✓ Implemented

### 3. Unit Tests - Data Structures
Tests for serialization and validation of temporal analysis structures.

#### Test Cases:
- [ ] **test_temporal_analysis_result_serialization**: Round-trip JSON serialization
  - Expected: All fields preserved after deserialize
  - Status: ✓ Implemented

- [ ] **test_analysis_insight_structure**: Insight fields validated
  - Expected: Non-empty strings, valid category
  - Status: ✓ Implemented

- [ ] **test_temporal_feedback_item_structure**: Feedback structure validated
  - Expected: Valid timestamp format, non-empty insights
  - Status: ✓ Implemented

- [ ] **test_overall_assessment_structure**: Assessment has required counts
  - Expected: 2-5 strengths, 2-4 priority areas
  - Status: ✓ Implemented

- [ ] **test_practice_recommendations_structure**: Recommendations validated
  - Expected: 3-5 immediate priorities, 2-3 long-term goals
  - Status: ✓ Implemented

### 4. Integration Tests - End-to-End Workflow
Tests for complete temporal analysis pipeline (requires WASM runtime).

#### Test Cases:
- [ ] **test_full_workflow_5s_audio**: Complete analysis of 5-second file
  - Expected: 2 chunks, valid insights, complete result structure
  - Status: Pending (requires mock environment)

- [ ] **test_full_workflow_10s_audio**: Complete analysis of 10-second file
  - Expected: 4-5 chunks, temporal feedback array matches chunk count
  - Status: Pending

- [ ] **test_llm_integration**: LLM calls produce valid insights
  - Expected: Non-empty observations and advice
  - Status: Pending

- [ ] **test_storage_roundtrip**: Store and retrieve temporal results
  - Expected: Data integrity maintained
  - Status: Pending

## Manual Testing Procedures

### Setup
```bash
# Navigate to server directory
cd server

# Build for WASM target
cargo build --release --target wasm32-unknown-unknown

# Start local development server
wrangler dev
```

### Test Scenarios

#### Scenario 1: Upload and Analyze 5-Second Audio
```bash
# 1. Upload audio file
curl -X POST http://localhost:8787/api/v1/upload \
  -F "audio=@tests/fixtures/test_audio_5s.wav" \
  -H "X-API-Key: test-key"

# Expected response:
# {
#   "id": "file-uuid",
#   "status": "uploaded",
#   "message": "Audio file uploaded successfully"
# }

# 2. Start temporal analysis
curl -X POST http://localhost:8787/api/v1/analyze \
  -H "Content-Type: application/json" \
  -H "X-Use-Temporal: true" \
  -H "X-API-Key: test-key" \
  -d '{"id": "file-uuid"}'

# Expected response:
# {
#   "job_id": "job-uuid",
#   "status": "processing",
#   "message": "Temporal analysis started successfully",
#   "analysis_type": "temporal"
# }

# 3. Check job status
curl http://localhost:8787/api/v1/job/job-uuid \
  -H "X-API-Key: test-key"

# Expected response (while processing):
# {
#   "job_id": "job-uuid",
#   "status": "processing",
#   "progress": 45.0,
#   "error": null
# }

# 4. Retrieve results
curl http://localhost:8787/api/v1/result/job-uuid \
  -H "X-API-Key: test-key"

# Expected response structure:
# {
#   "id": "job-uuid",
#   "status": "completed",
#   "file_id": "file-uuid",
#   "overall_assessment": {
#     "strengths": ["...", "..."],
#     "priority_areas": ["...", "..."],
#     "performance_character": "..."
#   },
#   "temporal_feedback": [
#     {
#       "timestamp": "0:00-0:03",
#       "insights": [...],
#       "practice_focus": "..."
#     },
#     {
#       "timestamp": "0:02-0:05",
#       "insights": [...],
#       "practice_focus": "..."
#     }
#   ],
#   "practice_recommendations": {
#     "immediate_priorities": [...],
#     "long_term_development": [...]
#   },
#   "encouragement": "...",
#   "created_at": "...",
#   "processing_time": 12.5
# }
```

#### Scenario 2: Upload and Analyze 10-Second Audio
```bash
# Follow same steps as Scenario 1, but with 10-second file
# Expected: 4-5 chunks in temporal_feedback array
```

#### Scenario 3: Error Handling - Audio Too Short
```bash
# Upload 2-second audio file
curl -X POST http://localhost:8787/api/v1/upload \
  -F "audio=@tests/fixtures/test_audio_2s.wav" \
  -H "X-API-Key: test-key"

# Start analysis
curl -X POST http://localhost:8787/api/v1/analyze \
  -H "Content-Type: application/json" \
  -H "X-Use-Temporal: true" \
  -H "X-API-Key: test-key" \
  -d '{"id": "file-uuid"}'

# Check job status - should show error
curl http://localhost:8787/api/v1/job/job-uuid \
  -H "X-API-Key: test-key"

# Expected:
# {
#   "job_id": "job-uuid",
#   "status": "failed",
#   "progress": 0.0,
#   "error": "Audio too short: ... samples, need at least ... for one 3-second chunk"
# }
```

### Validation Checklist

For each test scenario, verify:

- [ ] **Chunk Count**: temporal_feedback array length matches expected chunks
  - 5s audio → 2 chunks
  - 10s audio → 4-5 chunks
  - 30s audio → ~15 chunks

- [ ] **Timestamp Format**: All timestamps follow "MM:SS-MM:SS" pattern

- [ ] **Insights Completeness**: Each temporal_feedback item has:
  - [ ] Non-empty insights array
  - [ ] Each insight has category, observation, actionable_advice
  - [ ] Non-empty practice_focus

- [ ] **Overall Assessment**: Contains:
  - [ ] 2-5 strengths
  - [ ] 2-4 priority areas
  - [ ] Substantial performance_character (> 20 chars)

- [ ] **Practice Recommendations**: Contains:
  - [ ] 3-5 immediate_priorities
  - [ ] 2-3 long_term_development items
  - [ ] All fields non-empty

- [ ] **Processing Time**: Reasonable duration (< 30s for 10s audio)

- [ ] **Error Handling**: Graceful failures with informative messages

### Console Log Monitoring

During manual testing, monitor console logs for:

```
✓ Good patterns:
- "Created X chunks from audio"
- "Analyzing chunk N/M"
- "Generated temporal feedback for all chunks"
- "=== Temporal Analysis Complete ==="

✗ Warning patterns:
- "Failed to ..."
- "Error: ..."
- Unexpected panics or stack traces
```

## Performance Benchmarks

### Expected Processing Times (with LLM calls):
- 5-second audio: 8-15 seconds
- 10-second audio: 15-25 seconds
- 30-second audio: 45-75 seconds
- 60-second audio: 90-150 seconds

### Memory Usage:
- Peak memory should stay < 100MB for typical audio files
- No memory leaks over multiple analyses

### Chunk Processing:
- ~2-3 seconds per chunk (including LLM generation)
- Linear scaling with number of chunks

## Common Issues and Troubleshooting

### Issue: "Audio too short" error for valid audio
**Cause**: Sample rate mismatch or incorrect duration calculation
**Solution**: Verify sample rate is correctly detected, check audio file format

### Issue: Temporal analysis never completes
**Cause**: LLM API timeout or infinite loop in processing
**Solution**: Check LLM API status, add timeout handling, verify chunk loop termination

### Issue: Empty or malformed insights
**Cause**: LLM response parsing failure
**Solution**: Check LLM prompt format, verify response parsing logic, add fallbacks

### Issue: Incorrect chunk count
**Cause**: Off-by-one error in chunking logic
**Solution**: Verify hop_samples calculation, check end boundary conditions

### Issue: Tests fail in CI but pass locally
**Cause**: Environment differences (WASM runtime, KV storage)
**Solution**: Use mock environment, ensure test isolation

## Coverage Requirements

### Minimum Coverage Targets:
- audio_dsp.rs: 90% line coverage
- processing.rs (temporal functions): 85% line coverage
- lib.rs (data structures): 100% coverage

### Running Coverage:
```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Run coverage
cargo tarpaulin --out Html --output-dir coverage

# View report
open coverage/index.html
```

## Test Data Files

### Required Test Fixtures:
- `tests/fixtures/test_audio_5s.wav` - 5-second valid piano recording
- `tests/fixtures/test_audio_10s.wav` - 10-second valid piano recording
- `tests/fixtures/test_audio_30s.wav` - 30-second valid piano recording
- `tests/fixtures/test_audio_2s.wav` - 2-second file (too short)

### Generating Test Files:
```bash
# Use FFmpeg to generate test files
ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -ar 44100 tests/fixtures/test_audio_5s.wav
ffmpeg -f lavfi -i "sine=frequency=440:duration=10" -ar 44100 tests/fixtures/test_audio_10s.wav
ffmpeg -f lavfi -i "sine=frequency=440:duration=30" -ar 44100 tests/fixtures/test_audio_30s.wav
ffmpeg -f lavfi -i "sine=frequency=440:duration=2" -ar 44100 tests/fixtures/test_audio_2s.wav
```

## Adding New Tests

### Steps to Add a New Test:
1. Identify the feature or edge case to test
2. Add test function to appropriate module in `temporal_analysis_tests.rs`
3. Update this checklist with the new test case
4. Run test locally to verify it passes
5. Add to CI pipeline if applicable
6. Document expected behavior and edge cases

### Test Naming Convention:
- `test_<feature>_<scenario>`: Basic functionality tests
- `test_<feature>_edge_<case>`: Edge case tests
- `test_<feature>_error_<condition>`: Error handling tests

## Status Summary

### Completed:
- ✓ Unit tests for audio chunking (6 tests)
- ✓ Unit tests for edge cases (6 tests)
- ✓ Unit tests for data structures (5 tests)
- ✓ Test module setup and integration

### Pending:
- ⚠ Integration tests (requires WASM test environment)
- ⚠ LLM integration tests (requires mock LLM responses)
- ⚠ Storage round-trip tests (requires mock KV storage)
- ⚠ Manual testing with real audio files
- ⚠ Performance benchmarking

### Known Issues:
- Tests require WASM runtime (wasm-bindgen-test) which has setup complexity
- LLM API calls in tests need mocking infrastructure
- KV storage operations need mock implementation for testing

## Next Steps

1. **Immediate**: Run manual tests with wrangler dev
2. **Short-term**: Set up mock environment for integration tests
3. **Medium-term**: Add performance benchmarking suite
4. **Long-term**: Integrate into CI/CD pipeline with automated testing

---

**Last Updated**: 2025-10-01
**Maintainer**: CrescendAI Backend Team
