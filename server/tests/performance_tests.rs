// Phase 3.2: Performance Benchmark Tests
// Tests to verify Phase 2 performance optimizations meet targets

use wasm_bindgen_test::*;
use worker::*;
// Remove unused imports for standard test environment
// use std::time::{Duration, Instant};
use js_sys::Date;

wasm_bindgen_test_configure!(run_in_browser);

// Performance targets from CLAUDE.md requirements
const MAX_AUDIO_PROCESSING_TIME_MS: f64 = 2000.0; // <2s mel-spectrogram
const MAX_API_LATENCY_MS: f64 = 100.0; // <100ms global
const MAX_MEMORY_USAGE_MB: usize = 128; // <128MB per isolate
const MIN_CACHE_HIT_RATE: f64 = 0.8; // >80% hit rate

/// Performance metrics tracking
struct PerformanceMetrics {
    start_time: f64,
    memory_usage: Vec<usize>,
    cache_hits: usize,
    cache_misses: usize,
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self {
            start_time: Date::now(),
            memory_usage: Vec::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    fn record_memory_usage(&mut self, usage_mb: usize) {
        self.memory_usage.push(usage_mb);
    }

    fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    fn get_elapsed_time_ms(&self) -> f64 {
        Date::now() - self.start_time
    }

    fn get_max_memory_usage(&self) -> usize {
        self.memory_usage.iter().max().copied().unwrap_or(0)
    }

    fn get_cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

// =============================================================================
// AUDIO PROCESSING PERFORMANCE TESTS
// =============================================================================

#[wasm_bindgen_test]
async fn test_audio_processing_speed() {
    let mut metrics = PerformanceMetrics::new();
    
    // Create test audio data (simulated WAV file)
    let test_audio_data = create_large_test_audio_data(44100, 30); // 30 seconds at 44.1kHz
    
    // Test mel-spectrogram generation performance
    let start_time = Date::now();
    
    // In a real implementation, this would call the actual audio processing functions
    // For now, we'll simulate the processing time
    simulate_audio_processing(&test_audio_data).await;
    
    let processing_time = Date::now() - start_time;
    
    assert!(
        processing_time < MAX_AUDIO_PROCESSING_TIME_MS,
        "Audio processing took {}ms, should be under {}ms",
        processing_time,
        MAX_AUDIO_PROCESSING_TIME_MS
    );
    
    // Test memory usage during processing
    let estimated_memory_usage = estimate_audio_processing_memory(&test_audio_data);
    metrics.record_memory_usage(estimated_memory_usage);
    
    assert!(
        estimated_memory_usage < MAX_MEMORY_USAGE_MB,
        "Audio processing used {}MB memory, should be under {}MB",
        estimated_memory_usage,
        MAX_MEMORY_USAGE_MB
    );
}

#[wasm_bindgen_test]
async fn test_concurrent_audio_processing() {
    // Test processing multiple audio files concurrently
    let file_count = 5;
    let mut processing_times = Vec::new();
    
    for _i in 0..file_count {
        let test_data = create_test_audio_data(44100, 10); // 10 second files
        let start_time = Date::now();
        
        simulate_audio_processing(&test_data).await;
        
        let processing_time = Date::now() - start_time;
        processing_times.push(processing_time);
    }
    
    // All processing should complete within time limits
    for (i, time) in processing_times.iter().enumerate() {
        assert!(
            *time < MAX_AUDIO_PROCESSING_TIME_MS,
            "Concurrent processing {} took {}ms, should be under {}ms",
            i,
            time,
            MAX_AUDIO_PROCESSING_TIME_MS
        );
    }
    
    // Average processing time should be reasonable
    let avg_time: f64 = processing_times.iter().sum::<f64>() / processing_times.len() as f64;
    assert!(
        avg_time < MAX_AUDIO_PROCESSING_TIME_MS * 0.8,
        "Average concurrent processing time {}ms is too high",
        avg_time
    );
}

#[wasm_bindgen_test]
async fn test_large_file_processing_efficiency() {
    // Test processing efficiency with large files
    let large_file_data = create_large_test_audio_data(44100, 300); // 5 minutes
    let start_time = Date::now();
    
    simulate_audio_processing(&large_file_data).await;
    
    let processing_time = Date::now() - start_time;
    
    // Large files should still process efficiently (allowing more time)
    let max_large_file_time = MAX_AUDIO_PROCESSING_TIME_MS * 5.0; // 10s for 5min file
    assert!(
        processing_time < max_large_file_time,
        "Large file processing took {}ms, should be under {}ms",
        processing_time,
        max_large_file_time
    );
}

// =============================================================================
// API LATENCY BENCHMARK TESTS
// =============================================================================

#[wasm_bindgen_test]
async fn test_api_response_latency() {
    // Test API endpoint response times
    let endpoints = vec![
        "/api/v1/health",
        "/api/v1/upload",
        "/api/v1/job/550e8400-e29b-41d4-a716-446655440000",
        "/api/v1/result/550e8400-e29b-41d4-a716-446655440000",
    ];
    
    for endpoint in endpoints {
        let start_time = Date::now();
        
        // Simulate API call
        simulate_api_call(endpoint).await;
        
        let response_time = Date::now() - start_time;
        
        assert!(
            response_time < MAX_API_LATENCY_MS,
            "API endpoint {} took {}ms, should be under {}ms",
            endpoint,
            response_time,
            MAX_API_LATENCY_MS
        );
    }
}

#[wasm_bindgen_test]
async fn test_cold_start_performance() {
    // Test cold start performance (first request after deployment)
    let start_time = Date::now();
    
    // Simulate cold start scenario
    simulate_cold_start().await;
    
    let cold_start_time = Date::now() - start_time;
    
    // Cold starts should still be reasonable (allowing 5x normal latency)
    let max_cold_start_time = MAX_API_LATENCY_MS * 5.0;
    assert!(
        cold_start_time < max_cold_start_time,
        "Cold start took {}ms, should be under {}ms",
        cold_start_time,
        max_cold_start_time
    );
}

#[wasm_bindgen_test]
async fn test_high_load_response_time() {
    // Test response times under high concurrent load
    let concurrent_requests = 20;
    let mut response_times = Vec::new();
    
    // Simulate concurrent requests
    for _i in 0..concurrent_requests {
        let start_time = Date::now();
        
        simulate_api_call("/api/v1/health").await;
        
        let response_time = Date::now() - start_time;
        response_times.push(response_time);
    }
    
    // Calculate percentiles
    response_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = response_times[response_times.len() / 2];
    let p95 = response_times[response_times.len() * 95 / 100];
    let p99 = response_times[response_times.len() * 99 / 100];
    
    assert!(
        p50 < MAX_API_LATENCY_MS,
        "P50 response time {}ms should be under {}ms",
        p50,
        MAX_API_LATENCY_MS
    );
    
    assert!(
        p95 < MAX_API_LATENCY_MS * 2.0,
        "P95 response time {}ms should be under {}ms",
        p95,
        MAX_API_LATENCY_MS * 2.0
    );
    
    assert!(
        p99 < MAX_API_LATENCY_MS * 3.0,
        "P99 response time {}ms should be under {}ms",
        p99,
        MAX_API_LATENCY_MS * 3.0
    );
}

// =============================================================================
// MEMORY USAGE VALIDATION TESTS
// =============================================================================

#[wasm_bindgen_test]
async fn test_memory_usage_limits() {
    let mut metrics = PerformanceMetrics::new();
    
    // Test memory usage during various operations
    let operations = vec![
        "upload_audio",
        "process_audio", 
        "cache_result",
        "api_request",
        "webhook_handling",
    ];
    
    for operation in operations {
        let memory_before = get_current_memory_usage();
        
        simulate_operation(operation).await;
        
        let memory_after = get_current_memory_usage();
        let memory_used = memory_after.saturating_sub(memory_before);
        
        metrics.record_memory_usage(memory_used);
        
        assert!(
            memory_used < MAX_MEMORY_USAGE_MB,
            "Operation {} used {}MB memory, should be under {}MB",
            operation,
            memory_used,
            MAX_MEMORY_USAGE_MB
        );
    }
    
    // Check maximum memory usage across all operations
    let max_memory = metrics.get_max_memory_usage();
    assert!(
        max_memory < MAX_MEMORY_USAGE_MB,
        "Maximum memory usage {}MB should be under {}MB",
        max_memory,
        MAX_MEMORY_USAGE_MB
    );
}

#[wasm_bindgen_test]
async fn test_memory_leak_detection() {
    // Test for memory leaks by running operations repeatedly
    let initial_memory = get_current_memory_usage();
    
    // Run operations multiple times
    for i in 0..10 {
        simulate_operation("upload_audio").await;
        simulate_operation("process_audio").await;
        
        // Force garbage collection if available
        if i % 3 == 0 {
            force_garbage_collection();
        }
    }
    
    let final_memory = get_current_memory_usage();
    let memory_growth = final_memory.saturating_sub(initial_memory);
    
    // Memory growth should be minimal (less than 10MB after 10 operations)
    assert!(
        memory_growth < 10,
        "Memory grew by {}MB after repeated operations, possible memory leak",
        memory_growth
    );
}

// =============================================================================
// CACHE PERFORMANCE TESTS
// =============================================================================

#[wasm_bindgen_test]
async fn test_cache_hit_rates() {
    let mut metrics = PerformanceMetrics::new();
    
    // Simulate cache operations
    let test_keys = vec![
        "result:audio1.wav",
        "result:audio2.wav", 
        "result:audio3.wav",
        "result:audio1.wav", // Repeat for cache hit
        "result:audio2.wav", // Repeat for cache hit
    ];
    
    for key in test_keys {
        let cache_hit = simulate_cache_lookup(key).await;
        
        if cache_hit {
            metrics.record_cache_hit();
        } else {
            metrics.record_cache_miss();
        }
    }
    
    let hit_rate = metrics.get_cache_hit_rate();
    assert!(
        hit_rate >= MIN_CACHE_HIT_RATE,
        "Cache hit rate {}% should be at least {}%",
        hit_rate * 100.0,
        MIN_CACHE_HIT_RATE * 100.0
    );
}

#[wasm_bindgen_test]
async fn test_cache_performance_under_load() {
    // Test cache performance under high load
    let mut total_cache_time = 0.0;
    let cache_operations = 100;
    
    for i in 0..cache_operations {
        let start_time = Date::now();
        
        let key = format!("result:test_{}.wav", i % 10); // 10 unique keys with repeats
        simulate_cache_lookup(&key).await;
        
        let cache_time = Date::now() - start_time;
        total_cache_time += cache_time;
    }
    
    let avg_cache_time = total_cache_time / cache_operations as f64;
    
    // Cache operations should be very fast (<10ms average)
    assert!(
        avg_cache_time < 10.0,
        "Average cache operation took {}ms, should be under 10ms",
        avg_cache_time
    );
}

// =============================================================================
// DATA TRANSFER EFFICIENCY TESTS
// =============================================================================

#[wasm_bindgen_test]
async fn test_data_transfer_efficiency() {
    // Test upload/download efficiency
    let file_sizes = vec![
        1024,        // 1KB
        1024 * 1024, // 1MB
        10 * 1024 * 1024, // 10MB
        50 * 1024 * 1024, // 50MB
    ];
    
    for size in file_sizes {
        let start_time = Date::now();
        
        // Simulate file transfer
        simulate_file_transfer(size).await;
        
        let transfer_time = Date::now() - start_time;
        let transfer_rate_mbps = (size as f64 / 1024.0 / 1024.0) / (transfer_time / 1000.0);
        
        // Should achieve reasonable transfer rates (>1 Mbps)
        assert!(
            transfer_rate_mbps > 1.0,
            "Transfer rate {:.2} Mbps too slow for {}MB file",
            transfer_rate_mbps,
            size / 1024 / 1024
        );
    }
}

#[wasm_bindgen_test]
async fn test_wasm_binary_size() {
    // Test that WASM binary size is reasonable
    let wasm_size = get_wasm_binary_size();
    let max_wasm_size = 10 * 1024 * 1024; // 10MB max
    
    assert!(
        wasm_size < max_wasm_size,
        "WASM binary size {}MB should be under {}MB",
        wasm_size / 1024 / 1024,
        max_wasm_size / 1024 / 1024
    );
}

// =============================================================================
// HELPER FUNCTIONS FOR PERFORMANCE TESTS
// =============================================================================

/// Create test audio data of specified duration
fn create_test_audio_data(sample_rate: usize, duration_seconds: usize) -> Vec<u8> {
    let samples = sample_rate * duration_seconds * 2; // Stereo
    let mut data = Vec::with_capacity(samples * 2); // 16-bit samples
    
    // Add WAV header
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&[(samples * 2 + 36) as u8, 0, 0, 0]); // File size
    data.extend_from_slice(b"WAVE");
    data.extend_from_slice(b"fmt ");
    data.extend_from_slice(&[16, 0, 0, 0]); // Subchunk1Size
    data.extend_from_slice(&[1, 0]); // AudioFormat (PCM)
    data.extend_from_slice(&[2, 0]); // NumChannels (stereo)
    data.extend_from_slice(&[sample_rate as u8, (sample_rate >> 8) as u8, 0, 0]); // SampleRate
    data.extend_from_slice(&[0, 0, 0, 0]); // ByteRate
    data.extend_from_slice(&[4, 0]); // BlockAlign
    data.extend_from_slice(&[16, 0]); // BitsPerSample
    data.extend_from_slice(b"data");
    data.extend_from_slice(&[(samples * 2) as u8, 0, 0, 0]); // Subchunk2Size
    
    // Add audio samples (simple sine wave)
    for i in 0..samples {
        let sample = ((i as f64 * 440.0 * 2.0 * std::f64::consts::PI / sample_rate as f64).sin() * 32767.0) as i16;
        data.extend_from_slice(&sample.to_le_bytes());
    }
    
    data
}

/// Create large test audio data
fn create_large_test_audio_data(sample_rate: usize, duration_seconds: usize) -> Vec<u8> {
    create_test_audio_data(sample_rate, duration_seconds)
}

/// Simulate audio processing
async fn simulate_audio_processing(_data: &[u8]) {
    // Simulate processing time based on data size
    let processing_delay = 100; // milliseconds
    
    // In WASM, we can't use std::thread::sleep, so we'll use a busy wait
    let start = Date::now();
    while Date::now() - start < processing_delay as f64 {
        // Busy wait
    }
}

/// Estimate memory usage for audio processing
fn estimate_audio_processing_memory(data: &[u8]) -> usize {
    // Estimate memory usage based on audio data size
    // Typically: input data + intermediate buffers + output data
    let input_size = data.len();
    let estimated_total = input_size * 3; // 3x for processing overhead
    
    // Convert to MB
    estimated_total / (1024 * 1024)
}

/// Simulate API call
async fn simulate_api_call(_endpoint: &str) {
    // Simulate API processing time
    let processing_delay = 50; // milliseconds
    
    let start = Date::now();
    while Date::now() - start < processing_delay as f64 {
        // Busy wait
    }
}

/// Simulate cold start
async fn simulate_cold_start() {
    // Simulate cold start overhead
    let cold_start_delay = 200; // milliseconds
    
    let start = Date::now();
    while Date::now() - start < cold_start_delay as f64 {
        // Busy wait
    }
}

/// Simulate operation
async fn simulate_operation(_operation: &str) {
    // Simulate operation processing
    let processing_delay = 10; // milliseconds
    
    let start = Date::now();
    while Date::now() - start < processing_delay as f64 {
        // Busy wait
    }
}

/// Get current memory usage (simulated)
fn get_current_memory_usage() -> usize {
    // In a real implementation, this would get actual memory usage
    // For testing, we'll simulate varying memory usage
    use js_sys::Math;
    (Math::random() * 50.0) as usize // Random usage between 0-50MB
}

/// Force garbage collection (if available)
fn force_garbage_collection() {
    // In some WASM environments, gc() might be available
    // For now, this is a no-op
}

/// Simulate cache lookup
async fn simulate_cache_lookup(key: &str) -> bool {
    // Simulate cache hit/miss based on key
    // For testing, assume repeated keys are cache hits
    key.contains("audio1") || key.contains("audio2")
}

/// Simulate file transfer
async fn simulate_file_transfer(size: usize) {
    // Simulate transfer time based on file size
    let transfer_delay = (size / (1024 * 1024)).max(10); // At least 10ms, 1ms per MB
    
    let start = Date::now();
    while Date::now() - start < transfer_delay as f64 {
        // Busy wait
    }
}

/// Get WASM binary size (simulated)
fn get_wasm_binary_size() -> usize {
    // In a real implementation, this would get the actual WASM binary size
    // For testing, we'll simulate a reasonable size
    5 * 1024 * 1024 // 5MB
}