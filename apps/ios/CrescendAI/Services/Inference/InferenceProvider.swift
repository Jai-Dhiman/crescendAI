import Foundation

/// Protocol for running audio inference on PCM samples.
/// Core ML implementation deferred to Slice 03; this defines the contract.
protocol InferenceProvider: Sendable {
    func infer(samples: [Float], sampleRate: Int) async throws -> InferenceResult
}

/// Mock provider that returns random dimension scores.
/// Used for end-to-end pipeline testing before the Core ML model exists.
struct MockInferenceProvider: InferenceProvider {
    func infer(samples: [Float], sampleRate: Int) async throws -> InferenceResult {
        // Simulate inference latency
        try await Task.sleep(for: .milliseconds(Int.random(in: 50...150)))

        let dimensions: [String: Float] = [
            "dynamics": Float.random(in: 0.3...0.95),
            "timing": Float.random(in: 0.3...0.95),
            "pedaling": Float.random(in: 0.3...0.95),
            "articulation": Float.random(in: 0.3...0.95),
            "phrasing": Float.random(in: 0.3...0.95),
            "interpretation": Float.random(in: 0.3...0.95),
        ]

        return InferenceResult(
            dimensions: dimensions,
            processingTimeMs: Int.random(in: 80...200)
        )
    }
}
