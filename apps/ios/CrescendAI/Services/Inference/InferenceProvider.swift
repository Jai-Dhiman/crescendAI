import Foundation

/// Protocol for running audio inference on PCM samples.
///
/// The live practice path does NOT run on-device inference: the authoritative
/// 6-dim scores are computed server-side and delivered over the WebSocket as
/// `chunk_processed`. This protocol is retained as the seam for a possible future
/// real Core ML provider; there is deliberately no random-mock implementation.
protocol InferenceProvider: Sendable {
    func infer(samples: [Float], sampleRate: Int) async throws -> InferenceResult
}
