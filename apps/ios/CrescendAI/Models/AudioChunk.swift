import Foundation

struct AudioChunk: Sendable, Identifiable {
    let id: UUID
    let sessionId: UUID
    let index: Int
    let startOffset: TimeInterval
    let duration: TimeInterval
    let localFileURL: URL?
    var inferenceResult: InferenceResult?

    init(
        id: UUID = UUID(),
        sessionId: UUID,
        index: Int,
        startOffset: TimeInterval,
        duration: TimeInterval,
        localFileURL: URL? = nil,
        inferenceResult: InferenceResult? = nil
    ) {
        self.id = id
        self.sessionId = sessionId
        self.index = index
        self.startOffset = startOffset
        self.duration = duration
        self.localFileURL = localFileURL
        self.inferenceResult = inferenceResult
    }
}

struct InferenceResult: Sendable {
    let dimensions: [String: Float]
    let processingTimeMs: Int
}
