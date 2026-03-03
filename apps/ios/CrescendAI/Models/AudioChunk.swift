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

// MARK: - SwiftData Conversion

extension AudioChunk {
    func toRecord(session: PracticeSessionRecord) -> ChunkResultRecord {
        let result = inferenceResult
        let status: InferenceStatus = result != nil ? .completed : .pending

        return ChunkResultRecord(
            index: index,
            startOffset: startOffset,
            duration: duration,
            dynamics: Double(result?.dimensions["dynamics"] ?? 0),
            timing: Double(result?.dimensions["timing"] ?? 0),
            pedaling: Double(result?.dimensions["pedaling"] ?? 0),
            articulation: Double(result?.dimensions["articulation"] ?? 0),
            phrasing: Double(result?.dimensions["phrasing"] ?? 0),
            interpretation: Double(result?.dimensions["interpretation"] ?? 0),
            inferenceStatus: status,
            processingTimeMs: result?.processingTimeMs ?? 0,
            session: session
        )
    }
}
