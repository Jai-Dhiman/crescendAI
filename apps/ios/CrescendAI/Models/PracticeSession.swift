import Foundation

struct PracticeSession: Sendable {
    let id: UUID
    let startedAt: Date
    var endedAt: Date?
    var chunks: [AudioChunk]

    init(id: UUID = UUID(), startedAt: Date = Date()) {
        self.id = id
        self.startedAt = startedAt
        self.endedAt = nil
        self.chunks = []
    }

    var duration: TimeInterval {
        let end = endedAt ?? Date()
        return end.timeIntervalSince(startedAt)
    }
}
