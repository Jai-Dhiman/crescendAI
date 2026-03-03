import Foundation
import SwiftData

@Model
final class PracticeSessionRecord {
    @Attribute(.unique) var id: UUID
    var startedAt: Date
    var endedAt: Date?
    var synced: Bool

    @Relationship(deleteRule: .cascade, inverse: \ChunkResultRecord.session)
    var chunks: [ChunkResultRecord]

    init(id: UUID = UUID(), startedAt: Date = Date(), synced: Bool = false) {
        self.id = id
        self.startedAt = startedAt
        self.endedAt = nil
        self.synced = synced
        self.chunks = []
    }
}
