import Foundation
import SwiftData

@Model
final class CheckInRecord {
    @Attribute(.unique) var id: UUID
    var sessionId: UUID
    var question: String
    var answer: String?
    var createdAt: Date
    var synced: Bool

    init(
        id: UUID = UUID(),
        sessionId: UUID,
        question: String,
        answer: String? = nil
    ) {
        self.id = id
        self.sessionId = sessionId
        self.question = question
        self.answer = answer
        self.createdAt = Date()
        self.synced = false
    }
}
