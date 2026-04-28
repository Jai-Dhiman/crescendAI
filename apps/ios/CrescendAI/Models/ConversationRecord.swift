import Foundation
import SwiftData

@Model
final class ConversationRecord {
    @Attribute(.unique) var conversationId: String
    var sessionId: UUID
    var startedAt: Date
    var title: String?
    var synthesisText: String?

    init(conversationId: String, sessionId: UUID, startedAt: Date, title: String? = nil, synthesisText: String? = nil) {
        self.conversationId = conversationId
        self.sessionId = sessionId
        self.startedAt = startedAt
        self.title = title
        self.synthesisText = synthesisText
    }

    var displayTitle: String {
        title ?? DateFormatter.localizedString(from: startedAt, dateStyle: .medium, timeStyle: .short)
    }
}
