import XCTest
import SwiftData
@testable import CrescendAI

final class ConversationRecordTests: XCTestCase {

    func test_insertAndFetchConversationRecord() throws {
        let schema = Schema([ConversationRecord.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        let context = ModelContext(container)

        let record = ConversationRecord(
            conversationId: "conv-001",
            sessionId: UUID(),
            startedAt: Date(),
            synthesisText: "Good session"
        )
        context.insert(record)
        try context.save()

        let descriptor = FetchDescriptor<ConversationRecord>()
        let results = try context.fetch(descriptor)

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].conversationId, "conv-001")
        XCTAssertEqual(results[0].synthesisText, "Good session")
    }

    func test_conversationRecordTitleDefaultsToDate() throws {
        let schema = Schema([ConversationRecord.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        let context = ModelContext(container)

        let now = Date()
        let record = ConversationRecord(
            conversationId: "conv-002",
            sessionId: UUID(),
            startedAt: now,
            synthesisText: nil
        )
        context.insert(record)

        XCTAssertNil(record.title)
        XCTAssertFalse(record.displayTitle.isEmpty)
    }
}
