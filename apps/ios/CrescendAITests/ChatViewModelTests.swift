import XCTest
import SwiftData
@testable import CrescendAI

@MainActor
final class MockPracticeService: PracticeSessionServiceProtocol {
    var eventStream: AsyncStream<PracticeEvent>
    var _continuation: AsyncStream<PracticeEvent>.Continuation

    var state: PracticeSessionService.State = .idle
    var currentLevel: Float = 0
    var elapsedSeconds: TimeInterval = 0
    var conversationId: String?

    init() {
        var cont: AsyncStream<PracticeEvent>.Continuation!
        eventStream = AsyncStream { cont = $0 }
        _continuation = cont
    }

    func start() async throws { state = .recording }
    func stop() async { state = .idle }
    func askForFeedback() async {}
}

@MainActor
final class MockChatService: ChatServiceProtocol {
    var stubbedEvents: [ChatEvent] = []

    func send(message: String, conversationId: String?) -> AsyncStream<ChatEvent> {
        let events = stubbedEvents
        return AsyncStream { continuation in
            Task {
                for event in events { continuation.yield(event) }
                continuation.finish()
            }
        }
    }
}

@MainActor
final class ChatViewModelTests: XCTestCase {

    func test_observationEventAppendsObservationMessage() async throws {
        let schema = Schema([Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self, CheckInRecord.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        let context = ModelContext(container)

        let practiceService = MockPracticeService()
        let chatService = MockChatService()

        let vm = ChatViewModel()
        vm.configureForTesting(modelContext: context, practiceService: practiceService, chatService: chatService)

        practiceService._continuation.yield(.observation(text: "Nice phrasing", dimension: "phrasing", artifacts: []))

        try await Task.sleep(for: .milliseconds(50))

        let obsMessages = vm.messages.filter { $0.role == .observation }
        XCTAssertEqual(obsMessages.count, 1)
        XCTAssertEqual(obsMessages[0].text, "Nice phrasing")
        XCTAssertEqual(obsMessages[0].dimension, "phrasing")
    }

    func test_sendMessageAppendsDeltaAsTeacherMessage() async throws {
        let schema = Schema([Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self, CheckInRecord.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        let context = ModelContext(container)

        let practiceService = MockPracticeService()
        let chatService = MockChatService()
        chatService.stubbedEvents = [
            .start(conversationId: "c-1"),
            .delta("Great work"),
            .done,
        ]

        let vm = ChatViewModel()
        vm.configureForTesting(modelContext: context, practiceService: practiceService, chatService: chatService)

        vm.inputText = "How am I doing?"
        await vm.sendMessage()

        try await Task.sleep(for: .milliseconds(50))

        let userMessages = vm.messages.filter { $0.role == .user }
        let teacherMessages = vm.messages.filter { $0.role == .teacher }

        XCTAssertEqual(userMessages.count, 1)
        XCTAssertEqual(userMessages[0].text, "How am I doing?")
        XCTAssertFalse(teacherMessages.isEmpty)

        let fullText = teacherMessages.map(\.text).joined()
        XCTAssertTrue(fullText.contains("Great work"))
    }

    func test_synthesisEventAppendsSynthesisMessage() async throws {
        let schema = Schema([Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self, CheckInRecord.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        let context = ModelContext(container)

        let practiceService = MockPracticeService()
        let chatService = MockChatService()

        let vm = ChatViewModel()
        vm.configureForTesting(modelContext: context, practiceService: practiceService, chatService: chatService)

        practiceService._continuation.yield(.synthesis(text: "Session summary text", artifacts: []))
        try await Task.sleep(for: .milliseconds(50))

        let teacherMessages = vm.messages.filter { $0.role == .teacher }
        XCTAssertEqual(teacherMessages.count, 1)
        XCTAssertEqual(teacherMessages[0].text, "Session summary text")
    }
}
