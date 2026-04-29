import Foundation
import SwiftData
import SwiftUI

enum ChatMessageRole: Equatable {
    case user
    case system
    case observation
    case teacher
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: ChatMessageRole
    let text: String
    let dimension: String?
    let timestamp: Date
    var elaboration: String?
    var artifacts: [ArtifactConfig]

    init(
        role: ChatMessageRole,
        text: String,
        dimension: String? = nil,
        timestamp: Date = .now,
        elaboration: String? = nil,
        artifacts: [ArtifactConfig] = []
    ) {
        self.role = role
        self.text = text
        self.dimension = dimension
        self.timestamp = timestamp
        self.elaboration = elaboration
        self.artifacts = artifacts
    }
}

enum PracticeState {
    case idle
    case recording
    case processing
}

@MainActor
@Observable
final class ChatViewModel {
    var messages: [ChatMessage] = []
    var inputText = ""
    var practiceState: PracticeState = .idle

    private var practiceService: (any PracticeSessionServiceProtocol)?
    private var chatService: (any ChatServiceProtocol)?
    private var modelContext: ModelContext?
    nonisolated(unsafe) private var practiceEventTask: Task<Void, Never>?

    deinit {
        practiceEventTask?.cancel()
    }

    var currentLevel: Float { practiceService?.currentLevel ?? 0 }
    var sessionDuration: TimeInterval { practiceService?.elapsedSeconds ?? 0 }

    var greeting: String {
        let hour = Calendar.current.component(.hour, from: .now)
        if hour < 12 { return "Good morning! Ready to practice?" }
        if hour < 17 { return "Good afternoon! Ready to practice?" }
        return "Good evening! Ready to practice?"
    }

    func configure(modelContext: ModelContext) {
        self.modelContext = modelContext
        let practice = PracticeSessionService()
        practice.configure(modelContext: modelContext)
        self.practiceService = practice
        self.chatService = ChatService()
        subscribeToEvents()
    }

    func configureForTesting(
        modelContext: ModelContext,
        practiceService: any PracticeSessionServiceProtocol,
        chatService: any ChatServiceProtocol
    ) {
        self.modelContext = modelContext
        self.practiceService = practiceService
        self.chatService = chatService
        subscribeToEvents()
    }

    func startPractice() async {
        do {
            try await practiceService?.start()
            practiceState = .recording
            addSystemMessage("Session started. Play when you're ready.")
        } catch {
            addSystemMessage("Could not start session: \(error.localizedDescription)")
        }
    }

    func stopPractice() async {
        practiceState = .processing
        addSystemMessage("Ending session...")
        await practiceService?.stop()
        practiceState = .idle
    }

    func pausePractice() {}

    func askForFeedback() async {
        await practiceService?.askForFeedback()
    }

    func sendMessage() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        messages.append(ChatMessage(role: .user, text: text))
        inputText = ""

        guard let chatService else { return }
        let conversationId = practiceService?.conversationId

        var teacherMsgIndex: Int? = nil
        var teacherArtifacts: [ArtifactConfig] = []

        for await event in chatService.send(message: text, conversationId: conversationId) {
            switch event {
            case .start:
                break
            case .delta(let chunk):
                if let idx = teacherMsgIndex {
                    messages[idx] = ChatMessage(
                        role: .teacher,
                        text: messages[idx].text + chunk,
                        artifacts: teacherArtifacts
                    )
                } else {
                    let msg = ChatMessage(role: .teacher, text: chunk, artifacts: [])
                    messages.append(msg)
                    teacherMsgIndex = messages.count - 1
                }
            case .toolResult(let artifacts):
                teacherArtifacts.append(contentsOf: artifacts)
                if let idx = teacherMsgIndex {
                    messages[idx] = ChatMessage(
                        role: .teacher,
                        text: messages[idx].text,
                        artifacts: teacherArtifacts
                    )
                }
            case .done:
                break
            case .error(let msg):
                addSystemMessage("Error: \(msg)")
            case .toolStart:
                break
            }
        }
    }

    func addSystemMessage(_ text: String) {
        messages.append(ChatMessage(role: .system, text: text))
    }

    func requestElaboration(for messageId: UUID) {}

    private func subscribeToEvents() {
        practiceEventTask?.cancel()
        guard let practiceService else { return }
        practiceEventTask = Task {
            for await event in practiceService.eventStream {
                guard !Task.isCancelled else { break }
                handlePracticeEvent(event)
            }
        }
    }

    private func handlePracticeEvent(_ event: PracticeEvent) {
        switch event {
        case .sessionStarted:
            break
        case .observation(let text, let dimension, let artifacts):
            messages.append(ChatMessage(role: .observation, text: text, dimension: dimension, artifacts: artifacts))
        case .synthesis(let text, let artifacts):
            messages.append(ChatMessage(role: .teacher, text: text, artifacts: artifacts))
        case .reconnecting(let attempt):
            addSystemMessage("Reconnecting... (attempt \(attempt))")
        case .error(let msg):
            addSystemMessage("Error: \(msg)")
        case .chunkUploaded:
            break
        case .sessionEnded:
            break
        }
    }
}
