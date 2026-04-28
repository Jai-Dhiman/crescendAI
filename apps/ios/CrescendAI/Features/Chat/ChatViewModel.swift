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

    private(set) var manager: PracticeSessionManager?
    private var modelContext: ModelContext?

    var currentLevel: Float {
        manager?.currentLevel ?? 0
    }

    var sessionDuration: TimeInterval {
        manager?.currentSession?.duration ?? 0
    }

    var greeting: String {
        let hour = Calendar.current.component(.hour, from: .now)
        let period: String
        if hour < 12 {
            period = "morning"
        } else if hour < 17 {
            period = "afternoon"
        } else {
            period = "evening"
        }
        return "Good \(period)! Ready to practice?"
    }

    func configure(modelContext: ModelContext) {
        self.modelContext = modelContext
    }

    func startPractice() async {
        guard let modelContext else { return }
        let mgr = PracticeSessionManager(modelContext: modelContext)
        manager = mgr
        do {
            try await mgr.startSession()
            practiceState = .recording
            addSystemMessage("Session started. Play when you're ready.")
        } catch {
            addSystemMessage("Could not start session: \(error.localizedDescription)")
            manager = nil
        }
    }

    func stopPractice() async {
        practiceState = .processing
        addSystemMessage("Ending session and analyzing your practice...")
        
        await manager?.endSession()
        
        // Simulate getting observations (will be replaced by real API call)
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            addObservation(
                text: "Nice work! Your rhythm was steady throughout the session.",
                dimension: "rhythm"
            )
            addObservation(
                text: "Your dynamics showed good contrast. Try increasing the dynamic range even more.",
                dimension: "dynamics"
            )
            practiceState = .idle
        }
    }

    func pausePractice() {
        // Pause is handled by the session manager's audio interruption
    }

    func askForFeedback() {
        practiceState = .processing
        addSystemMessage("Analyzing your playing...")

        // Simulate observation arrival (will be replaced by real /api/ask call)
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            addObservation(
                text: "Your dynamics showed nice contrast between the forte and piano sections. The crescendo in the second phrase built naturally.",
                dimension: "dynamics"
            )
            practiceState = .recording
        }
    }

    func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        messages.append(ChatMessage(role: .user, text: text))
        inputText = ""

        // Echo response placeholder (will be replaced by real LLM call)
        Task {
            try? await Task.sleep(for: .seconds(0.5))
            addSystemMessage("I'll help you with that. Let's work on it in your next session.")
        }
    }

    func addSystemMessage(_ text: String) {
        messages.append(ChatMessage(role: .system, text: text))
    }

    func addObservation(text: String, dimension: String) {
        messages.append(ChatMessage(role: .observation, text: text, dimension: dimension))
    }

    func requestElaboration(for messageId: UUID) {
        guard let index = messages.firstIndex(where: { $0.id == messageId }) else { return }
        // Placeholder elaboration (will call LLM)
        Task {
            try? await Task.sleep(for: .seconds(0.8))
            messages[index].elaboration = "The dynamic range you achieved spans roughly mezzo-piano to forte. To push further, try lifting your wrist slightly before the forte passages to allow more arm weight into the keys."
        }
    }
}
