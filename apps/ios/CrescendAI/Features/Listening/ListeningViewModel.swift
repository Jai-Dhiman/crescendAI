import SwiftUI
import Combine

@MainActor
@Observable
final class ListeningViewModel {
    let audioMonitor = AudioMonitor()
    let chatViewModel: FeedbackChatViewModel

    var showChat = false
    private(set) var hasHeardAudio = false

    /// Tracks whether the user has played at least once before the chat triggers.
    private var wasPlaying = false

    private var cancellables = Set<AnyCancellable>()

    let demoMode: DemoMode

    var contextPrompt: String {
        if let focus = chatViewModel.focusDimension {
            return "Focusing on: \(focus.rawValue)"
        }
        switch demoMode {
        case .firstTime:
            return "Play anything -- I'm listening."
        case .returning:
            return DemoScenarios.returningGreeting
        }
    }

    init(demoMode: DemoMode = .firstTime) {
        self.demoMode = demoMode
        self.chatViewModel = FeedbackChatViewModel(demoMode: demoMode)
        observeAudioState()
    }

    func startListening() {
        do {
            try audioMonitor.start()
        } catch {
            // In prototype, we log but don't block the UI
            print("AudioMonitor failed to start: \(error)")
        }
    }

    func stopListening() {
        audioMonitor.stop()
    }

    func dismissChat() {
        showChat = false
        chatViewModel.resetDismiss()
    }

    private func observeAudioState() {
        // Watch for silence transitions: playing -> silent = trigger chat
        audioMonitor.$isSilent
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isSilent in
                guard let self else { return }
                if !isSilent {
                    self.wasPlaying = true
                    self.hasHeardAudio = true
                } else if isSilent && self.wasPlaying {
                    // Silence after playing -- show feedback
                    self.wasPlaying = false
                    self.triggerFeedback()
                }
            }
            .store(in: &cancellables)
    }

    private func triggerFeedback() {
        chatViewModel.clearMessages()
        chatViewModel.onPerformanceEnded()
        showChat = true
    }
}
