import SwiftUI

@MainActor
@Observable
final class FeedbackChatViewModel {
    private(set) var messages: [ChatMessage] = []
    private(set) var chips: [SuggestionChip] = []
    private(set) var focusDimension: FocusDimension?
    private(set) var shouldDismiss = false

    var isInFocusMode: Bool { focusDimension != nil }
    let demoMode: DemoMode

    /// Tracks which responses have been shown to avoid repeats.
    private var shownResponses: Set<String> = []

    init(demoMode: DemoMode) {
        self.demoMode = demoMode
    }

    /// Called when the user stops playing and the chat appears.
    func onPerformanceEnded() {
        if isInFocusMode {
            appendMessage(DemoScenarios.focusDynamicsFeedback)
            chips = DemoScenarios.focusDynamicsChips
        } else if !shownResponses.contains("firstGreeting") {
            appendMessage(DemoScenarios.firstTimeGreeting)
            chips = DemoScenarios.firstTimeChips
            shownResponses.insert("firstGreeting")
        } else {
            // Subsequent general plays after the first -- re-show first greeting
            appendMessage(DemoScenarios.firstTimeGreeting)
            chips = DemoScenarios.firstTimeChips
        }
    }

    func handleChip(_ chip: SuggestionChip) {
        // Show user's selection as a message
        appendMessage(ChatMessage(sender: .user, blocks: [.text(chip.label)]))

        switch chip.actionKey {
        case "start_focus_dynamics":
            focusDimension = .dynamics
            appendMessage(ChatMessage(sender: .ai, blocks: [
                .text("Focus mode: Dynamics. I'll only give feedback on dynamic contrast and shaping. Play when you're ready."),
            ]))
            chips = [SuggestionChip(label: "Play now", actionKey: "dismiss_chat")]

        case "explain_app":
            appendMessage(DemoScenarios.howItWorks)
            chips = DemoScenarios.howItWorksChips

        case "what_to_work_on":
            appendMessage(DemoScenarios.whatToWorkOn)
            chips = DemoScenarios.whatToWorkOnChips

        case "another_reference":
            appendMessage(DemoScenarios.referenceFollowUp)
            chips = DemoScenarios.referenceFollowUpChips

        case "exit_focus":
            focusDimension = nil
            appendMessage(ChatMessage(sender: .ai, blocks: [
                .text("Exiting focus mode. I'll give general feedback on your next play-through."),
            ]))
            chips = [SuggestionChip(label: "Play now", actionKey: "dismiss_chat")]

        case "done":
            appendMessage(DemoScenarios.doneMessage)
            chips = DemoScenarios.doneChips

        case "dismiss_chat":
            shouldDismiss = true

        default:
            break
        }
    }

    func resetDismiss() {
        shouldDismiss = false
    }

    func clearMessages() {
        messages.removeAll()
        chips = []
    }

    private func appendMessage(_ message: ChatMessage) {
        messages.append(message)
    }
}
