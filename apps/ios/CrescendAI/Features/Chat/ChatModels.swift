import Foundation

enum ChatMessageSender {
    case ai
    case user
}

/// A single element within a chat message.
/// Messages are composed of one or more of these blocks.
enum ChatBlock: Identifiable {
    case text(String)
    case dimensionCard(label: String, score: Double, interpretation: String)
    case referencePlayback(label: String, audioFileName: String)
    case musicSnippet(imageAssetName: String, caption: String)

    var id: String {
        switch self {
        case .text(let s): return "text-\(s.prefix(20).hashValue)"
        case .dimensionCard(let label, _, _): return "dim-\(label)"
        case .referencePlayback(let label, _): return "ref-\(label)"
        case .musicSnippet(let name, _): return "img-\(name)"
        }
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let sender: ChatMessageSender
    let blocks: [ChatBlock]
    let timestamp = Date()
}

/// Suggestion chips shown at the bottom of the chat.
struct SuggestionChip: Identifiable, Hashable {
    let id = UUID()
    let label: String
    /// Key used to look up the action in the scenario handler.
    let actionKey: String
}

/// Focus mode targets a specific performance dimension.
enum FocusDimension: String, CaseIterable {
    case dynamics = "Dynamics"
    case articulation = "Articulation"
    case pedaling = "Pedaling"
    case timing = "Timing"
    case tone = "Tone"
}

/// Which demo scenario the app is currently running.
enum DemoMode {
    case firstTime
    case returning
}
