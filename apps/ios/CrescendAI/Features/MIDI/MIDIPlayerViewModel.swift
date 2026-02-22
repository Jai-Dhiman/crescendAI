import SwiftUI
import UniformTypeIdentifiers

@MainActor
@Observable
final class MIDIPlayerViewModel {
    enum State {
        case empty
        case loaded
        case error(String)
    }

    private(set) var state: State = .empty
    let engine = MIDIEngine()
    var showFilePicker = false

    func setError(_ message: String) {
        state = .error(message)
    }

    func loadFile(url: URL) {
        do {
            // Gain access to security-scoped resource
            let accessing = url.startAccessingSecurityScopedResource()
            defer {
                if accessing { url.stopAccessingSecurityScopedResource() }
            }

            try engine.loadFile(url: url)
            state = .loaded
        } catch {
            state = .error("Failed to load MIDI file: \(error.localizedDescription)")
        }
    }
}
