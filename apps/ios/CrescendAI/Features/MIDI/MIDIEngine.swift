import AVFoundation
import Combine

@MainActor
final class MIDIEngine: ObservableObject {
    @Published private(set) var isPlaying = false
    @Published private(set) var currentPosition: TimeInterval = 0
    @Published private(set) var duration: TimeInterval = 0
    @Published private(set) var loadedFileName: String?

    private var midiPlayer: AVMIDIPlayer?
    private var progressTimer: Timer?

    var soundFontURL: URL? {
        Bundle.main.url(forResource: "piano", withExtension: "sf2")
    }

    func loadFile(url: URL) throws {
        guard let sfURL = soundFontURL else {
            throw MIDIError.soundFontNotFound
        }

        stop()

        let player = try AVMIDIPlayer(contentsOf: url, soundBankURL: sfURL)
        player.prepareToPlay()

        self.midiPlayer = player
        self.duration = player.duration
        self.currentPosition = 0
        self.loadedFileName = url.deletingPathExtension().lastPathComponent
    }

    func play() {
        guard let player = midiPlayer else { return }
        player.play { [weak self] in
            Task { @MainActor [weak self] in
                self?.isPlaying = false
                self?.stopProgressTimer()
            }
        }
        isPlaying = true
        startProgressTimer()
    }

    func pause() {
        midiPlayer?.stop()
        isPlaying = false
        stopProgressTimer()
    }

    func stop() {
        midiPlayer?.stop()
        midiPlayer?.currentPosition = 0
        isPlaying = false
        currentPosition = 0
        stopProgressTimer()
    }

    private func startProgressTimer() {
        progressTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, let player = self.midiPlayer else { return }
                self.currentPosition = player.currentPosition
            }
        }
    }

    private func stopProgressTimer() {
        progressTimer?.invalidate()
        progressTimer = nil
    }
}

enum MIDIError: LocalizedError {
    case soundFontNotFound

    var errorDescription: String? {
        switch self {
        case .soundFontNotFound:
            "Piano sound font not found. Ensure piano.sf2 is bundled with the app."
        }
    }
}
