import AVFoundation
import Combine

@MainActor
final class AudioPlayer: ObservableObject {
    @Published private(set) var isPlaying = false
    @Published private(set) var currentTime: TimeInterval = 0
    @Published private(set) var duration: TimeInterval = 0

    private var player: AVAudioPlayer?
    private var progressTimer: Timer?

    func load(url: URL) throws {
        let player = try AVAudioPlayer(contentsOf: url)
        player.prepareToPlay()
        self.player = player
        self.duration = player.duration
        self.currentTime = 0
    }

    func play() {
        guard let player else { return }
        player.play()
        isPlaying = true
        startProgressTimer()
    }

    func pause() {
        player?.pause()
        isPlaying = false
        stopProgressTimer()
    }

    func stop() {
        player?.stop()
        player?.currentTime = 0
        isPlaying = false
        currentTime = 0
        stopProgressTimer()
    }

    func seek(to time: TimeInterval) {
        player?.currentTime = time
        currentTime = time
    }

    private func startProgressTimer() {
        progressTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, let player = self.player else { return }
                self.currentTime = player.currentTime
                if !player.isPlaying {
                    self.isPlaying = false
                    self.stopProgressTimer()
                }
            }
        }
    }

    private func stopProgressTimer() {
        progressTimer?.invalidate()
        progressTimer = nil
    }
}
