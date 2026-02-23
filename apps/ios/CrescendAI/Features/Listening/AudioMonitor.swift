import AVFoundation

@MainActor
final class AudioMonitor: ObservableObject {
    @Published private(set) var currentLevel: Float = 0
    @Published private(set) var isSilent: Bool = true
    @Published private(set) var isRunning: Bool = false

    private var audioEngine: AVAudioEngine?
    private var silenceTimer: Timer?

    /// How many seconds of silence before `isSilent` becomes true.
    var silenceThreshold: TimeInterval = 3.0

    /// Level below which audio is considered silence.
    private let silenceLevel: Float = 0.02

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
        try session.setActive(true)

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let format = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            guard let channelData = buffer.floatChannelData?[0] else { return }
            let frameLength = Int(buffer.frameLength)
            var sum: Float = 0
            for i in 0..<frameLength {
                sum += channelData[i] * channelData[i]
            }
            let rms = sqrtf(sum / Float(frameLength))
            let level = max(0, min(1, rms * 5))

            Task { @MainActor [weak self] in
                self?.updateLevel(level)
            }
        }

        try engine.start()
        self.audioEngine = engine
        self.isRunning = true
        self.isSilent = true
    }

    func stop() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        silenceTimer?.invalidate()
        silenceTimer = nil
        isRunning = false
        currentLevel = 0
        isSilent = true
    }

    private func updateLevel(_ level: Float) {
        currentLevel = level

        if level > silenceLevel {
            // Sound detected -- reset silence timer
            isSilent = false
            silenceTimer?.invalidate()
            silenceTimer = Timer.scheduledTimer(withTimeInterval: silenceThreshold, repeats: false) { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.isSilent = true
                }
            }
        }
    }
}
