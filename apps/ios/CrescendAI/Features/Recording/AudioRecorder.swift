import AVFoundation
import Combine

@MainActor
final class AudioRecorder: ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var currentLevel: Float = 0
    @Published private(set) var duration: TimeInterval = 0
    @Published private(set) var recordingURL: URL?

    private var audioEngine: AVAudioEngine?
    private var audioFile: AVAudioFile?
    private var durationTimer: Timer?

    func startRecording() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setActive(true)

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Create output file in documents directory
        let url = Self.newRecordingURL()
        let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 44100,
            channels: 1,
            interleaved: false
        )!

        let file = try AVAudioFile(
            forWriting: url,
            settings: outputFormat.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, _ in
            // Calculate RMS level
            guard let channelData = buffer.floatChannelData?[0] else { return }
            let frameLength = Int(buffer.frameLength)
            var sum: Float = 0
            for i in 0..<frameLength {
                sum += channelData[i] * channelData[i]
            }
            let rms = sqrtf(sum / Float(frameLength))
            let level = max(0, min(1, rms * 5)) // Normalize to 0-1

            Task { @MainActor [weak self] in
                self?.currentLevel = level
            }

            // Write to file (convert format if needed)
            if let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity: buffer.frameCapacity
            ) {
                let converter = AVAudioConverter(from: inputFormat, to: outputFormat)
                var error: NSError?
                converter?.convert(to: convertedBuffer, error: &error) { _, outStatus in
                    outStatus.pointee = .haveData
                    return buffer
                }
                if error == nil {
                    try? file.write(from: convertedBuffer)
                }
            }
        }

        try engine.start()

        self.audioEngine = engine
        self.audioFile = file
        self.recordingURL = url
        self.isRecording = true
        self.duration = 0

        durationTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.duration += 0.1
            }
        }
    }

    func stopRecording() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        audioFile = nil

        durationTimer?.invalidate()
        durationTimer = nil

        isRecording = false
        currentLevel = 0
    }

    static func newRecordingURL() -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let name = "recording_\(formatter.string(from: Date())).m4a"
        return documents.appendingPathComponent(name)
    }
}
