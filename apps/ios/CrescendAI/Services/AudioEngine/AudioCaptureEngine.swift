import AVFoundation
import Observation

enum AudioCaptureError: Error {
    case inputFormatUnavailable
    case targetFormatCreationFailed
    case converterCreationFailed
    case engineStartFailed(underlying: Error)
}

/// Captures audio from the device microphone at 24kHz mono Float32
/// and writes samples into a shared RingBuffer.
@MainActor
@Observable
final class AudioCaptureEngine {
    private(set) var currentLevel: Float = 0
    private(set) var isRunning: Bool = false

    private var engine: AVAudioEngine?
    private var levelTask: Task<Void, Never>?
    let ringBuffer: RingBuffer

    init(ringBuffer: RingBuffer) {
        self.ringBuffer = ringBuffer
    }

    func start() throws {
        guard !isRunning else { return }

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
            throw AudioCaptureError.inputFormatUnavailable
        }

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 24_000,
            channels: 1,
            interleaved: false
        ) else {
            throw AudioCaptureError.targetFormatCreationFailed
        }

        // Set up resampler if hardware format differs from target
        let needsConversion = inputFormat.sampleRate != 24_000 || inputFormat.channelCount != 1
        let resampler: AudioResampler?
        if needsConversion {
            resampler = try AudioResampler(from: inputFormat, to: targetFormat)
        } else {
            resampler = nil
        }

        let ringBuf = self.ringBuffer

        // AsyncStream bridges audio levels from the real-time tap thread
        // to MainActor. The tap handler must be nonisolated (@Sendable)
        // because AVAudioEngine calls it on its real-time audio thread,
        // not on MainActor (Swift 6 enforces this at runtime).
        let (stream, continuation) = AsyncStream.makeStream(of: Float.self)

        let tapHandler: @Sendable (AVAudioPCMBuffer, AVAudioTime) -> Void = { pcmBuffer, _ in
            let channelData: UnsafePointer<Float>
            let frameLength: Int

            if let resampler {
                guard let converted = resampler.convert(pcmBuffer),
                      let convertedData = converted.floatChannelData?[0] else { return }
                channelData = UnsafePointer(convertedData)
                frameLength = Int(converted.frameLength)
            } else {
                guard let rawData = pcmBuffer.floatChannelData?[0] else { return }
                channelData = UnsafePointer(rawData)
                frameLength = Int(pcmBuffer.frameLength)
            }

            guard frameLength > 0 else { return }

            // Write to ring buffer (thread-safe)
            ringBuf.write(channelData, count: frameLength)

            // Calculate RMS for UI level metering
            var sum: Float = 0
            for i in 0..<frameLength {
                sum += channelData[i] * channelData[i]
            }
            let rms = sqrtf(sum / Float(frameLength))
            let level = max(0, min(1, rms * 5))
            continuation.yield(level)
        }
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat, block: tapHandler)

        do {
            try engine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            throw AudioCaptureError.engineStartFailed(underlying: error)
        }

        self.engine = engine
        self.isRunning = true

        levelTask = Task {
            for await level in stream {
                guard !Task.isCancelled else { break }
                self.currentLevel = min(1.0, level)
            }
        }
    }

    func stop() {
        levelTask?.cancel()
        levelTask = nil
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        isRunning = false
        currentLevel = 0
    }
}

// MARK: - AudioResampler

/// Wraps AVAudioConverter for use from the audio tap callback thread.
/// Marked @unchecked Sendable because the converter is only called
/// sequentially from the single audio tap callback thread.
private final class AudioResampler: @unchecked Sendable {
    private let converter: AVAudioConverter
    private let outputFormat: AVAudioFormat

    init(from inputFormat: AVAudioFormat, to outputFormat: AVAudioFormat) throws {
        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw AudioCaptureError.converterCreationFailed
        }
        self.converter = converter
        self.outputFormat = outputFormat
    }

    func convert(_ inputBuffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        let ratio = outputFormat.sampleRate / inputBuffer.format.sampleRate
        let outputFrameCount = AVAudioFrameCount(Double(inputBuffer.frameLength) * ratio)
        guard outputFrameCount > 0 else { return nil }

        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: outputFrameCount
        ) else {
            return nil
        }

        var error: NSError?
        converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if error != nil {
            return nil
        }

        return outputBuffer
    }
}
