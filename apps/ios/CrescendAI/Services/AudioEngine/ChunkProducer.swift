import AVFoundation

enum ChunkProducerError: Error {
    case formatCreationFailed
    case bufferCreationFailed
    case encodingFailed(underlying: Error)
}

/// Produces 15-second audio chunks from the RingBuffer on a repeating timer.
/// Each chunk is encoded to AAC and optionally run through an InferenceProvider.
/// Publishes chunks via an AsyncStream for consumers.
@MainActor
final class ChunkProducer {
    private(set) var isProducing = false

    private let ringBuffer: RingBuffer
    private let sampleRate: Int = 24_000
    private let chunkDurationSeconds: Int = 15
    private let inferenceProvider: (any InferenceProvider)?

    private var producerTask: Task<Void, Never>?
    private var chunkIndex: Int = 0
    private var sessionId: UUID = UUID()

    private let _continuation: AsyncStream<AudioChunk>.Continuation
    let chunkStream: AsyncStream<AudioChunk>

    init(ringBuffer: RingBuffer, inferenceProvider: (any InferenceProvider)? = nil) {
        self.ringBuffer = ringBuffer
        self.inferenceProvider = inferenceProvider
        let (stream, continuation) = AsyncStream.makeStream(of: AudioChunk.self)
        self.chunkStream = stream
        self._continuation = continuation
    }

    static func rms(samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sumOfSquares = samples.reduce(0) { $0 + $1 * $1 }
        return sqrt(sumOfSquares / Float(samples.count))
    }

    func start(sessionId: UUID, startDate: Date) {
        guard !isProducing else { return }

        self.sessionId = sessionId
        self.chunkIndex = 0
        self.isProducing = true

        producerTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(chunkDurationSeconds))
                guard !Task.isCancelled else { break }
                await produceChunk()
            }
        }
    }

    func stop() {
        producerTask?.cancel()
        producerTask = nil
        isProducing = false
        _continuation.finish()
    }

    private func produceChunk() async {
        let samplesNeeded = chunkDurationSeconds * sampleRate
        let samples = ringBuffer.read(last: samplesNeeded)

        // Skip if too few samples (less than 1 second)
        guard samples.count >= sampleRate else { return }
        guard Self.rms(samples: samples) >= 0.01 else { return }

        let startOffset = TimeInterval(chunkIndex * chunkDurationSeconds)
        let actualDuration = TimeInterval(samples.count) / TimeInterval(sampleRate)

        // Encode to AAC file
        var fileURL: URL?
        do {
            fileURL = try encodeToAAC(samples: samples)
        } catch {
            // AAC encoding failure is non-fatal; chunk is still useful for inference
        }

        // Run inference if provider is available
        var inferenceResult: InferenceResult?
        if let provider = inferenceProvider {
            do {
                inferenceResult = try await provider.infer(samples: samples, sampleRate: sampleRate)
            } catch {
                // Inference failure is non-fatal
            }
        }

        let chunk = AudioChunk(
            sessionId: sessionId,
            index: chunkIndex,
            startOffset: startOffset,
            duration: actualDuration,
            localFileURL: fileURL,
            inferenceResult: inferenceResult
        )

        _continuation.yield(chunk)
        chunkIndex += 1
    }

    private func encodeToAAC(samples: [Float]) throws -> URL {
        guard let pcmFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            throw ChunkProducerError.formatCreationFailed
        }

        guard let pcmBuffer = AVAudioPCMBuffer(
            pcmFormat: pcmFormat,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw ChunkProducerError.bufferCreationFailed
        }

        samples.withUnsafeBufferPointer { src in
            pcmBuffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }
        pcmBuffer.frameLength = AVAudioFrameCount(samples.count)

        let chunkDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("crescendai_chunks", isDirectory: true)
        try FileManager.default.createDirectory(at: chunkDir, withIntermediateDirectories: true)

        let fileURL = chunkDir.appendingPathComponent(
            "chunk_\(sessionId.uuidString)_\(chunkIndex).m4a"
        )

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: Double(sampleRate),
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 128_000,
        ]

        do {
            let file = try AVAudioFile(
                forWriting: fileURL,
                settings: settings,
                commonFormat: .pcmFormatFloat32,
                interleaved: false
            )
            try file.write(from: pcmBuffer)
        } catch {
            throw ChunkProducerError.encodingFailed(underlying: error)
        }

        return fileURL
    }
}
