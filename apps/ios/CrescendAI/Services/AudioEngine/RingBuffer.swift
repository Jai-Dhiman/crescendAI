import os

/// Thread-safe circular buffer for Float32 PCM audio samples.
/// Capacity: 5 minutes at 24kHz (7,200,000 samples, ~29 MB).
/// Uses OSAllocatedUnfairLock for audio-thread-safe synchronization.
final class RingBuffer: @unchecked Sendable {
    private let capacity: Int
    private var buffer: [Float]
    private var writeIndex: Int = 0
    private var _totalSamplesWritten: Int = 0
    private let lock = OSAllocatedUnfairLock()

    init(capacity: Int) {
        self.capacity = capacity
        self.buffer = [Float](repeating: 0, count: capacity)
    }

    /// Appends PCM samples from the audio tap callback.
    func write(_ samples: UnsafePointer<Float>, count: Int) {
        lock.withLockUnchecked {
            var idx = writeIndex
            for i in 0..<count {
                buffer[idx] = samples[i]
                idx += 1
                if idx >= capacity { idx = 0 }
            }
            writeIndex = idx
            _totalSamplesWritten += count
        }
    }

    /// Reads the most recent N samples. Returns fewer if not enough have been written.
    func read(last sampleCount: Int) -> [Float] {
        lock.withLockUnchecked {
            let available = min(sampleCount, min(_totalSamplesWritten, capacity))
            guard available > 0 else { return [] }

            var result = [Float](repeating: 0, count: available)
            var readIndex = (writeIndex - available + capacity) % capacity

            for i in 0..<available {
                result[i] = buffer[readIndex]
                readIndex += 1
                if readIndex >= capacity { readIndex = 0 }
            }

            return result
        }
    }

    /// Total samples written since creation or last reset.
    var totalSamplesWritten: Int {
        lock.withLockUnchecked { _totalSamplesWritten }
    }

    /// Resets the write position and sample counter. Call before starting a new session.
    func reset() {
        lock.withLockUnchecked {
            writeIndex = 0
            _totalSamplesWritten = 0
        }
    }
}
