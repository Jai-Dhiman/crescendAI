import XCTest
@testable import CrescendAI

final class RingBufferTests: XCTestCase {

    // MARK: - Basic Read/Write

    func testWriteAndReadExact() {
        let buffer = RingBuffer(capacity: 100)
        var samples: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        samples.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }

        let result = buffer.read(last: 5)
        XCTAssertEqual(result, [1.0, 2.0, 3.0, 4.0, 5.0])
    }

    func testReadFewerThanWritten() {
        let buffer = RingBuffer(capacity: 100)
        var samples: [Float] = (0..<50).map { Float($0) }
        samples.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }

        let result = buffer.read(last: 10)
        XCTAssertEqual(result, (40..<50).map { Float($0) })
    }

    func testReadMoreThanAvailable() {
        let buffer = RingBuffer(capacity: 100)
        var samples: [Float] = [1.0, 2.0, 3.0]
        samples.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }

        let result = buffer.read(last: 10)
        XCTAssertEqual(result, [1.0, 2.0, 3.0])
    }

    func testEmptyBufferRead() {
        let buffer = RingBuffer(capacity: 100)
        XCTAssertEqual(buffer.read(last: 10), [])
        XCTAssertEqual(buffer.read(last: 0), [])
    }

    // MARK: - Wraparound

    func testWraparound() {
        let buffer = RingBuffer(capacity: 10)
        var samples: [Float] = (0..<15).map { Float($0) }
        samples.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }

        // Capacity is 10, wrote 15: oldest 5 are overwritten
        let result = buffer.read(last: 10)
        XCTAssertEqual(result, (5..<15).map { Float($0) })
    }

    func testWraparoundPartialRead() {
        let buffer = RingBuffer(capacity: 10)
        var samples: [Float] = (0..<15).map { Float($0) }
        samples.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }

        let result = buffer.read(last: 5)
        XCTAssertEqual(result, (10..<15).map { Float($0) })
    }

    func testReadAtExactCapacity() {
        let buffer = RingBuffer(capacity: 10)
        var samples: [Float] = (0..<10).map { Float($0) }
        samples.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }

        let result = buffer.read(last: 10)
        XCTAssertEqual(result, (0..<10).map { Float($0) })
    }

    func testReadMoreThanCapacity() {
        let buffer = RingBuffer(capacity: 10)
        var samples: [Float] = (0..<10).map { Float($0) }
        samples.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }

        let result = buffer.read(last: 20)
        XCTAssertEqual(result.count, 10)
        XCTAssertEqual(result, (0..<10).map { Float($0) })
    }

    func testMultipleWraparounds() {
        let buffer = RingBuffer(capacity: 10)
        // Write 35 samples into capacity-10 buffer (3.5x wraparound)
        var samples: [Float] = (0..<35).map { Float($0) }
        samples.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }

        let result = buffer.read(last: 10)
        XCTAssertEqual(result, (25..<35).map { Float($0) })
    }

    // MARK: - Sequential Writes

    func testMultipleSequentialWrites() {
        let buffer = RingBuffer(capacity: 100)

        for i in 0..<10 {
            var samples: [Float] = (0..<5).map { Float(i * 5 + Int($0)) }
            samples.withUnsafeBufferPointer {
                buffer.write($0.baseAddress!, count: $0.count)
            }
        }

        let result = buffer.read(last: 50)
        XCTAssertEqual(result, (0..<50).map { Float($0) })
    }

    func testSequentialWritesWithWraparound() {
        let buffer = RingBuffer(capacity: 10)

        // Write 3 samples at a time, 5 times (15 total, capacity 10)
        for i in 0..<5 {
            var samples: [Float] = (0..<3).map { Float(i * 3 + Int($0)) }
            samples.withUnsafeBufferPointer {
                buffer.write($0.baseAddress!, count: $0.count)
            }
        }

        let result = buffer.read(last: 10)
        XCTAssertEqual(result, (5..<15).map { Float($0) })
    }

    // MARK: - totalSamplesWritten

    func testTotalSamplesWritten() {
        let buffer = RingBuffer(capacity: 10)
        XCTAssertEqual(buffer.totalSamplesWritten, 0)

        var s1: [Float] = [1.0, 2.0, 3.0]
        s1.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }
        XCTAssertEqual(buffer.totalSamplesWritten, 3)

        var s2: [Float] = [4.0, 5.0]
        s2.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }
        XCTAssertEqual(buffer.totalSamplesWritten, 5)
    }

    func testTotalSamplesWrittenSurvivesWraparound() {
        let buffer = RingBuffer(capacity: 10)
        var samples: [Float] = (0..<25).map { Float($0) }
        samples.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }

        // Total written tracks all samples, not just what's in the buffer
        XCTAssertEqual(buffer.totalSamplesWritten, 25)
    }

    // MARK: - Reset

    func testReset() {
        let buffer = RingBuffer(capacity: 100)
        var samples: [Float] = [1.0, 2.0, 3.0]
        samples.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }

        buffer.reset()

        XCTAssertEqual(buffer.totalSamplesWritten, 0)
        XCTAssertEqual(buffer.read(last: 10), [])
    }

    func testWriteAfterReset() {
        let buffer = RingBuffer(capacity: 10)
        var s1: [Float] = [99.0, 98.0, 97.0]
        s1.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }

        buffer.reset()

        var s2: [Float] = [1.0, 2.0]
        s2.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }

        let result = buffer.read(last: 10)
        XCTAssertEqual(result, [1.0, 2.0])
        XCTAssertEqual(buffer.totalSamplesWritten, 2)
    }

    // MARK: - Realistic Audio Sizes

    func testRealisticChunkExtraction() {
        // Simulate 24kHz audio, 5-minute buffer, 15-second chunk
        let sampleRate = 24_000
        let bufferCapacity = sampleRate * 300  // 5 minutes
        let chunkSamples = sampleRate * 15     // 15 seconds

        let buffer = RingBuffer(capacity: bufferCapacity)

        // Write 20 seconds of audio in small batches (simulating tap callbacks)
        let batchSize = 4096
        var totalWritten = 0
        let targetSamples = sampleRate * 20

        while totalWritten < targetSamples {
            let count = min(batchSize, targetSamples - totalWritten)
            var batch = (0..<count).map { Float(totalWritten + $0) }
            batch.withUnsafeBufferPointer {
                buffer.write($0.baseAddress!, count: $0.count)
            }
            totalWritten += count
        }

        XCTAssertEqual(buffer.totalSamplesWritten, targetSamples)

        // Extract a 15-second chunk (last 360,000 samples)
        let chunk = buffer.read(last: chunkSamples)
        XCTAssertEqual(chunk.count, chunkSamples)

        // Verify the chunk contains the most recent 15s of audio
        let expectedStart = Float(targetSamples - chunkSamples)
        XCTAssertEqual(chunk.first, expectedStart)
        XCTAssertEqual(chunk.last, Float(targetSamples - 1))
    }

    // MARK: - Thread Safety

    func testConcurrentReadWrite() {
        let buffer = RingBuffer(capacity: 24_000 * 30) // 30 seconds
        let writeExpectation = expectation(description: "Writes complete")
        let readExpectation = expectation(description: "Reads complete")

        let writeIterations = 200
        let writeSize = 4096

        // Simulate audio thread writing 4096-sample buffers
        DispatchQueue.global(qos: .userInteractive).async {
            for _ in 0..<writeIterations {
                var samples = [Float](repeating: 1.0, count: writeSize)
                samples.withUnsafeBufferPointer {
                    buffer.write($0.baseAddress!, count: $0.count)
                }
            }
            writeExpectation.fulfill()
        }

        // Simulate main thread reading chunks
        DispatchQueue.global(qos: .userInitiated).async {
            for _ in 0..<200 {
                _ = buffer.read(last: 24_000) // 1 second of audio
            }
            readExpectation.fulfill()
        }

        wait(for: [writeExpectation, readExpectation], timeout: 30.0)
        XCTAssertEqual(buffer.totalSamplesWritten, writeIterations * writeSize)
    }

    func testConcurrentReadsDoNotCrash() {
        let buffer = RingBuffer(capacity: 1000)
        var samples: [Float] = (0..<500).map { Float($0) }
        samples.withUnsafeBufferPointer {
            buffer.write($0.baseAddress!, count: $0.count)
        }

        let expectation = expectation(description: "Concurrent reads complete")
        expectation.expectedFulfillmentCount = 4

        for _ in 0..<4 {
            DispatchQueue.global().async {
                for _ in 0..<1000 {
                    let result = buffer.read(last: 100)
                    XCTAssertEqual(result.count, 100)
                }
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 10.0)
    }
}
