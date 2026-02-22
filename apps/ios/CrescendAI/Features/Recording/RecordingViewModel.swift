import SwiftUI

@MainActor
@Observable
final class RecordingViewModel {
    enum State {
        case idle
        case recording
        case recorded(URL)
        case uploading
        case error(String)
    }

    private(set) var state: State = .idle
    let recorder = AudioRecorder()
    let player = AudioPlayer()

    var isRecording: Bool { recorder.isRecording }
    var hasRecording: Bool {
        if case .recorded = state { return true }
        return false
    }

    func toggleRecording() {
        if recorder.isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    func startRecording() {
        do {
            try recorder.startRecording()
            state = .recording
        } catch {
            state = .error("Failed to start recording: \(error.localizedDescription)")
        }
    }

    func stopRecording() {
        recorder.stopRecording()
        if let url = recorder.recordingURL {
            state = .recorded(url)
            do {
                try player.load(url: url)
            } catch {
                state = .error("Failed to load recording: \(error.localizedDescription)")
            }
        }
    }

    func discardRecording() {
        player.stop()
        if case .recorded(let url) = state {
            try? FileManager.default.removeItem(at: url)
        }
        state = .idle
    }

    func uploadAndAnalyze() async -> AnalysisResult? {
        guard case .recorded(let url) = state else { return nil }

        state = .uploading
        do {
            let data = try Data(contentsOf: url)
            let uploaded = try await APIClient.shared.upload(audioData: data, title: "My Recording")
            let result = try await APIClient.shared.analyze(performanceId: uploaded.id)
            return result
        } catch {
            state = .error("Analysis failed: \(error.localizedDescription)")
            return nil
        }
    }
}
