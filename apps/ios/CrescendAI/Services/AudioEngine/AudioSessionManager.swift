import AVFoundation

actor AudioSessionManager {
    enum State: Sendable {
        case inactive
        case active
        case interrupted
    }

    private(set) var state: State = .inactive
    private var interruptionTask: Task<Void, Never>?

    let stateStream: AsyncStream<State>
    private let stateContinuation: AsyncStream<State>.Continuation

    init() {
        let (stream, continuation) = AsyncStream.makeStream(of: State.self)
        self.stateStream = stream
        self.stateContinuation = continuation
    }

    func configure() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement)
        try session.setPreferredSampleRate(24_000)
        try session.setActive(true)
        updateState(.active)
        observeInterruptions()
    }

    func deactivate() {
        interruptionTask?.cancel()
        interruptionTask = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        updateState(.inactive)
        stateContinuation.finish()
    }

    func resume() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setActive(true)
        updateState(.active)
    }

    private func updateState(_ newState: State) {
        state = newState
        stateContinuation.yield(newState)
    }

    private func observeInterruptions() {
        interruptionTask?.cancel()
        interruptionTask = Task {
            let notifications = NotificationCenter.default.notifications(
                named: AVAudioSession.interruptionNotification
            )
            for await notification in notifications {
                guard !Task.isCancelled else { break }
                handleInterruption(notification)
            }
        }
    }

    private func handleInterruption(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else {
            return
        }

        switch type {
        case .began:
            updateState(.interrupted)
        case .ended:
            let optionValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt ?? 0
            let options = AVAudioSession.InterruptionOptions(rawValue: optionValue)
            if options.contains(.shouldResume) {
                do {
                    try AVAudioSession.sharedInstance().setActive(true)
                    updateState(.active)
                } catch {
                    // Remain interrupted if reactivation fails
                }
            }
        @unknown default:
            break
        }
    }
}
