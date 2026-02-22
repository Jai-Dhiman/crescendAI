import SwiftUI

struct RecordingView: View {
    @State private var viewModel = RecordingViewModel()
    @State private var navigateToAnalysis = false
    @State private var analysisResult: AnalysisResult?

    var body: some View {
        ZStack {
            CrescendColor.background
                .ignoresSafeArea()

            VStack(spacing: CrescendSpacing.space8) {
                Spacer()

                // Level meter
                LevelMeterView(level: viewModel.recorder.currentLevel, isActive: viewModel.isRecording)
                    .frame(height: 120)
                    .padding(.horizontal, CrescendSpacing.space8)

                // Duration
                Text(formatDuration(viewModel.recorder.duration))
                    .font(CrescendFont.displayLG(.medium))
                    .foregroundStyle(CrescendColor.foreground)
                    .monospacedDigit()

                // Status text
                statusText

                Spacer()

                // Controls
                controlsSection

                Spacer()
                    .frame(height: CrescendSpacing.space8)
            }
            .padding(.horizontal, CrescendSpacing.space6)
        }
        .navigationTitle("Record")
        .navigationBarTitleDisplayMode(.inline)
        .navigationDestination(isPresented: $navigateToAnalysis) {
            if let result = analysisResult {
                AnalysisView(result: result)
            }
        }
    }

    @ViewBuilder
    private var statusText: some View {
        switch viewModel.state {
        case .idle:
            Text("Tap to start recording")
                .font(CrescendFont.bodyMD())
                .foregroundStyle(CrescendColor.secondaryText)
        case .recording:
            Text("Recording...")
                .font(CrescendFont.bodyMD())
                .foregroundStyle(CrescendColor.foreground)
        case .recorded:
            Text("Recording complete")
                .font(CrescendFont.bodyMD())
                .foregroundStyle(CrescendColor.foreground)
        case .uploading:
            Text("Analyzing your performance...")
                .font(CrescendFont.bodyMD())
                .foregroundStyle(CrescendColor.secondaryText)
        case .error(let message):
            Text(message)
                .font(CrescendFont.bodySM())
                .foregroundStyle(.red)
                .multilineTextAlignment(.center)
        }
    }

    @ViewBuilder
    private var controlsSection: some View {
        switch viewModel.state {
        case .idle:
            recordButton

        case .recording:
            stopButton

        case .recorded:
            VStack(spacing: CrescendSpacing.space4) {
                playbackControls

                CrescendButton("Analyze Performance", style: .primary, icon: "waveform.badge.magnifyingglass") {
                    Task {
                        if let result = await viewModel.uploadAndAnalyze() {
                            analysisResult = result
                            navigateToAnalysis = true
                        }
                    }
                }

                CrescendButton("Discard", style: .ghost) {
                    viewModel.discardRecording()
                }
            }

        case .uploading:
            ProgressView()
                .tint(CrescendColor.foreground)

        case .error:
            CrescendButton("Try Again", style: .secondary) {
                viewModel.discardRecording()
            }
        }
    }

    private var recordButton: some View {
        Button(action: { viewModel.toggleRecording() }) {
            Circle()
                .fill(CrescendColor.foreground)
                .frame(width: 72, height: 72)
                .overlay(
                    Image(systemName: "mic.fill")
                        .font(.system(size: 28, weight: .medium))
                        .foregroundStyle(CrescendColor.background)
                )
        }
        .buttonStyle(CrescendPressStyle())
    }

    private var stopButton: some View {
        Button(action: { viewModel.toggleRecording() }) {
            Circle()
                .fill(CrescendColor.foreground)
                .frame(width: 72, height: 72)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(CrescendColor.background)
                        .frame(width: 24, height: 24)
                )
        }
        .buttonStyle(CrescendPressStyle())
    }

    private var playbackControls: some View {
        CrescendCard {
            VStack(spacing: CrescendSpacing.space3) {
                // Progress bar
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(CrescendColor.subtleFill)
                            .frame(height: 4)

                        RoundedRectangle(cornerRadius: 2)
                            .fill(CrescendColor.foreground)
                            .frame(
                                width: viewModel.player.duration > 0
                                    ? geometry.size.width * (viewModel.player.currentTime / viewModel.player.duration)
                                    : 0,
                                height: 4
                            )
                    }
                }
                .frame(height: 4)

                HStack {
                    Text(formatDuration(viewModel.player.currentTime))
                        .font(CrescendFont.labelSM())
                        .foregroundStyle(CrescendColor.secondaryText)
                        .monospacedDigit()

                    Spacer()

                    HStack(spacing: CrescendSpacing.space4) {
                        CrescendIconButton(icon: viewModel.player.isPlaying ? "pause.fill" : "play.fill") {
                            if viewModel.player.isPlaying {
                                viewModel.player.pause()
                            } else {
                                viewModel.player.play()
                            }
                        }

                        CrescendIconButton(icon: "stop.fill") {
                            viewModel.player.stop()
                        }
                    }

                    Spacer()

                    Text(formatDuration(viewModel.player.duration))
                        .font(CrescendFont.labelSM())
                        .foregroundStyle(CrescendColor.secondaryText)
                        .monospacedDigit()
                }
            }
        }
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        let tenths = Int((seconds.truncatingRemainder(dividingBy: 1)) * 10)
        return String(format: "%d:%02d.%d", mins, secs, tenths)
    }
}

private struct CrescendPressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.92 : 1.0)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

struct LevelMeterView: View {
    let level: Float
    let isActive: Bool

    private let barCount = 30

    var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<barCount, id: \.self) { index in
                let barLevel = barHeight(for: index)
                RoundedRectangle(cornerRadius: 1.5)
                    .fill(CrescendColor.foreground.opacity(isActive ? 0.8 : 0.15))
                    .frame(width: 3, height: barLevel)
                    .animation(
                        .easeOut(duration: 0.08).delay(Double(index) * 0.005),
                        value: level
                    )
            }
        }
    }

    private func barHeight(for index: Int) -> CGFloat {
        guard isActive else { return 8 }
        let center = Float(barCount) / 2.0
        let distance = abs(Float(index) - center) / center
        let envelope = 1.0 - (distance * distance)
        let height = max(8, CGFloat(level * envelope * 100 + 4))
        return min(height, 120)
    }
}

#Preview {
    NavigationStack {
        RecordingView()
    }
    .crescendTheme()
}
