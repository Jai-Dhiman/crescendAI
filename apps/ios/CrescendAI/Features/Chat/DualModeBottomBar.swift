import SwiftUI

struct DualModeBottomBar: View {
    @Binding var text: String
    let isRecording: Bool
    let waveformLevel: Float
    let sessionDuration: TimeInterval
    let onSend: () -> Void
    let onStartPractice: () -> Void
    let onPause: () -> Void
    let onStop: () -> Void
    let onAsk: () -> Void

    var body: some View {
        VStack(spacing: CrescendSpacing.space2) {
            if isRecording {
                recordingMode
            } else {
                idleMode
            }
        }
        .padding(.horizontal, CrescendSpacing.space4)
        .padding(.vertical, CrescendSpacing.space2)
        .background(CrescendColor.background)
        .animation(.easeInOut(duration: 0.3), value: isRecording)
    }

    // MARK: - Idle Mode

    private var idleMode: some View {
        VStack(spacing: CrescendSpacing.space3) {
            // Start Practicing button
            Button(action: onStartPractice) {
                HStack(spacing: CrescendSpacing.space2) {
                    Image(systemName: "mic.fill")
                        .font(.system(size: 16, weight: .medium))
                    Text("Start Practicing")
                        .font(CrescendFont.labelLG())
                }
                .foregroundStyle(CrescendColor.background)
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(CrescendColor.foreground)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .buttonStyle(CrescendPressStyle())

            // Text input
            HStack(spacing: CrescendSpacing.space2) {
                TextField("Ask your teacher...", text: $text, axis: .vertical)
                    .font(CrescendFont.bodyMD())
                    .foregroundStyle(CrescendColor.foreground)
                    .lineLimit(1...4)
                    .padding(.horizontal, CrescendSpacing.space3)
                    .padding(.vertical, CrescendSpacing.space2)
                    .background(CrescendColor.inputBackground)
                    .clipShape(RoundedRectangle(cornerRadius: 20))
                    .overlay(
                        RoundedRectangle(cornerRadius: 20)
                            .stroke(CrescendColor.border, lineWidth: 1)
                    )
                    .onSubmit { onSend() }

                if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Button(action: onSend) {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundStyle(CrescendColor.background)
                            .frame(width: 36, height: 36)
                            .background(CrescendColor.foreground)
                            .clipShape(Circle())
                    }
                    .buttonStyle(CrescendPressStyle())
                }
            }
        }
    }

    // MARK: - Recording Mode

    private var recordingMode: some View {
        VStack(spacing: CrescendSpacing.space3) {
            // Duration label
            Text(formatDuration(sessionDuration))
                .font(CrescendFont.labelMD())
                .foregroundStyle(CrescendColor.secondaryText)

            // Waveform
            WaveformView(level: waveformLevel, duration: sessionDuration)
                .frame(height: 48)

            // Controls
            HStack(spacing: CrescendSpacing.space6) {
                // Pause
                Button(action: onPause) {
                    Circle()
                        .stroke(CrescendColor.foreground, lineWidth: 1.5)
                        .frame(width: 44, height: 44)
                        .overlay {
                            Image(systemName: "pause.fill")
                                .font(.system(size: 16, weight: .medium))
                                .foregroundStyle(CrescendColor.foreground)
                        }
                }
                .buttonStyle(CrescendPressStyle())

                // Stop
                Button(action: onStop) {
                    Circle()
                        .fill(CrescendColor.foreground)
                        .frame(width: 44, height: 44)
                        .overlay {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(CrescendColor.background)
                                .frame(width: 16, height: 16)
                        }
                }
                .buttonStyle(CrescendPressStyle())

                // How was that?
                Button(action: onAsk) {
                    Text("How was that?")
                        .font(CrescendFont.labelLG())
                        .foregroundStyle(CrescendColor.foreground)
                        .padding(.horizontal, CrescendSpacing.space4)
                        .padding(.vertical, CrescendSpacing.space2)
                        .background(CrescendColor.surface)
                        .clipShape(Capsule())
                        .overlay(
                            Capsule().stroke(CrescendColor.border, lineWidth: 1)
                        )
                }
                .buttonStyle(CrescendPressStyle())
            }
        }
        .padding(.vertical, CrescendSpacing.space2)
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", mins, secs)
    }
}

#Preview("Idle") {
    VStack {
        Spacer()
        DualModeBottomBar(
            text: .constant(""),
            isRecording: false,
            waveformLevel: 0,
            sessionDuration: 0,
            onSend: {},
            onStartPractice: {},
            onPause: {},
            onStop: {},
            onAsk: {}
        )
    }
    .background(CrescendColor.background)
    .crescendTheme()
}

#Preview("Recording") {
    VStack {
        Spacer()
        DualModeBottomBar(
            text: .constant(""),
            isRecording: true,
            waveformLevel: 0.6,
            sessionDuration: 127,
            onSend: {},
            onStartPractice: {},
            onPause: {},
            onStop: {},
            onAsk: {}
        )
    }
    .background(CrescendColor.background)
    .crescendTheme()
}
