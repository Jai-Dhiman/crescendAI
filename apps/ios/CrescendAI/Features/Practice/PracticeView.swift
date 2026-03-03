import SwiftData
import SwiftUI

struct PracticeView: View {
    @Environment(\.modelContext) private var modelContext
    @State private var manager: PracticeSessionManager?
    @State private var error: String?

    var body: some View {
        VStack(spacing: CrescendSpacing.space6) {
            Spacer()

            // Session state indicator
            Text(stateLabel)
                .font(CrescendFont.headingLG())
                .foregroundStyle(CrescendColor.foreground)

            // Audio level meter
            if let manager, manager.state == .recording {
                levelMeter(level: manager.currentLevel)

                Text("\(manager.currentSession?.chunks.count ?? 0) chunks recorded")
                    .font(CrescendFont.bodyMD())
                    .foregroundStyle(CrescendColor.secondaryText)

                if let session = manager.currentSession {
                    Text(formatDuration(session.duration))
                        .font(CrescendFont.displayMD())
                        .foregroundStyle(CrescendColor.foreground)
                        .monospacedDigit()
                }
            }

            if let error {
                Text(error)
                    .font(CrescendFont.bodySM())
                    .foregroundStyle(.red)
            }

            Spacer()

            // Controls
            HStack(spacing: CrescendSpacing.space4) {
                if manager?.state == .recording {
                    CrescendButton("End Session", style: .secondary) {
                        Task { await endSession() }
                    }
                } else {
                    CrescendButton("Start Practice", style: .primary) {
                        Task { await startSession() }
                    }
                }
            }
            .padding(.bottom, CrescendSpacing.space8)
        }
        .padding(.horizontal, CrescendSpacing.space6)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(CrescendColor.background)
    }

    private var stateLabel: String {
        guard let manager else { return "Ready to Practice" }
        switch manager.state {
        case .idle: return "Ready to Practice"
        case .recording: return "Listening..."
        case .paused: return "Paused"
        case .ended: return "Session Complete"
        }
    }

    private func levelMeter(level: Float) -> some View {
        GeometryReader { geo in
            RoundedRectangle(cornerRadius: 4)
                .fill(CrescendColor.foreground.opacity(0.15))
                .frame(width: geo.size.width, height: 8)
                .overlay(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(CrescendColor.foreground)
                        .frame(width: geo.size.width * CGFloat(level), height: 8)
                        .animation(.linear(duration: 0.1), value: level)
                }
        }
        .frame(height: 8)
    }

    private func formatDuration(_ interval: TimeInterval) -> String {
        let minutes = Int(interval) / 60
        let seconds = Int(interval) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }

    private func startSession() async {
        error = nil
        let mgr = PracticeSessionManager(modelContext: modelContext)
        manager = mgr
        do {
            try await mgr.startSession()
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func endSession() async {
        await manager?.endSession()
    }
}

#Preview {
    PracticeView()
        .crescendTheme()
        .modelContainer(for: [PracticeSessionRecord.self, ChunkResultRecord.self], inMemory: true)
}
