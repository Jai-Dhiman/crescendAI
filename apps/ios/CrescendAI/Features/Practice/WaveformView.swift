import SwiftUI

struct WaveformView: View {
    let level: Float
    let duration: TimeInterval

    private let barCount = 48
    @State private var barHeights: [CGFloat] = Array(repeating: 0.1, count: 48)
    @State private var breathingPhase: CGFloat = 0

    var body: some View {
        VStack(spacing: CrescendSpacing.space4) {
            Text("Listening...")
                .font(CrescendFont.bodySM())
                .foregroundStyle(CrescendColor.secondaryText)

            waveformBars
                .frame(height: 120)

            Text(formatDuration(duration))
                .font(CrescendFont.displayMD())
                .foregroundStyle(CrescendColor.foreground)
                .monospacedDigit()
        }
        .onChange(of: level) { _, newLevel in
            updateBars(level: newLevel)
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 2.0).repeatForever(autoreverses: true)) {
                breathingPhase = 1.0
            }
        }
    }

    private var waveformBars: some View {
        GeometryReader { geo in
            let barWidth = max(2, (geo.size.width - CGFloat(barCount - 1) * 2) / CGFloat(barCount))
            let totalWidth = CGFloat(barCount) * barWidth + CGFloat(barCount - 1) * 2
            let offsetX = (geo.size.width - totalWidth) / 2

            HStack(spacing: 2) {
                ForEach(0..<barCount, id: \.self) { index in
                    let height = barHeights[index]
                    let breathOffset = breathingPhase * 0.05 * sin(CGFloat(index) * 0.3)
                    let finalHeight = max(0.05, min(1.0, height + breathOffset))

                    RoundedRectangle(cornerRadius: barWidth / 2)
                        .fill(barColor(for: finalHeight))
                        .frame(width: barWidth, height: geo.size.height * finalHeight)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            .offset(x: offsetX)
        }
    }

    private func barColor(for height: CGFloat) -> Color {
        if height > 0.15 {
            return CrescendColor.waveformActive
        }
        return CrescendColor.waveformInactive
    }

    private func updateBars(level: Float) {
        withAnimation(.linear(duration: 0.1)) {
            // Shift bars left
            for i in 0..<(barCount - 1) {
                barHeights[i] = barHeights[i + 1]
            }
            // Add new bar from current level with some randomness for visual interest
            let normalizedLevel = CGFloat(min(1.0, level * 3.0))
            let variation = CGFloat.random(in: -0.05...0.05)
            barHeights[barCount - 1] = max(0.05, normalizedLevel + variation)
        }
    }

    private func formatDuration(_ interval: TimeInterval) -> String {
        let minutes = Int(interval) / 60
        let seconds = Int(interval) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

#Preview {
    VStack {
        WaveformView(level: 0.3, duration: 125)
    }
    .padding()
    .background(CrescendColor.background)
    .crescendTheme()
}
