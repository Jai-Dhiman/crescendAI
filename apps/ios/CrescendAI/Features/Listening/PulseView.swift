import SwiftUI

struct PulseView: View {
    /// Audio level from 0 to 1
    let level: Float
    /// Whether the monitor is currently hearing sound
    let isActive: Bool

    @State private var phase: Double = 0

    private let timer = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()

    var body: some View {
        Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let baseRadius = min(size.width, size.height) * 0.15
            let maxRadius = min(size.width, size.height) * 0.4
            let amplitude = CGFloat(level)

            // Draw 4 concentric rings with decreasing opacity
            for i in (0..<4).reversed() {
                let layerFraction = CGFloat(i) / 3.0
                let layerAmplitude = amplitude * (1.0 - layerFraction * 0.3)
                let radius = baseRadius + (maxRadius - baseRadius) * layerAmplitude
                let opacity = isActive ? (0.6 - layerFraction * 0.15) : 0.08

                var path = Path()
                let points = 120
                for j in 0..<points {
                    let angle = (Double(j) / Double(points)) * .pi * 2
                    // Organic undulation: combine multiple sine waves
                    let wave1 = sin(angle * 3 + phase + Double(i) * 0.5) * Double(layerAmplitude) * 0.12
                    let wave2 = sin(angle * 5 - phase * 0.7 + Double(i)) * Double(layerAmplitude) * 0.06
                    let r = radius * (1.0 + wave1 + wave2)

                    let x = center.x + r * cos(angle)
                    let y = center.y + r * sin(angle)

                    if j == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
                path.closeSubpath()

                context.fill(path, with: .color(CrescendColor.foreground.opacity(opacity)))
            }
        }
        .onReceive(timer) { _ in
            let speed = isActive ? 0.03 + Double(level) * 0.02 : 0.008
            phase += speed
        }
        .animation(.easeOut(duration: 0.15), value: level)
    }
}

#Preview("Pulse - Silent") {
    PulseView(level: 0, isActive: false)
        .frame(height: 300)
        .background(CrescendColor.background)
}

#Preview("Pulse - Active") {
    PulseView(level: 0.6, isActive: true)
        .frame(height: 300)
        .background(CrescendColor.background)
}
