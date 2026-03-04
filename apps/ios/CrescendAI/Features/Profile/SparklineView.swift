import SwiftUI

struct SparklineView: View {
    let values: [Double]
    let color: Color
    let height: CGFloat

    init(values: [Double], color: Color, height: CGFloat = 32) {
        self.values = values
        self.color = color
        self.height = height
    }

    var body: some View {
        if values.count < 2 {
            Rectangle()
                .fill(color.opacity(0.2))
                .frame(height: height)
                .clipShape(RoundedRectangle(cornerRadius: 4))
        } else {
            GeometryReader { geo in
                let minVal = values.min() ?? 0
                let maxVal = values.max() ?? 1
                let range = max(maxVal - minVal, 0.01)

                Path { path in
                    for (index, value) in values.enumerated() {
                        let x = geo.size.width * CGFloat(index) / CGFloat(values.count - 1)
                        let y = geo.size.height * (1 - CGFloat((value - minVal) / range))

                        if index == 0 {
                            path.move(to: CGPoint(x: x, y: y))
                        } else {
                            path.addLine(to: CGPoint(x: x, y: y))
                        }
                    }
                }
                .stroke(color, lineWidth: 1.5)
            }
            .frame(height: height)
        }
    }
}

#Preview {
    VStack(spacing: CrescendSpacing.space4) {
        SparklineView(values: [0.4, 0.5, 0.45, 0.6, 0.55, 0.7, 0.65], color: CrescendColor.dimDynamics)
        SparklineView(values: [0.3, 0.35, 0.5, 0.45, 0.6], color: CrescendColor.dimTiming)
        SparklineView(values: [], color: CrescendColor.dimPedaling)
    }
    .padding()
    .background(CrescendColor.background)
    .crescendTheme()
}
