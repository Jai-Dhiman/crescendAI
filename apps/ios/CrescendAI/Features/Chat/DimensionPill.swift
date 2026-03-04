import SwiftUI

struct DimensionPill: View {
    let dimension: String

    var body: some View {
        Text(dimension.capitalized)
            .font(CrescendFont.labelSM())
            .foregroundStyle(CrescendColor.background)
            .padding(.horizontal, CrescendSpacing.space2)
            .padding(.vertical, 3)
            .background(CrescendColor.dimensionColor(for: dimension))
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

#Preview {
    HStack(spacing: CrescendSpacing.space2) {
        DimensionPill(dimension: "dynamics")
        DimensionPill(dimension: "timing")
        DimensionPill(dimension: "pedaling")
        DimensionPill(dimension: "articulation")
        DimensionPill(dimension: "phrasing")
        DimensionPill(dimension: "interpretation")
    }
    .padding()
    .background(CrescendColor.background)
    .crescendTheme()
}
