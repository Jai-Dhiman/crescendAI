import SwiftUI

struct ScoreHighlightCard: View {
    let title: String
    let barRange: String
    let annotations: [String]

    var body: some View {
        CrescendCard {
            VStack(alignment: .leading, spacing: CrescendSpacing.space3) {
                HStack {
                    Image(systemName: "music.note.list")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(CrescendColor.secondaryText)
                    Text(title)
                        .font(CrescendFont.labelLG())
                        .foregroundStyle(CrescendColor.foreground)
                    Spacer()
                    Text(barRange)
                        .font(CrescendFont.labelSM())
                        .foregroundStyle(CrescendColor.tertiaryText)
                }

                // Placeholder notation area
                RoundedRectangle(cornerRadius: 8)
                    .fill(CrescendColor.surface2)
                    .frame(height: 80)
                    .overlay {
                        Text("Notation preview")
                            .font(CrescendFont.bodySM())
                            .foregroundStyle(CrescendColor.tertiaryText)
                    }

                if !annotations.isEmpty {
                    VStack(alignment: .leading, spacing: CrescendSpacing.space1) {
                        ForEach(annotations, id: \.self) { annotation in
                            HStack(spacing: CrescendSpacing.space2) {
                                Circle()
                                    .fill(CrescendColor.dimDynamics)
                                    .frame(width: 6, height: 6)
                                Text(annotation)
                                    .font(CrescendFont.bodySM())
                                    .foregroundStyle(CrescendColor.secondaryText)
                            }
                        }
                    }
                }
            }
        }
    }
}

#Preview {
    ScoreHighlightCard(
        title: "Crescendo Passage",
        barRange: "Bars 12-16",
        annotations: ["Build gradually from mp to f", "Arm weight increases through phrase"]
    )
    .padding()
    .background(CrescendColor.background)
    .crescendTheme()
}
