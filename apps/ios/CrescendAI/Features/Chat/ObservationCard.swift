import SwiftUI

struct ObservationCard: View {
    let text: String
    let dimension: String
    let timestamp: Date
    let elaboration: String?
    let onTellMeMore: () -> Void

    var body: some View {
        CrescendCard {
            VStack(alignment: .leading, spacing: CrescendSpacing.space3) {
                // Header: dimension pill + timestamp
                HStack {
                    DimensionPill(dimension: dimension)
                    Spacer()
                    Text(timestamp, style: .time)
                        .font(CrescendFont.labelSM())
                        .foregroundStyle(CrescendColor.tertiaryText)
                }

                // Observation text
                Text(text)
                    .font(CrescendFont.bodyLG())
                    .foregroundStyle(CrescendColor.foreground)
                    .fixedSize(horizontal: false, vertical: true)

                // Elaboration (if expanded)
                if let elaboration {
                    Text(elaboration)
                        .font(CrescendFont.bodyMD())
                        .foregroundStyle(CrescendColor.secondaryText)
                        .padding(.leading, CrescendSpacing.space3)
                        .fixedSize(horizontal: false, vertical: true)
                        .transition(.opacity.combined(with: .move(edge: .top)))
                }

                // "Tell me more" link
                if elaboration == nil {
                    Button(action: onTellMeMore) {
                        HStack(spacing: CrescendSpacing.space1) {
                            Text("Tell me more")
                                .font(CrescendFont.bodySM())
                            Image(systemName: "arrow.right")
                                .font(.system(size: 11, weight: .medium))
                        }
                        .foregroundStyle(CrescendColor.accent)
                    }
                    .buttonStyle(CrescendPressStyle())
                }
            }
        }
    }
}

#Preview {
    VStack(spacing: CrescendSpacing.space4) {
        ObservationCard(
            text: "Your dynamics showed nice contrast between the forte and piano sections. The crescendo in the second phrase built naturally.",
            dimension: "dynamics",
            timestamp: .now,
            elaboration: nil,
            onTellMeMore: {}
        )

        ObservationCard(
            text: "The pedaling through the transition was clean -- you released right at the harmony change.",
            dimension: "pedaling",
            timestamp: .now,
            elaboration: "Specifically, the half-pedal technique at measure 12 kept the bass resonance without muddying the upper voices.",
            onTellMeMore: {}
        )
    }
    .padding()
    .background(CrescendColor.background)
    .crescendTheme()
}
