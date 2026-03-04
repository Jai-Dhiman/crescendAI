import SwiftUI

struct KeyboardGuideCard: View {
    let title: String
    let highlightedKeys: [Int]
    let keyLabels: [Int: String]

    private let whiteKeyCount = 14
    private let whiteKeyWidth: CGFloat = 22
    private let blackKeyWidth: CGFloat = 14
    private let whiteKeyHeight: CGFloat = 80
    private let blackKeyHeight: CGFloat = 50

    // Pattern of black keys relative to white key index (C major octave)
    private let blackKeyIndices: Set<Int> = [1, 2, 4, 5, 6]

    var body: some View {
        CrescendCard {
            VStack(alignment: .leading, spacing: CrescendSpacing.space3) {
                HStack {
                    Image(systemName: "pianokeys")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(CrescendColor.secondaryText)
                    Text(title)
                        .font(CrescendFont.labelLG())
                        .foregroundStyle(CrescendColor.foreground)
                }

                keyboardView
                    .frame(height: whiteKeyHeight)

                if !keyLabels.isEmpty {
                    HStack(spacing: CrescendSpacing.space2) {
                        ForEach(Array(keyLabels.sorted(by: { $0.key < $1.key })), id: \.key) { key, label in
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(CrescendColor.dimArticulation)
                                    .frame(width: 8, height: 8)
                                Text(label)
                                    .font(CrescendFont.labelSM())
                                    .foregroundStyle(CrescendColor.secondaryText)
                            }
                        }
                    }
                }
            }
        }
    }

    private var keyboardView: some View {
        GeometryReader { geo in
            let availableWidth = geo.size.width
            let keyWidth = availableWidth / CGFloat(whiteKeyCount)

            ZStack(alignment: .topLeading) {
                // White keys
                HStack(spacing: 0) {
                    ForEach(0..<whiteKeyCount, id: \.self) { index in
                        let isHighlighted = highlightedKeys.contains(index)
                        RoundedRectangle(cornerRadius: 2)
                            .fill(isHighlighted ? CrescendColor.dimArticulation.opacity(0.6) : CrescendColor.foreground.opacity(0.9))
                            .frame(width: keyWidth - 1, height: whiteKeyHeight)
                            .padding(.trailing, 1)
                    }
                }

                // Black keys
                ForEach(0..<whiteKeyCount, id: \.self) { index in
                    let octavePosition = index % 7
                    if blackKeyIndices.contains(octavePosition) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(CrescendColor.background)
                            .frame(width: keyWidth * 0.6, height: blackKeyHeight)
                            .offset(x: CGFloat(index) * keyWidth + keyWidth * 0.7)
                    }
                }
            }
        }
    }
}

#Preview {
    KeyboardGuideCard(
        title: "Hand Position",
        highlightedKeys: [0, 2, 4, 5, 7],
        keyLabels: [0: "C4", 4: "E4", 7: "G4"]
    )
    .padding()
    .background(CrescendColor.background)
    .crescendTheme()
}
