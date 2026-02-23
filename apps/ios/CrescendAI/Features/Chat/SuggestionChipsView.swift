import SwiftUI

struct SuggestionChipsView: View {
    let chips: [SuggestionChip]
    let onTap: (SuggestionChip) -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: CrescendSpacing.space2) {
                ForEach(chips) { chip in
                    Button(action: { onTap(chip) }) {
                        Text(chip.label)
                            .font(CrescendFont.labelLG())
                            .foregroundStyle(CrescendColor.foreground)
                            .padding(.horizontal, CrescendSpacing.space4)
                            .padding(.vertical, CrescendSpacing.space2)
                            .background(CrescendColor.subtleFill)
                            .clipShape(Capsule())
                            .overlay(
                                Capsule()
                                    .stroke(CrescendColor.border, lineWidth: 1)
                            )
                    }
                    .buttonStyle(ChipPressStyle())
                }
            }
            .padding(.horizontal, CrescendSpacing.space4)
        }
    }
}

private struct ChipPressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            .opacity(configuration.isPressed ? 0.7 : 1.0)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

#Preview {
    VStack {
        Spacer()
        SuggestionChipsView(
            chips: [
                SuggestionChip(label: "Focus on dynamics", actionKey: "focus"),
                SuggestionChip(label: "Play it again", actionKey: "play"),
                SuggestionChip(label: "How does this work?", actionKey: "help"),
            ],
            onTap: { _ in }
        )
    }
    .background(CrescendColor.background)
}
