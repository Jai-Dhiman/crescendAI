import SwiftUI

struct PracticeControlBar: View {
    let onPause: () -> Void
    let onStop: () -> Void
    let onAsk: () -> Void

    var body: some View {
        HStack(spacing: CrescendSpacing.space6) {
            // Pause button
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

            // Stop button
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

            // "How was that?" button
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
        .padding(.horizontal, CrescendSpacing.space4)
        .padding(.vertical, CrescendSpacing.space3)
        .background(CrescendColor.background.opacity(0.95))
    }
}

#Preview {
    VStack {
        Spacer()
        PracticeControlBar(onPause: {}, onStop: {}, onAsk: {})
    }
    .background(CrescendColor.background)
    .crescendTheme()
}
