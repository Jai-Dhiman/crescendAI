import SwiftUI

struct ChatInputBar: View {
    @Binding var text: String
    let onSend: () -> Void
    let onMicTap: () -> Void

    var body: some View {
        HStack(spacing: CrescendSpacing.space2) {
            TextField("What are you practicing today?", text: $text, axis: .vertical)
                .font(CrescendFont.bodyMD())
                .foregroundStyle(CrescendColor.foreground)
                .lineLimit(1...4)
                .padding(.horizontal, CrescendSpacing.space3)
                .padding(.vertical, CrescendSpacing.space2)
                .background(CrescendColor.inputBackground)
                .clipShape(RoundedRectangle(cornerRadius: 20))
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(CrescendColor.border, lineWidth: 1)
                )
                .onSubmit { onSend() }

            if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Button(action: onMicTap) {
                    Image(systemName: "mic.fill")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundStyle(CrescendColor.foreground)
                        .frame(width: 36, height: 36)
                        .background(CrescendColor.surface2)
                        .clipShape(Circle())
                }
                .buttonStyle(CrescendPressStyle())
            } else {
                Button(action: onSend) {
                    Image(systemName: "arrow.up")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(CrescendColor.background)
                        .frame(width: 36, height: 36)
                        .background(CrescendColor.foreground)
                        .clipShape(Circle())
                }
                .buttonStyle(CrescendPressStyle())
            }
        }
        .padding(.horizontal, CrescendSpacing.space4)
        .padding(.vertical, CrescendSpacing.space2)
        .background(CrescendColor.background)
    }
}

#Preview {
    VStack {
        Spacer()
        ChatInputBar(text: .constant(""), onSend: {}, onMicTap: {})
        ChatInputBar(text: .constant("Working on Chopin"), onSend: {}, onMicTap: {})
    }
    .background(CrescendColor.background)
    .crescendTheme()
}
