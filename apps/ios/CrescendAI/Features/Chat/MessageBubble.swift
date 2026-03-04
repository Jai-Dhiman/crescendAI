import SwiftUI

struct MessageBubble: View {
    let text: String
    let isUser: Bool

    var body: some View {
        HStack {
            if isUser { Spacer(minLength: CrescendSpacing.space16) }

            Text(text)
                .font(CrescendFont.bodyMD())
                .foregroundStyle(CrescendColor.foreground)
                .padding(.horizontal, CrescendSpacing.space4)
                .padding(.vertical, CrescendSpacing.space3)
                .background(isUser ? CrescendColor.surface : .clear)
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .overlay {
                    if isUser {
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(CrescendColor.border, lineWidth: 1)
                    }
                }

            if !isUser { Spacer(minLength: CrescendSpacing.space16) }
        }
    }
}

#Preview {
    VStack(spacing: CrescendSpacing.space4) {
        MessageBubble(text: "I'm working on Chopin's Ballade No. 1", isUser: true)
        MessageBubble(text: "Let's focus on the opening section. Play through it and I'll listen.", isUser: false)
        MessageBubble(text: "How should I approach the tempo in the introduction?", isUser: true)
    }
    .padding()
    .background(CrescendColor.background)
    .crescendTheme()
}
