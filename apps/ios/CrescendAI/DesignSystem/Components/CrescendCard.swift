import SwiftUI

enum CrescendCardStyle {
    case base
    case interactive
    case elevated
}

struct CrescendCard<Content: View>: View {
    let style: CrescendCardStyle
    let action: (() -> Void)?
    @ViewBuilder let content: () -> Content

    init(
        style: CrescendCardStyle = .base,
        action: (() -> Void)? = nil,
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.style = style
        self.action = action
        self.content = content
    }

    var body: some View {
        Group {
            if let action, style == .interactive {
                Button(action: action) {
                    cardContent
                }
                .buttonStyle(CardPressStyle())
            } else {
                cardContent
            }
        }
    }

    private var cardContent: some View {
        content()
            .padding(CrescendSpacing.space4)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(CrescendColor.background)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(borderColor, lineWidth: style == .elevated ? 0 : 1)
            )
            .shadow(color: shadowColor, radius: shadowRadius, y: shadowY)
    }

    private var borderColor: Color {
        switch style {
        case .base, .interactive: CrescendColor.border
        case .elevated: .clear
        }
    }

    private var shadowColor: Color {
        switch style {
        case .base: CrescendColor.foreground.opacity(0.04)
        case .interactive: CrescendColor.foreground.opacity(0.04)
        case .elevated: CrescendColor.foreground.opacity(0.08)
        }
    }

    private var shadowRadius: CGFloat {
        switch style {
        case .base: 2
        case .interactive: 2
        case .elevated: 12
        }
    }

    private var shadowY: CGFloat {
        switch style {
        case .base: 1
        case .interactive: 1
        case .elevated: 4
        }
    }
}

private struct CardPressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .shadow(
                color: CrescendColor.foreground.opacity(configuration.isPressed ? 0.02 : 0.04),
                radius: configuration.isPressed ? 1 : 4,
                y: configuration.isPressed ? 0 : 2
            )
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

#Preview("Cards") {
    VStack(spacing: CrescendSpacing.space4) {
        CrescendCard {
            VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
                Text("Base Card")
                    .font(CrescendFont.headingMD())
                    .foregroundStyle(CrescendColor.foreground)
                Text("A simple container with subtle border.")
                    .font(CrescendFont.bodySM())
                    .foregroundStyle(CrescendColor.secondaryText)
            }
        }

        CrescendCard(style: .interactive, action: {}) {
            VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
                Text("Interactive Card")
                    .font(CrescendFont.headingMD())
                    .foregroundStyle(CrescendColor.foreground)
                Text("Press me for feedback.")
                    .font(CrescendFont.bodySM())
                    .foregroundStyle(CrescendColor.secondaryText)
            }
        }

        CrescendCard(style: .elevated) {
            VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
                Text("Elevated Card")
                    .font(CrescendFont.headingMD())
                    .foregroundStyle(CrescendColor.foreground)
                Text("Floating with a drop shadow.")
                    .font(CrescendFont.bodySM())
                    .foregroundStyle(CrescendColor.secondaryText)
            }
        }
    }
    .padding(CrescendSpacing.space6)
    .background(CrescendColor.background)
}
