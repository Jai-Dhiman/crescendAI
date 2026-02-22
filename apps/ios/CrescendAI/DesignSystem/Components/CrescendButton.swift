import SwiftUI

enum CrescendButtonStyle {
    case primary
    case secondary
    case ghost
}

struct CrescendButton: View {
    let title: String
    let style: CrescendButtonStyle
    let icon: String?
    let action: () -> Void

    init(
        _ title: String,
        style: CrescendButtonStyle = .primary,
        icon: String? = nil,
        action: @escaping () -> Void
    ) {
        self.title = title
        self.style = style
        self.icon = icon
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: CrescendSpacing.space2) {
                if let icon {
                    Image(systemName: icon)
                        .font(.system(size: 16, weight: .medium))
                }
                Text(title)
                    .font(CrescendFont.labelLG())
            }
            .frame(maxWidth: style == .primary ? .infinity : nil)
            .padding(.horizontal, horizontalPadding)
            .padding(.vertical, verticalPadding)
            .foregroundStyle(foregroundColor)
            .background(backgroundColor)
            .overlay(borderOverlay)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(CrescendPressStyle())
    }

    private var horizontalPadding: CGFloat {
        switch style {
        case .primary: CrescendSpacing.space6
        case .secondary: CrescendSpacing.space5
        case .ghost: CrescendSpacing.space4
        }
    }

    private var verticalPadding: CGFloat {
        switch style {
        case .primary: CrescendSpacing.space3
        case .secondary: CrescendSpacing.space3
        case .ghost: CrescendSpacing.space2
        }
    }

    private var foregroundColor: Color {
        switch style {
        case .primary: CrescendColor.background
        case .secondary: CrescendColor.foreground
        case .ghost: CrescendColor.foreground
        }
    }

    private var backgroundColor: Color {
        switch style {
        case .primary: CrescendColor.foreground
        case .secondary: .clear
        case .ghost: .clear
        }
    }

    @ViewBuilder
    private var borderOverlay: some View {
        switch style {
        case .primary:
            EmptyView()
        case .secondary:
            RoundedRectangle(cornerRadius: 8)
                .stroke(CrescendColor.border, lineWidth: 1)
        case .ghost:
            EmptyView()
        }
    }
}

struct CrescendIconButton: View {
    let icon: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 18, weight: .medium))
                .foregroundStyle(CrescendColor.foreground)
                .frame(width: 44, height: 44)
                .background(CrescendColor.subtleFill)
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(CrescendPressStyle())
    }
}

private struct CrescendPressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
            .opacity(configuration.isPressed ? 0.85 : 1.0)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

#Preview("Buttons") {
    VStack(spacing: CrescendSpacing.space4) {
        CrescendButton("Record Performance", style: .primary, icon: "mic.fill") {}
        CrescendButton("View Details", style: .secondary) {}
        CrescendButton("Learn More", style: .ghost) {}
        HStack(spacing: CrescendSpacing.space3) {
            CrescendIconButton(icon: "play.fill") {}
            CrescendIconButton(icon: "pause.fill") {}
            CrescendIconButton(icon: "stop.fill") {}
        }
    }
    .padding(CrescendSpacing.space6)
    .background(CrescendColor.background)
}
