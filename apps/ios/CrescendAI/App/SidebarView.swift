import SwiftUI

struct SidebarView: View {
    let onNewSession: () -> Void
    let onShowSessions: () -> Void
    let onShowMetronome: () -> Void
    let onShowProfile: () -> Void

    var body: some View {
        VStack(spacing: CrescendSpacing.space4) {
            sidebarButton(icon: "plus.message", action: onNewSession)

            sidebarButton(icon: "clock", action: onShowSessions)

            sidebarButton(icon: "metronome", action: onShowMetronome)

            Spacer()

            Button(action: onShowProfile) {
                Circle()
                    .fill(CrescendColor.surface2)
                    .frame(width: 32, height: 32)
                    .overlay {
                        Image(systemName: "person.fill")
                            .font(.system(size: 14, weight: .medium))
                            .foregroundStyle(CrescendColor.secondaryText)
                    }
            }
            .buttonStyle(CrescendPressStyle())
        }
        .padding(.vertical, CrescendSpacing.space4)
        .padding(.horizontal, CrescendSpacing.space3)
        .frame(width: 56)
        .frame(maxHeight: .infinity)
        .background(CrescendColor.sidebarBackground)
    }

    private func sidebarButton(icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 18, weight: .medium))
                .foregroundStyle(CrescendColor.secondaryText)
                .frame(width: 36, height: 36)
                .contentShape(Rectangle())
        }
        .buttonStyle(CrescendPressStyle())
    }
}

#Preview {
    HStack(spacing: 0) {
        SidebarView(
            onNewSession: {},
            onShowSessions: {},
            onShowMetronome: {},
            onShowProfile: {}
        )
        Spacer()
    }
    .background(CrescendColor.background)
    .crescendTheme()
}
