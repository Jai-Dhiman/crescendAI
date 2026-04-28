import SwiftData
import SwiftUI

struct SidebarView: View {
    @Binding var isOpen: Bool
    let onNewSession: () -> Void
    let onSelectSession: (String) -> Void
    let onShowProfile: () -> Void

    @Query(sort: \ConversationRecord.startedAt, order: .reverse) private var sessions: [ConversationRecord]

    var body: some View {
        ZStack(alignment: .leading) {
            // Dimmed background overlay
            if isOpen {
                Color.black.opacity(0.4)
                    .ignoresSafeArea()
                    .onTapGesture { closeSidebar() }
                    .transition(.opacity)
            }

            // Drawer panel
            if isOpen {
                drawerContent
                    .frame(width: 280)
                    .frame(maxHeight: .infinity)
                    .background(CrescendColor.sidebarBackground)
                    .ignoresSafeArea(edges: .bottom)
                    .transition(.move(edge: .leading))
            }
        }
        .animation(.easeInOut(duration: 0.25), value: isOpen)
        .gesture(
            DragGesture()
                .onEnded { value in
                    if value.translation.width < -50 {
                        closeSidebar()
                    }
                }
        )
    }

    private var drawerContent: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Text("crescend")
                    .font(CrescendFont.displayMD())
                    .foregroundStyle(CrescendColor.foreground)

                Spacer()

                Button(action: onNewSession) {
                    Image(systemName: "square.and.pencil")
                        .font(.system(size: 18, weight: .medium))
                        .foregroundStyle(CrescendColor.foreground)
                        .frame(width: 36, height: 36)
                        .contentShape(Rectangle())
                }
                .buttonStyle(CrescendPressStyle())
            }
            .padding(.horizontal, CrescendSpacing.space4)
            .padding(.top, CrescendSpacing.space2)
            .padding(.bottom, CrescendSpacing.space6)

            // Session list
            ScrollView {
                LazyVStack(alignment: .leading, spacing: CrescendSpacing.space1) {
                    Text("Recents")
                        .font(CrescendFont.labelMD())
                        .foregroundStyle(CrescendColor.tertiaryText)
                        .padding(.horizontal, CrescendSpacing.space4)
                        .padding(.bottom, CrescendSpacing.space2)

                    ForEach(sessions) { session in
                        Button {
                            onSelectSession(session.conversationId)
                            closeSidebar()
                        } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(session.displayTitle)
                                    .font(CrescendFont.bodyMD())
                                    .foregroundStyle(CrescendColor.foreground)
                                    .lineLimit(1)

                                Text(session.startedAt, style: .relative)
                                    .font(CrescendFont.labelSM())
                                    .foregroundStyle(CrescendColor.tertiaryText)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, CrescendSpacing.space4)
                            .padding(.vertical, CrescendSpacing.space2)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(CrescendPressStyle())
                    }
                }
            }

            Spacer()

            // Profile footer
            Button(action: {
                onShowProfile()
                closeSidebar()
            }) {
                HStack(spacing: CrescendSpacing.space3) {
                    Circle()
                        .fill(CrescendColor.surface2)
                        .frame(width: 32, height: 32)
                        .overlay {
                            Image(systemName: "person.fill")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundStyle(CrescendColor.secondaryText)
                        }

                    Text("Profile")
                        .font(CrescendFont.bodyMD())
                        .foregroundStyle(CrescendColor.foreground)

                    Spacer()
                }
                .padding(.horizontal, CrescendSpacing.space4)
                .padding(.vertical, CrescendSpacing.space3)
                .contentShape(Rectangle())
            }
            .buttonStyle(CrescendPressStyle())
            .padding(.bottom, CrescendSpacing.space4)

            Divider()
                .overlay(CrescendColor.border)
                .padding(.horizontal, CrescendSpacing.space4)
        }
    }

    private func closeSidebar() {
        isOpen = false
    }
}

#Preview {
    ZStack {
        CrescendColor.background.ignoresSafeArea()
        SidebarView(
            isOpen: .constant(true),
            onNewSession: {},
            onSelectSession: { _ in },
            onShowProfile: {}
        )
    }
    .crescendTheme()
    .modelContainer(for: [ConversationRecord.self], inMemory: true)
}
