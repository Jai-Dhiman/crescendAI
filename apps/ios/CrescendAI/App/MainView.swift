import SwiftData
import SwiftUI

struct MainView: View {
    @Environment(AuthService.self) private var authService
    @Environment(\.modelContext) private var modelContext
    @State private var sidebarOpen = false

    var body: some View {
        NavigationStack {
            ZStack {
                // Main content
                ChatView()
                
                // Sidebar (slides under the floating button)
                SidebarView(
                    isOpen: $sidebarOpen,
                    onNewSession: {
                        // TODO: reset chat to new session
                    },
                    onSelectSession: { _ in
                        // TODO: load selected session
                    },
                    onShowProfile: {
                        // NavigationStack push handled via NavigationLink or path
                    }
                )
                
                // Floating menu button (hidden when sidebar is open)
                if !sidebarOpen {
                    VStack {
                        HStack {
                            Button(action: {
                                withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                                    sidebarOpen.toggle()
                                }
                            }) {
                                Image(systemName: "line.3.horizontal")
                                    .font(.system(size: 18, weight: .medium))
                                    .foregroundStyle(CrescendColor.foreground)
                                    .frame(width: 44, height: 44)
                                    .background(CrescendColor.surface.opacity(0.95))
                                    .clipShape(Circle())
                                    .shadow(color: .black.opacity(0.1), radius: 8, y: 2)
                            }
                            .buttonStyle(.plain)
                            .padding(.leading, CrescendSpacing.space4)
                            .padding(.top, CrescendSpacing.space2)

                            Spacer()
                        }

                        Spacer()
                    }
                    .transition(.opacity)
                }
            }
            .navigationDestination(for: String.self) { destination in
                switch destination {
                case "profile":
                    ProfileView()
                default:
                    EmptyView()
                }
            }
        }
    }
}

#Preview {
    MainView()
        .crescendTheme()
        .environment(AuthService())
        .modelContainer(
            for: [Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self, ConversationRecord.self],
            inMemory: true
        )
}
