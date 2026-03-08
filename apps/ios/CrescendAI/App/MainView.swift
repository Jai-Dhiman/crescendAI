import SwiftData
import SwiftUI

struct MainView: View {
    @Environment(AuthService.self) private var authService
    @Environment(\.modelContext) private var modelContext
    @State private var sidebarOpen = false

    var body: some View {
        NavigationStack {
            ZStack {
                ChatView()

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
            }
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        sidebarOpen.toggle()
                    } label: {
                        Image(systemName: "line.3.horizontal")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(CrescendColor.foreground)
                    }
                }
            }
            .toolbarBackground(CrescendColor.background, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .navigationBarTitleDisplayMode(.inline)
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
            for: [Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self],
            inMemory: true
        )
}
