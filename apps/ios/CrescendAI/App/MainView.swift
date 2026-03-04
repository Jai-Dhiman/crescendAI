import SwiftData
import SwiftUI

struct MainView: View {
    @Environment(AuthService.self) private var authService
    @Environment(\.modelContext) private var modelContext
    @State private var showProfile = false
    @State private var showSessions = false

    var body: some View {
        HStack(spacing: 0) {
            SidebarView(
                onNewSession: { /* TODO: reset chat to new session */ },
                onShowSessions: { showSessions = true },
                onShowMetronome: { /* TODO: metronome sheet */ },
                onShowProfile: { showProfile = true }
            )

            ChatView()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(CrescendColor.background)
        .sheet(isPresented: $showProfile) {
            ProfileView()
                .crescendTheme()
        }
        .sheet(isPresented: $showSessions) {
            SessionsListView()
                .crescendTheme()
        }
    }
}

/// Placeholder for session history list
struct SessionsListView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ZStack {
                CrescendColor.background.ignoresSafeArea()

                VStack(spacing: CrescendSpacing.space4) {
                    Text("No sessions yet")
                        .font(CrescendFont.bodyMD())
                        .foregroundStyle(CrescendColor.secondaryText)
                }
            }
            .navigationTitle("Sessions")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                        .foregroundStyle(CrescendColor.foreground)
                }
            }
            .toolbarBackground(CrescendColor.surface, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
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
