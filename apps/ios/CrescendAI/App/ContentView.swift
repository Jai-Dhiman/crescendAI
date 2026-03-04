import SwiftUI

struct ContentView: View {
    @State private var authService = AuthService()

    var body: some View {
        if authService.isAuthenticated {
            MainView()
                .environment(authService)
        } else {
            SignInView(authService: authService)
        }
    }
}

#Preview {
    ContentView()
        .crescendTheme()
        .modelContainer(
            for: [Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self],
            inMemory: true
        )
}
