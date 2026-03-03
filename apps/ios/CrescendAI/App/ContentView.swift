import SwiftUI

struct ContentView: View {
    var body: some View {
        PracticeView()
    }
}

#Preview {
    ContentView()
        .crescendTheme()
        .modelContainer(for: [PracticeSessionRecord.self, ChunkResultRecord.self], inMemory: true)
}
