import SwiftData
import SwiftUI

@main
struct CrescendAIApp: App {
    let modelContainer: ModelContainer

    init() {
        do {
            let schema = Schema([PracticeSessionRecord.self, ChunkResultRecord.self])
            let config = ModelConfiguration(schema: schema)
            modelContainer = try ModelContainer(for: schema, configurations: [config])
        } catch {
            fatalError("Failed to create ModelContainer: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .crescendTheme()
                .modelContainer(modelContainer)
        }
    }
}
