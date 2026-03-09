import CoreText
import SwiftData
import SwiftUI

@main
struct CrescendAIApp: App {
    let modelContainer: ModelContainer

    init() {
        do {
            let schema = Schema([
                Student.self,
                PracticeSessionRecord.self,
                ChunkResultRecord.self,
                ObservationRecord.self,
                CheckInRecord.self,
            ])
            let config = ModelConfiguration(schema: schema)
            modelContainer = try ModelContainer(for: schema, configurations: [config])
        } catch {
            fatalError("Failed to create ModelContainer: \(error)")
        }
        Self.registerFonts()
    }

    static func registerFonts() {
        guard let fontURL = Bundle.main.url(forResource: "Lora-VariableFont_wght", withExtension: "ttf") else {
            return
        }
        var error: Unmanaged<CFError>?
        CTFontManagerRegisterFontsForURL(fontURL as CFURL, .process, &error)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .crescendTheme()
                .modelContainer(modelContainer)
        }
    }
}
