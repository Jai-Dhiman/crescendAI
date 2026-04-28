import CoreText
import Sentry
import SwiftData
import SwiftUI

@main
struct CrescendAIApp: App {
    let modelContainer: ModelContainer

    init() {
        SentrySDK.start { options in
            options.dsn = "https://d7fc0b7e8d2663a2bad41faa86dbfa1b@o4511017227321344.ingest.us.sentry.io/4511017237217280"
            #if DEBUG
            options.environment = "development"
            #else
            options.environment = "production"
            #endif
            options.tracesSampleRate = 0.1
            options.enableCrashHandler = true
            options.enableAutoSessionTracking = true
            options.attachScreenshot = true
            options.enableMetricKit = true
        }

        do {
            let schema = Schema([
                Student.self,
                PracticeSessionRecord.self,
                ChunkResultRecord.self,
                ObservationRecord.self,
                CheckInRecord.self,
                ConversationRecord.self,
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
