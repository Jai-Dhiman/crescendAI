import XCTest
import WebKit
@testable import CrescendAI

@MainActor
final class ScoreWebViewIntegrationTests: XCTestCase {

    func test_loadCzernyScoreFiresRenderedEvent() async throws {
        // This test requires dist-scorehost/ to be in the app bundle.
        // It will be skipped if the bundle is not present (CI without a pre-build step).
        guard Bundle.main.url(forResource: "index", withExtension: "html", subdirectory: "dist-scorehost") != nil else {
            throw XCTSkip("dist-scorehost not in app bundle — run `just build-scorehost` first")
        }

        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []
        let schemeHandler = ScoreSchemeHandler()
        config.setURLSchemeHandler(schemeHandler, forURLScheme: "scorehost")

        let webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 375, height: 667), configuration: config)
        let bridge = ScoreHostBridge(webView: webView)
        config.userContentController.add(bridge, name: "scoreHostEvents")

        // WKWebView requires a visible window to execute JavaScript (timers, effects).
        let window = UIWindow(frame: CGRect(x: 0, y: 0, width: 375, height: 667))
        window.addSubview(webView)
        window.makeKeyAndVisible()

        var receivedEvents: [ScoreHostEvent] = []
        bridge.onEvent = { event in receivedEvents.append(event) }

        let request = URLRequest(url: URL(string: "scorehost://app/index.html")!)
        webView.load(request)

        // Wait for ready event (up to 15s for Verovio WASM init)
        let readyDeadline = Date().addingTimeInterval(15)
        while Date() < readyDeadline {
            if receivedEvents.contains(where: { if case .ready = $0 { return true }; return false }) { break }
            try await Task.sleep(for: .milliseconds(200))
        }

        // Load piece and show score_highlight
        try await bridge.load(pieceId: "czerny-op299-no1")
        let highlightConfig = ScoreHighlightConfig(
            pieceId: "czerny-op299-no1",
            highlights: [ScoreHighlight(bars: [1, 4], dimension: "dynamics", annotation: "forte")]
        )
        try await bridge.showArtifact(.scoreHighlight(highlightConfig))

        // Wait for rendered event
        let renderDeadline = Date().addingTimeInterval(15)
        var noteCount = 0
        while Date() < renderDeadline {
            for event in receivedEvents {
                if case .rendered(let n) = event { noteCount = n; break }
            }
            if noteCount > 0 { break }
            try await Task.sleep(for: .milliseconds(200))
        }

        XCTAssertGreaterThan(noteCount, 0, "Expected rendered event with noteCount > 0")
    }
}
