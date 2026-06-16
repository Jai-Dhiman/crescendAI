import SwiftUI
import WebKit
import Sentry

/// A SwiftUI view that renders a score_highlight artifact via the score WebView (WKWebView + ScoreSchemeHandler + ScoreHostBridge).
///
/// Lifecycle:
///   1. makeUIView creates WKWebView + ScoreHostBridge together, loads scorehost://app/index.html
///   2. On .ready event the Coordinator calls load(pieceId:) then showArtifact(.scoreHighlight)
///   3. Errors are thrown explicitly; there are no silent fallbacks.
struct ArtifactScoreView: UIViewRepresentable {
    let config: ScoreHighlightConfig

    func makeCoordinator() -> Coordinator {
        Coordinator(config: config)
    }

    func makeUIView(context: Context) -> WKWebView {
        let webViewConfig = WKWebViewConfiguration()
        webViewConfig.allowsInlineMediaPlayback = true
        webViewConfig.mediaTypesRequiringUserActionForPlayback = []

        let schemeHandler = ScoreSchemeHandler()
        webViewConfig.setURLSchemeHandler(schemeHandler, forURLScheme: "scorehost")

        let webView = WKWebView(frame: .zero, configuration: webViewConfig)
        webView.scrollView.isScrollEnabled = false
        webView.backgroundColor = .white
        webView.isOpaque = false

        let bridge = ScoreHostBridge(webView: webView)
        bridge.onEvent = { [weak coordinator = context.coordinator] event in
            coordinator?.handle(event: event, bridge: bridge)
        }
        webViewConfig.userContentController.add(bridge, name: "scoreHostEvents")

        context.coordinator.bridge = bridge

        let url = URL(string: "scorehost://app/index.html")!
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        // Config changes handled by re-creating the view if pieceId changes.
    }

    static func dismantleUIView(_ uiView: WKWebView, coordinator: Coordinator) {
        uiView.configuration.userContentController.removeScriptMessageHandler(forName: "scoreHostEvents")
        coordinator.bridge = nil
    }

    @MainActor
    final class Coordinator {
        let config: ScoreHighlightConfig
        var bridge: ScoreHostBridge?
        private var didLoad = false

        init(config: ScoreHighlightConfig) {
            self.config = config
        }

        func handle(event: ScoreHostEvent, bridge: ScoreHostBridge) {
            guard case .ready = event, !didLoad else { return }
            didLoad = true
            Task {
                do {
                    try await bridge.load(pieceId: config.pieceId)
                    try await bridge.showArtifact(.scoreHighlight(config))
                } catch {
                    // Surface the error to Sentry and to the local log so it is observable.
                    // Do not swallow silently.
                    SentrySDK.capture(error: error)
                    print("ArtifactScoreView: bridge call failed: \(error)")
                }
            }
        }
    }
}
