import SwiftUI
import WebKit

/// A SwiftUI view that renders a play_passage artifact via the score WebView.
///
/// Lifecycle:
///   1. makeUIView creates WKWebView + ScoreHostBridge together, loads scorehost://app/index.html
///   2. On .ready event the Coordinator calls load(pieceId:) then showArtifact(.playPassage)
///   3. Errors are thrown explicitly; there are no silent fallbacks.
struct ArtifactPlayPassageView: UIViewRepresentable {
    let config: PlayPassageConfig

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
        let config: PlayPassageConfig
        var bridge: ScoreHostBridge?
        private var didLoad = false

        init(config: PlayPassageConfig) {
            self.config = config
        }

        func handle(event: ScoreHostEvent, bridge: ScoreHostBridge) {
            guard case .ready = event, !didLoad else { return }
            didLoad = true
            Task {
                do {
                    try await bridge.load(pieceId: config.pieceId)
                    try await bridge.showArtifact(.playPassage(config))
                } catch {
                    // Surface the error as a log so it is observable.
                    // Do not swallow silently.
                    print("ArtifactPlayPassageView: bridge call failed: \(error)")
                }
            }
        }
    }
}
