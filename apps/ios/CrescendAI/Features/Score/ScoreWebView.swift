import SwiftUI
import WebKit

struct ScoreWebView: UIViewRepresentable {
    let bridge: ScoreHostBridge

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []

        let schemeHandler = ScoreSchemeHandler()
        config.setURLSchemeHandler(schemeHandler, forURLScheme: "scorehost")
        config.userContentController.add(bridge, name: "scoreHostEvents")

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.scrollView.isScrollEnabled = false
        webView.backgroundColor = .white
        webView.isOpaque = false

        let url = URL(string: "scorehost://app/index.html")!
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        // Updates handled via bridge calls
    }
}
