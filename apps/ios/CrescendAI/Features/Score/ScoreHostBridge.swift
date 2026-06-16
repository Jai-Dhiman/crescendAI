import WebKit
import Foundation

enum ScoreHostEventError: Error {
    case unknownType(String)
    case missingPayloadKey(String)
    case malformedBody
    case webViewDeallocated
}

enum ScoreHostEvent {
    case ready
    case error(reason: String)
    case rendered(noteCount: Int)
    case playback(state: String)
    case activeBar(bar: Int)
    case log(msg: String)

    init(from body: [String: Any]) throws {
        guard let type = body["type"] as? String,
              let payload = body["payload"] as? [String: Any] else {
            throw ScoreHostEventError.malformedBody
        }
        switch type {
        case "ready":
            self = .ready
        case "error":
            guard let reason = payload["reason"] as? String else {
                throw ScoreHostEventError.missingPayloadKey("reason")
            }
            self = .error(reason: reason)
        case "rendered":
            guard let noteCount = payload["noteCount"] as? Int else {
                throw ScoreHostEventError.missingPayloadKey("noteCount")
            }
            self = .rendered(noteCount: noteCount)
        case "playback":
            guard let state = payload["state"] as? String else {
                throw ScoreHostEventError.missingPayloadKey("state")
            }
            self = .playback(state: state)
        case "activeBar":
            guard let bar = payload["bar"] as? Int else {
                throw ScoreHostEventError.missingPayloadKey("bar")
            }
            self = .activeBar(bar: bar)
        case "log":
            let msg = payload["msg"] as? String ?? ""
            self = .log(msg: msg)
        default:
            throw ScoreHostEventError.unknownType(type)
        }
    }
}

@MainActor
class ScoreHostBridge: NSObject, WKScriptMessageHandler {
    private weak var webView: WKWebView?
    var onEvent: ((ScoreHostEvent) -> Void)?

    init(webView: WKWebView) {
        self.webView = webView
    }

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage
    ) {
        guard let body = message.body as? [String: Any] else { return }
        do {
            let event = try ScoreHostEvent(from: body)
            onEvent?(event)
        } catch {
            onEvent?(.log(msg: "ScoreHostBridge: unrecognized event: \(error)"))
        }
    }

    func ready() async throws {
        try await callVoid(js: "ScoreHost.ready()")
    }

    func load(pieceId: String) async throws {
        try await callVoid(js: "ScoreHost.load(pieceId)", args: ["pieceId": pieceId])
    }

    func showArtifact(_ config: ArtifactConfig) async throws {
        let json = try ScoreHostBridge.artifactJSON(for: config)
        try await callVoid(js: "ScoreHost.showArtifact(json)", args: ["json": json])
    }

    func play() async throws {
        try await callVoid(js: "ScoreHost.play()")
    }

    func stop() async throws {
        try await callVoid(js: "ScoreHost.stop()")
    }

    func setTempo(_ factor: Double) async throws {
        try await callVoid(js: "ScoreHost.setTempo(factor)", args: ["factor": factor])
    }

    nonisolated static func artifactJSON(for config: ArtifactConfig) throws -> String {
        let encoder = JSONEncoder()
        let data = try encoder.encode(config)
        guard let json = String(data: data, encoding: .utf8) else {
            throw ScoreHostEventError.malformedBody
        }
        return json
    }

    private func callVoid(js: String, args: [String: Any] = [:]) async throws {
        guard let webView else {
            throw ScoreHostEventError.webViewDeallocated
        }
        let wrapped = "Promise.resolve(\(js)).then(() => undefined)"
        let result = try await webView.callAsyncJavaScript(
            wrapped,
            arguments: args,
            in: nil,
            contentWorld: .page
        )
        _ = result
    }
}
