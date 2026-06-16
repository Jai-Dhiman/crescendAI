import WebKit
import Foundation

class ScoreSchemeHandler: NSObject, WKURLSchemeHandler {
    private static let mimeTypes: [String: String] = [
        "html": "text/html; charset=utf-8",
        "js": "application/javascript",
        "css": "text/css",
        "wasm": "application/wasm",
        "json": "application/json",
        "mxl": "application/octet-stream",
        "mid": "audio/midi",
        "sf2": "application/octet-stream",
        "png": "image/png",
        "svg": "image/svg+xml",
        "ico": "image/x-icon",
        "map": "application/json",
    ]

    func webView(_ webView: WKWebView, start urlSchemeTask: WKURLSchemeTask) {
        guard let url = urlSchemeTask.request.url,
              url.scheme == "scorehost" else {
            urlSchemeTask.didFailWithError(
                NSError(domain: "ScoreSchemeHandler", code: 400,
                        userInfo: [NSLocalizedDescriptionKey: "Invalid scheme"])
            )
            return
        }

        // Map scorehost://app/<path> -> dist-scorehost/<path> in app bundle
        var relativePath = url.path
        if relativePath.hasPrefix("/") { relativePath = String(relativePath.dropFirst()) }
        if relativePath.isEmpty { relativePath = "index.html" }

        let bundlePath = Bundle.main.bundlePath
        let fullPath = "\(bundlePath)/dist-scorehost/\(relativePath)"
        guard FileManager.default.fileExists(atPath: fullPath) else {
            urlSchemeTask.didFailWithError(
                NSError(domain: "ScoreSchemeHandler", code: 404,
                        userInfo: [NSLocalizedDescriptionKey: "Not found: \(relativePath)"])
            )
            return
        }
        serveFile(at: URL(fileURLWithPath: fullPath), relativePath: relativePath, task: urlSchemeTask)
    }

    func webView(_ webView: WKWebView, stop urlSchemeTask: WKURLSchemeTask) {
        // No ongoing async work to cancel
    }

    private func serveFile(at fileURL: URL, relativePath: String, task: WKURLSchemeTask) {
        guard let data = try? Data(contentsOf: fileURL) else {
            task.didFailWithError(
                NSError(domain: "ScoreSchemeHandler", code: 500,
                        userInfo: [NSLocalizedDescriptionKey: "Failed to read: \(relativePath)"])
            )
            return
        }

        let ext = (relativePath as NSString).pathExtension.lowercased()
        let mimeType = ScoreSchemeHandler.mimeTypes[ext] ?? "application/octet-stream"

        let response = HTTPURLResponse(
            url: task.request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: [
                "Content-Type": mimeType,
                "Content-Length": "\(data.count)",
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*",
            ]
        )!
        task.didReceive(response)
        task.didReceive(data)
        task.didFinish()
    }
}
