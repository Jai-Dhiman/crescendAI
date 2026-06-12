import Foundation

enum APIEndpoints {
    /// Production API host. Release builds always target this.
    private static let productionBaseURL = URL(string: "https://api.crescend.ai")!

    /// Local `wrangler dev` host (see justfile: API runs on 8787).
    private static let localBaseURL = URL(string: "http://localhost:8787")!

    #if DEBUG
    /// Debug builds target local `wrangler dev` by default so the app runs against
    /// the local stack. Override with the `CRESCEND_API_BASE_URL` env var (set it in
    /// the Xcode scheme) to point a debug build at production or any other host.
    static let baseURL: URL = {
        if let override = ProcessInfo.processInfo.environment["CRESCEND_API_BASE_URL"],
           let url = URL(string: override) {
            return url
        }
        return localBaseURL
    }()
    #else
    static let baseURL = productionBaseURL
    #endif

    static func signInSocial() -> URL {
        baseURL.appendingPathComponent("/api/auth/sign-in/social")
    }

    static func practiceStart() -> URL {
        baseURL.appendingPathComponent("/api/practice/start")
    }

    static func practiceChunk(sessionId: String, chunkIndex: Int) -> URL {
        var components = URLComponents(url: baseURL.appendingPathComponent("/api/practice/chunk"), resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "sessionId", value: sessionId),
            URLQueryItem(name: "chunkIndex", value: String(chunkIndex)),
        ]
        return components.url!
    }

    static func practiceWs(sessionId: String, conversationId: String) -> URL {
        var components = URLComponents(url: baseURL.appendingPathComponent("/api/practice/ws/\(sessionId)"), resolvingAgainstBaseURL: false)!
        let scheme = components.scheme == "https" ? "wss" : "ws"
        components.scheme = scheme
        components.queryItems = [URLQueryItem(name: "conversationId", value: conversationId)]
        return components.url!
    }

    static func chat() -> URL {
        baseURL.appendingPathComponent("/api/chat")
    }

    static func conversation(id: String) -> URL {
        baseURL.appendingPathComponent("/api/conversations/\(id)")
    }

    static func conversations() -> URL {
        baseURL.appendingPathComponent("/api/conversations")
    }

    static func sync() -> URL {
        baseURL.appendingPathComponent("/api/sync")
    }

    static func extractGoals() -> URL {
        baseURL.appendingPathComponent("/api/extract-goals")
    }
}
