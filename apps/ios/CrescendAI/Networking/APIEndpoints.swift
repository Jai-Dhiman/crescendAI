import Foundation

enum APIEndpoints {
    #if DEBUG
    static let baseURL = URL(string: "https://api.crescend.ai")!
    #else
    static let baseURL = URL(string: "https://api.crescend.ai")!
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
