import Foundation

enum APIEndpoints {
    #if DEBUG
    static let baseURL = URL(string: "https://api.crescend.ai")!
    #else
    static let baseURL = URL(string: "https://api.crescend.ai")!
    #endif

    static func upload() -> URL {
        baseURL.appendingPathComponent("/api/upload")
    }

    static func analyze(performanceId: String) -> URL {
        baseURL.appendingPathComponent("/api/analyze/\(performanceId)")
    }

    static func performances() -> URL {
        baseURL.appendingPathComponent("/api/performances")
    }

    static func performance(id: String) -> URL {
        baseURL.appendingPathComponent("/api/performances/\(id)")
    }
}
