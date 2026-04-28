import Foundation

enum APIError: LocalizedError {
    case invalidResponse(Int)
    case decodingFailed(Error)
    case networkError(Error)
    case serverError(String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse(let code): "Server returned status \(code)"
        case .decodingFailed(let error): "Failed to decode response: \(error.localizedDescription)"
        case .networkError(let error): "Network error: \(error.localizedDescription)"
        case .serverError(let message): message
        }
    }
}

actor APIClient {
    static let shared = APIClient()

    private let session: URLSession
    private let decoder: JSONDecoder

    init(session: URLSession = .shared) {
        self.session = session
        self.decoder = JSONDecoder()
    }

    func perform<T: Decodable>(_ request: URLRequest) async throws -> T {
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw APIError.networkError(error)
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse(0)
        }

        guard (200..<300).contains(httpResponse.statusCode) else {
            if let errorBody = try? JSONDecoder().decode([String: String].self, from: data),
               let message = errorBody["error"] {
                throw APIError.serverError(message)
            }
            throw APIError.invalidResponse(httpResponse.statusCode)
        }

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError.decodingFailed(error)
        }
    }
}
