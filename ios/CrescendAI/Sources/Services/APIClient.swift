import Foundation

// MARK: - API Configuration

struct APIConfig {
    static let baseURL = ProcessInfo.processInfo.environment["API_URL"] ?? "https://api.pianoanalyzer.com"
    static let version = "v1"
    static var fullURL: String {
        return "\(baseURL)/\(version)"
    }
}

// MARK: - API Client Protocol

protocol APIClientProtocol {
    // Auth endpoints
    func authenticateWithGoogle(token: String) async throws -> APIResponse<AuthResponse>
    func refreshToken(refreshToken: String) async throws -> APIResponse<AuthTokens>
    func signOut() async throws -> APIResponse<EmptyResponse>
    
    // User endpoints
    func getUserProfile(userId: String) async throws -> APIResponse<User>
    func updateUserProfile(userId: String, updates: [String: Any]) async throws -> APIResponse<User>
    
    // Recordings endpoints
    func getUserRecordings(userId: String, page: Int, limit: Int) async throws -> PaginatedResponse<Recording>
    func getRecording(recordingId: String) async throws -> APIResponse<Recording>
    func createRecording(recordingData: CreateRecordingRequest) async throws -> APIResponse<Recording>
    func updateRecording(recordingId: String, updates: [String: Any]) async throws -> APIResponse<Recording>
    func deleteRecording(recordingId: String) async throws -> APIResponse<EmptyResponse>
    func uploadRecording(recordingId: String, audioData: Data) async throws -> APIResponse<UploadResponse>
    
    // Analysis endpoints
    func getAnalysis(recordingId: String) async throws -> APIResponse<Analysis>
    func requestAnalysis(recordingId: String) async throws -> APIResponse<Analysis>
    
    // Progress endpoints
    func getUserProgress(userId: String) async throws -> APIResponse<Progress>
    
    // Practice Sessions endpoints
    func getUserPracticeSessions(userId: String, page: Int, limit: Int) async throws -> PaginatedResponse<PracticeSession>
}

// MARK: - API Client Implementation

class APIClient: APIClientProtocol {
    
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder
    
    // Auth token provider closure
    private var getAccessToken: (() -> String?)?
    
    init(session: URLSession = .shared, getAccessToken: (() -> String?)? = nil) {
        self.session = session
        self.getAccessToken = getAccessToken
        
        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
        
        // Configure date formatting to match API
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'"
        decoder.dateDecodingStrategy = .formatted(dateFormatter)
        encoder.dateEncodingStrategy = .formatted(dateFormatter)
    }
    
    // MARK: - Private Helper Methods
    
    private func makeRequest<T: Codable>(
        endpoint: String,
        method: HTTPMethod = .GET,
        body: Data? = nil,
        responseType: T.Type
    ) async throws -> T {
        guard let url = URL(string: "\(APIConfig.fullURL)\(endpoint)") else {
            throw AppError(code: "INVALID_URL", message: "Invalid URL for endpoint: \(endpoint)")
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = method.rawValue
        
        // Set common headers
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Add authorization header if token is available
        if let accessToken = getAccessToken?() {
            request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        }
        
        // Set request body if provided
        if let body = body {
            request.httpBody = body
        }
        
        do {
            let (data, response) = try await session.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw AppError(code: "INVALID_RESPONSE", message: "Invalid HTTP response")
            }
            
            // Handle HTTP error status codes
            if !(200...299).contains(httpResponse.statusCode) {
                let errorMessage: String
                
                if let errorData = try? decoder.decode(APIErrorResponse.self, from: data) {
                    errorMessage = errorData.message ?? "Unknown error"
                } else {
                    errorMessage = "HTTP \(httpResponse.statusCode): \(HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode))"
                }
                
                throw AppError(code: "HTTP_\(httpResponse.statusCode)", message: errorMessage)
            }
            
            // Decode response
            do {
                return try decoder.decode(responseType, from: data)
            } catch {
                throw AppError(code: "DECODE_ERROR", message: "Failed to decode response: \(error.localizedDescription)")
            }
            
        } catch {
            if error is AppError {
                throw error
            } else {
                throw AppError(code: "NETWORK_ERROR", message: "Network request failed: \(error.localizedDescription)")
            }
        }
    }
    
    private func buildQueryString(parameters: [String: Any]) -> String {
        let queryItems = parameters.compactMap { key, value -> URLQueryItem? in
            return URLQueryItem(name: key, value: "\(value)")
        }
        
        var components = URLComponents()
        components.queryItems = queryItems
        
        return components.query ?? ""
    }
    
    // MARK: - Auth Endpoints
    
    func authenticateWithGoogle(token: String) async throws -> APIResponse<AuthResponse> {
        let requestBody = ["token": token]
        let bodyData = try encoder.encode(requestBody)
        
        return try await makeRequest(
            endpoint: "/auth/google",
            method: .POST,
            body: bodyData,
            responseType: APIResponse<AuthResponse>.self
        )
    }
    
    func refreshToken(refreshToken: String) async throws -> APIResponse<AuthTokens> {
        let requestBody = ["refreshToken": refreshToken]
        let bodyData = try encoder.encode(requestBody)
        
        return try await makeRequest(
            endpoint: "/auth/refresh",
            method: .POST,
            body: bodyData,
            responseType: APIResponse<AuthTokens>.self
        )
    }
    
    func signOut() async throws -> APIResponse<EmptyResponse> {
        return try await makeRequest(
            endpoint: "/auth/signout",
            method: .POST,
            responseType: APIResponse<EmptyResponse>.self
        )
    }
    
    // MARK: - User Endpoints
    
    func getUserProfile(userId: String) async throws -> APIResponse<User> {
        return try await makeRequest(
            endpoint: "/users/\(userId)",
            responseType: APIResponse<User>.self
        )
    }
    
    func updateUserProfile(userId: String, updates: [String: Any]) async throws -> APIResponse<User> {
        let bodyData = try JSONSerialization.data(withJSONObject: updates)
        
        return try await makeRequest(
            endpoint: "/users/\(userId)",
            method: .PATCH,
            body: bodyData,
            responseType: APIResponse<User>.self
        )
    }
    
    // MARK: - Recording Endpoints
    
    func getUserRecordings(userId: String, page: Int = 1, limit: Int = 20) async throws -> PaginatedResponse<Recording> {
        let queryString = buildQueryString(parameters: ["page": page, "limit": limit])
        let endpoint = "/users/\(userId)/recordings?\(queryString)"
        
        return try await makeRequest(
            endpoint: endpoint,
            responseType: PaginatedResponse<Recording>.self
        )
    }
    
    func getRecording(recordingId: String) async throws -> APIResponse<Recording> {
        return try await makeRequest(
            endpoint: "/recordings/\(recordingId)",
            responseType: APIResponse<Recording>.self
        )
    }
    
    func createRecording(recordingData: CreateRecordingRequest) async throws -> APIResponse<Recording> {
        let bodyData = try encoder.encode(recordingData)
        
        return try await makeRequest(
            endpoint: "/recordings",
            method: .POST,
            body: bodyData,
            responseType: APIResponse<Recording>.self
        )
    }
    
    func updateRecording(recordingId: String, updates: [String: Any]) async throws -> APIResponse<Recording> {
        let bodyData = try JSONSerialization.data(withJSONObject: updates)
        
        return try await makeRequest(
            endpoint: "/recordings/\(recordingId)",
            method: .PATCH,
            body: bodyData,
            responseType: APIResponse<Recording>.self
        )
    }
    
    func deleteRecording(recordingId: String) async throws -> APIResponse<EmptyResponse> {
        return try await makeRequest(
            endpoint: "/recordings/\(recordingId)",
            method: .DELETE,
            responseType: APIResponse<EmptyResponse>.self
        )
    }
    
    func uploadRecording(recordingId: String, audioData: Data) async throws -> APIResponse<UploadResponse> {
        // For multipart/form-data upload
        let boundary = "Boundary-\(UUID().uuidString)"
        var body = Data()
        
        // Add form data
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"audio\"; filename=\"recording.m4a\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/m4a\r\n\r\n".data(using: .utf8)!)
        body.append(audioData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        guard let url = URL(string: "\(APIConfig.fullURL)/recordings/\(recordingId)/upload") else {
            throw AppError(code: "INVALID_URL", message: "Invalid upload URL")
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        if let accessToken = getAccessToken?() {
            request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        }
        
        request.httpBody = body
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw AppError(code: "UPLOAD_ERROR", message: "Failed to upload recording")
        }
        
        return try decoder.decode(APIResponse<UploadResponse>.self, from: data)
    }
    
    // MARK: - Analysis Endpoints
    
    func getAnalysis(recordingId: String) async throws -> APIResponse<Analysis> {
        return try await makeRequest(
            endpoint: "/recordings/\(recordingId)/analysis",
            responseType: APIResponse<Analysis>.self
        )
    }
    
    func requestAnalysis(recordingId: String) async throws -> APIResponse<Analysis> {
        return try await makeRequest(
            endpoint: "/recordings/\(recordingId)/analyze",
            method: .POST,
            responseType: APIResponse<Analysis>.self
        )
    }
    
    // MARK: - Progress Endpoints
    
    func getUserProgress(userId: String) async throws -> APIResponse<Progress> {
        return try await makeRequest(
            endpoint: "/users/\(userId)/progress",
            responseType: APIResponse<Progress>.self
        )
    }
    
    // MARK: - Practice Session Endpoints
    
    func getUserPracticeSessions(userId: String, page: Int = 1, limit: Int = 20) async throws -> PaginatedResponse<PracticeSession> {
        let queryString = buildQueryString(parameters: ["page": page, "limit": limit])
        let endpoint = "/users/\(userId)/practice-sessions?\(queryString)"
        
        return try await makeRequest(
            endpoint: endpoint,
            responseType: PaginatedResponse<PracticeSession>.self
        )
    }
}

// MARK: - Supporting Types

enum HTTPMethod: String {
    case GET = "GET"
    case POST = "POST"
    case PUT = "PUT"
    case PATCH = "PATCH"
    case DELETE = "DELETE"
}

struct AuthResponse: Codable {
    let user: User
    let tokens: AuthTokens
}

struct CreateRecordingRequest: Codable {
    let userId: String
    let title: String
    let description: String?
    let duration: Double
    let status: RecordingStatus
}

struct UploadResponse: Codable {
    let uploadUrl: String
}

struct EmptyResponse: Codable {}

struct APIErrorResponse: Codable {
    let message: String?
    let error: String?
}
