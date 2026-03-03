import AuthenticationServices
import Foundation
import SwiftData

enum AuthError: LocalizedError {
    case appleSignInFailed(Error)
    case missingCredential
    case serverAuthFailed(String)
    case notAuthenticated

    var errorDescription: String? {
        switch self {
        case .appleSignInFailed(let error):
            "Apple sign-in failed: \(error.localizedDescription)"
        case .missingCredential:
            "Missing Apple credential data"
        case .serverAuthFailed(let message):
            "Authentication failed: \(message)"
        case .notAuthenticated:
            "Not signed in"
        }
    }
}

struct AuthResponse: Codable {
    let jwt: String
    let apple_user_id: String
    let email: String?
    let is_new_user: Bool
}

@MainActor
@Observable
final class AuthService {
    private(set) var isAuthenticated = false
    private(set) var appleUserId: String?
    private(set) var jwt: String?

    init() {
        loadStoredCredentials()
    }

    var authorizationHeader: String? {
        guard let jwt else { return nil }
        return "Bearer \(jwt)"
    }

    // MARK: - Sign In with Apple

    func handleAuthorization(result: Result<ASAuthorization, Error>) async throws {
        let authorization: ASAuthorization
        switch result {
        case .success(let auth):
            authorization = auth
        case .failure(let error):
            throw AuthError.appleSignInFailed(error)
        }

        guard let credential = authorization.credential as? ASAuthorizationAppleIDCredential,
              let identityTokenData = credential.identityToken,
              let identityToken = String(data: identityTokenData, encoding: .utf8) else {
            throw AuthError.missingCredential
        }

        let email = credential.email
        let userId = credential.user

        let response = try await sendTokenToServer(
            identityToken: identityToken,
            userId: userId,
            email: email
        )

        try KeychainService.save(response.jwt, for: .sessionJWT)
        try KeychainService.save(response.apple_user_id, for: .appleUserId)

        self.jwt = response.jwt
        self.appleUserId = response.apple_user_id
        self.isAuthenticated = true
    }

    func signOut() throws {
        try KeychainService.deleteAll()
        self.jwt = nil
        self.appleUserId = nil
        self.isAuthenticated = false
    }

    func ensureOrCreateStudent(in modelContext: ModelContext) throws -> Student {
        guard let appleUserId else {
            throw AuthError.notAuthenticated
        }

        let predicate = #Predicate<Student> { $0.appleUserId == appleUserId }
        let descriptor = FetchDescriptor<Student>(predicate: predicate)
        let existing = try modelContext.fetch(descriptor)

        if let student = existing.first {
            return student
        }

        let student = Student(appleUserId: appleUserId)
        modelContext.insert(student)
        try modelContext.save()
        return student
    }

    // MARK: - Private

    private func loadStoredCredentials() {
        do {
            if let storedJWT = try KeychainService.read(.sessionJWT),
               let storedUserId = try KeychainService.read(.appleUserId) {
                self.jwt = storedJWT
                self.appleUserId = storedUserId
                self.isAuthenticated = true
            }
        } catch {
            print("[AuthService] Failed to load stored credentials: \(error)")
        }
    }

    private func sendTokenToServer(
        identityToken: String,
        userId: String,
        email: String?
    ) async throws -> AuthResponse {
        let url = APIEndpoints.authApple()

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        struct AuthRequest: Encodable {
            let identity_token: String
            let user_id: String
            let email: String?
        }

        let body = AuthRequest(
            identity_token: identityToken,
            user_id: userId,
            email: email
        )
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw AuthError.serverAuthFailed("Invalid response")
        }

        guard (200..<300).contains(httpResponse.statusCode) else {
            if let errorBody = try? JSONDecoder().decode([String: String].self, from: data),
               let message = errorBody["error"] {
                throw AuthError.serverAuthFailed(message)
            }
            throw AuthError.serverAuthFailed("Server returned \(httpResponse.statusCode)")
        }

        return try JSONDecoder().decode(AuthResponse.self, from: data)
    }
}
