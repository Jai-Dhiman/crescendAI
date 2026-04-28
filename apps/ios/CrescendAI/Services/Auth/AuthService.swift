import AuthenticationServices
import Foundation
import Sentry
import SwiftData

enum AuthError: LocalizedError {
    case appleSignInFailed(Error)
    case missingCredential
    case serverAuthFailed(String)
    case notAuthenticated

    var errorDescription: String? {
        switch self {
        case .appleSignInFailed(let error): "Apple sign-in failed: \(error.localizedDescription)"
        case .missingCredential: "Missing Apple credential data"
        case .serverAuthFailed(let message): "Authentication failed: \(message)"
        case .notAuthenticated: "Not signed in"
        }
    }
}

@MainActor
@Observable
final class AuthService {
    private(set) var isAuthenticated = false
    private(set) var appleUserId: String?

    private let session: URLSession
    private let appleUserIdKey = "crescendai.appleUserId"

    init(session: URLSession = .shared) {
        self.session = session
        loadStoredCredentials()
    }

    func signIn(identityToken: String, userId: String, email: String?) async throws {
        let url = APIEndpoints.signInSocial()
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        struct SocialSignInBody: Encodable {
            let provider: String
            let idToken: IdToken
            struct IdToken: Encodable {
                let token: String
            }
        }

        request.httpBody = try JSONEncoder().encode(
            SocialSignInBody(provider: "apple", idToken: .init(token: identityToken))
        )

        let (data, response) = try await session.data(for: request)

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

        self.appleUserId = userId
        self.isAuthenticated = true
        UserDefaults.standard.set(userId, forKey: appleUserIdKey)

        let sentryUser = Sentry.User()
        sentryUser.userId = userId
        SentrySDK.setUser(sentryUser)
    }

    func handleAuthorization(result: Result<ASAuthorization, Error>) async throws {
        let authorization: ASAuthorization
        switch result {
        case .success(let auth): authorization = auth
        case .failure(let error): throw AuthError.appleSignInFailed(error)
        }

        guard let credential = authorization.credential as? ASAuthorizationAppleIDCredential,
              let identityTokenData = credential.identityToken,
              let identityToken = String(data: identityTokenData, encoding: .utf8) else {
            throw AuthError.missingCredential
        }

        try await signIn(
            identityToken: identityToken,
            userId: credential.user,
            email: credential.email
        )
    }

    func signOut() {
        HTTPCookieStorage.shared.cookies?.forEach { cookie in
            if cookie.name == "better-auth.session_token" {
                HTTPCookieStorage.shared.deleteCookie(cookie)
            }
        }
        self.appleUserId = nil
        self.isAuthenticated = false
        UserDefaults.standard.removeObject(forKey: appleUserIdKey)
        SentrySDK.setUser(nil)
    }

    func ensureOrCreateStudent(in modelContext: ModelContext) throws -> Student {
        guard let appleUserId else { throw AuthError.notAuthenticated }

        let predicate = #Predicate<Student> { $0.appleUserId == appleUserId }
        let descriptor = FetchDescriptor<Student>(predicate: predicate)
        let existing = try modelContext.fetch(descriptor)

        if let student = existing.first { return student }

        let student = Student(appleUserId: appleUserId)
        modelContext.insert(student)
        try modelContext.save()
        return student
    }

    // For testing only
    func _setAuthenticatedForTesting(userId: String) {
        self.appleUserId = userId
        self.isAuthenticated = true
    }

    private func loadStoredCredentials() {
        let hasCookie = HTTPCookieStorage.shared.cookies?.contains {
            $0.name == "better-auth.session_token" && ($0.expiresDate ?? .distantFuture) > .now
        } ?? false

        if hasCookie {
            self.isAuthenticated = true
            self.appleUserId = UserDefaults.standard.string(forKey: appleUserIdKey)
        }
    }
}
