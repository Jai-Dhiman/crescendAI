import SwiftData
import Testing
@testable import CrescendAI

@MainActor
struct AuthServiceTests {
    // MARK: - Helpers

    private func clearKeychain() {
        try? KeychainService.deleteAll()
    }

    private func seedKeychain(jwt: String, appleUserId: String) throws {
        try KeychainService.save(jwt, for: .sessionJWT)
        try KeychainService.save(appleUserId, for: .appleUserId)
    }

    // MARK: - loadStoredCredentials (via init)

    @Test func initWithStoredCredentialsSetsAuthenticated() throws {
        clearKeychain()
        try seedKeychain(jwt: "test.jwt.token", appleUserId: "apple.user.123")
        defer { clearKeychain() }

        let service = AuthService()

        #expect(service.isAuthenticated == true)
        #expect(service.jwt == "test.jwt.token")
        #expect(service.appleUserId == "apple.user.123")
    }

    @Test func initWithEmptyKeychainIsNotAuthenticated() {
        clearKeychain()

        let service = AuthService()

        #expect(service.isAuthenticated == false)
        #expect(service.jwt == nil)
        #expect(service.appleUserId == nil)
    }

    // MARK: - signOut

    @Test func signOutClearsStateAndKeychain() throws {
        clearKeychain()
        try seedKeychain(jwt: "test.jwt.token", appleUserId: "apple.user.123")
        defer { clearKeychain() }

        let service = AuthService()
        #expect(service.isAuthenticated == true)

        try service.signOut()

        #expect(service.isAuthenticated == false)
        #expect(service.jwt == nil)
        #expect(service.appleUserId == nil)

        // Verify Keychain was actually cleared
        let storedJWT = try KeychainService.read(.sessionJWT)
        let storedUserId = try KeychainService.read(.appleUserId)
        #expect(storedJWT == nil)
        #expect(storedUserId == nil)
    }

    // MARK: - ensureOrCreateStudent

    @Test func ensureOrCreateStudentThrowsWhenNotAuthenticated() throws {
        clearKeychain()

        let service = AuthService()
        #expect(service.appleUserId == nil)

        let config = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(for: Student.self, configurations: config)
        let context = ModelContext(container)

        #expect(throws: AuthError.notAuthenticated) {
            try service.ensureOrCreateStudent(in: context)
        }
    }
}
