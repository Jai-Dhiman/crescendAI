import XCTest
import SwiftData
@testable import CrescendAI

final class AuthServiceTests: XCTestCase {

    override func setUp() async throws {
        try await super.setUp()
        try? KeychainService.deleteAll()
        await MainActor.run { AuthServiceTests.clearSessionCookie() }
    }

    override func tearDown() async throws {
        try? KeychainService.deleteAll()
        await MainActor.run { AuthServiceTests.clearSessionCookie() }
        try await super.tearDown()
    }

    @MainActor
    private static func clearSessionCookie() {
        HTTPCookieStorage.shared.cookies?.filter {
            $0.name == "better-auth.session_token"
        }.forEach { HTTPCookieStorage.shared.deleteCookie($0) }
    }

    @MainActor
    private static func setSessionCookie() {
        guard let cookie = HTTPCookie(properties: [
            .name: "better-auth.session_token",
            .value: "test.session.token",
            .domain: "api.crescend.ai",
            .path: "/",
            .expires: Date.distantFuture
        ]) else { return }
        HTTPCookieStorage.shared.setCookie(cookie)
    }

    // MARK: - loadStoredCredentials (via init)

    @MainActor
    func testInitWithActiveSessionSetsAuthenticated() {
        Self.setSessionCookie()
        try? KeychainService.save("apple.user.123", for: .appleUserId)

        let service = AuthService()

        XCTAssertTrue(service.isAuthenticated)
        XCTAssertEqual(service.appleUserId, "apple.user.123")
    }

    @MainActor
    func testInitWithoutSessionIsNotAuthenticated() {
        let service = AuthService()

        XCTAssertFalse(service.isAuthenticated)
        XCTAssertNil(service.appleUserId)
    }

    // MARK: - signOut

    @MainActor
    func testSignOutClearsStateAndCookie() {
        Self.setSessionCookie()
        try? KeychainService.save("apple.user.123", for: .appleUserId)

        let service = AuthService()
        XCTAssertTrue(service.isAuthenticated)

        service.signOut()

        XCTAssertFalse(service.isAuthenticated)
        XCTAssertNil(service.appleUserId)

        let hasCookie = HTTPCookieStorage.shared.cookies?.contains {
            $0.name == "better-auth.session_token"
        } ?? false
        XCTAssertFalse(hasCookie)
    }

    // MARK: - ensureOrCreateStudent

    @MainActor
    func testEnsureOrCreateStudentThrowsWhenNotAuthenticated() throws {
        let service = AuthService()
        XCTAssertNil(service.appleUserId)

        let config = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(for: Student.self, configurations: config)
        let context = ModelContext(container)

        XCTAssertThrowsError(try service.ensureOrCreateStudent(in: context)) { error in
            guard let authError = error as? AuthError, case .notAuthenticated = authError else {
                XCTFail("Expected AuthError.notAuthenticated, got \(error)")
                return
            }
        }
    }

    // MARK: - Keychain persistence

    @MainActor
    func testSignInPersistsAppleUserIdToKeychain() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: config)
        MockURLProtocol.requestHandler = { request in
            (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!, Data())
        }
        defer { MockURLProtocol.requestHandler = nil }

        let service = AuthService(session: session)
        try await service.signIn(identityToken: "token", userId: "apple.user.456", email: nil)

        XCTAssertEqual(try KeychainService.read(.appleUserId), "apple.user.456")
    }

    @MainActor
    func testSignOutClearsKeychain() throws {
        Self.setSessionCookie()
        try KeychainService.save("apple.user.123", for: .appleUserId)

        let service = AuthService()
        XCTAssertEqual(service.appleUserId, "apple.user.123")

        service.signOut()

        XCTAssertNil(try KeychainService.read(.appleUserId))
    }
}
