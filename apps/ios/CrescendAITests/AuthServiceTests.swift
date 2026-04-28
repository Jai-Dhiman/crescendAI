import XCTest
import SwiftData
@testable import CrescendAI

@MainActor
final class AuthServiceTests: XCTestCase {

    override func setUp() {
        super.setUp()
        try? KeychainService.deleteAll()
    }

    override func tearDown() {
        try? KeychainService.deleteAll()
        super.tearDown()
    }

    // MARK: - loadStoredCredentials (via init)

    func testInitWithStoredCredentialsSetsAuthenticated() throws {
        try KeychainService.save("test.jwt.token", for: .sessionJWT)
        try KeychainService.save("apple.user.123", for: .appleUserId)

        let service = AuthService()

        XCTAssertTrue(service.isAuthenticated)
        XCTAssertEqual(service.jwt, "test.jwt.token")
        XCTAssertEqual(service.appleUserId, "apple.user.123")
    }

    func testInitWithEmptyKeychainIsNotAuthenticated() {
        let service = AuthService()

        XCTAssertFalse(service.isAuthenticated)
        XCTAssertNil(service.jwt)
        XCTAssertNil(service.appleUserId)
    }

    // MARK: - signOut

    func testSignOutClearsStateAndKeychain() throws {
        try KeychainService.save("test.jwt.token", for: .sessionJWT)
        try KeychainService.save("apple.user.123", for: .appleUserId)

        let service = AuthService()
        XCTAssertTrue(service.isAuthenticated)

        try service.signOut()

        XCTAssertFalse(service.isAuthenticated)
        XCTAssertNil(service.jwt)
        XCTAssertNil(service.appleUserId)

        let storedJWT = try KeychainService.read(.sessionJWT)
        let storedUserId = try KeychainService.read(.appleUserId)
        XCTAssertNil(storedJWT)
        XCTAssertNil(storedUserId)
    }

    // MARK: - ensureOrCreateStudent

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
}
