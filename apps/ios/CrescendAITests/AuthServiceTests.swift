import XCTest
@testable import CrescendAI

@MainActor
final class AuthServiceTests: XCTestCase {

    func test_signIn_callsSocialEndpointAndSetsAuthenticated() async throws {
        // Arrange: stub URLSession that returns a 200 with Set-Cookie header
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: config)

        MockURLProtocol.requestHandler = { request in
            XCTAssertEqual(request.url?.path, "/api/auth/sign-in/social")
            XCTAssertEqual(request.httpMethod, "POST")
            let body = try JSONDecoder().decode([String: AnyCodable].self, from: request.httpBody!)
            XCTAssertEqual(body["provider"]?.value as? String, "apple")

            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Set-Cookie": "better-auth.session_token=abc123; Path=/; HttpOnly"]
            )!
            return (response, Data("{}".utf8))
        }

        let service = AuthService(session: session)

        // Act
        try await service.signIn(identityToken: "fake.jwt.token", userId: "user123", email: "test@test.com")

        // Assert
        XCTAssertTrue(service.isAuthenticated)
        XCTAssertEqual(service.appleUserId, "user123")
    }

    func test_signIn_throwsServerAuthFailedOn401() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: config)

        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 401,
                httpVersion: nil,
                headerFields: nil
            )!
            return (response, Data(#"{"error":"invalid token"}"#.utf8))
        }

        let service = AuthService(session: session)

        do {
            try await service.signIn(identityToken: "bad.token", userId: "u", email: nil)
            XCTFail("Expected throw")
        } catch AuthError.serverAuthFailed(let msg) {
            XCTAssertEqual(msg, "invalid token")
        }
        XCTAssertFalse(service.isAuthenticated)
    }

    func test_signOut_clearsAuthState() {
        let service = AuthService(session: .shared)
        // Manually put it in authenticated state
        service._setAuthenticatedForTesting(userId: "u1")
        XCTAssertTrue(service.isAuthenticated)

        service.signOut()

        XCTAssertFalse(service.isAuthenticated)
        XCTAssertNil(service.appleUserId)
    }
}

// Minimal AnyCodable for decoding body in tests
struct AnyCodable: Decodable {
    let value: Any
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) { value = s }
        else if let i = try? c.decode(Int.self) { value = i }
        else if let b = try? c.decode(Bool.self) { value = b }
        else { value = "" }
    }
}

final class MockURLProtocol: URLProtocol {
    nonisolated(unsafe) static var requestHandler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = MockURLProtocol.requestHandler else {
            client?.urlProtocol(self, didFailWithError: URLError(.unknown))
            return
        }
        do {
            var mutableRequest = request
            if let body = request.httpBodyStream {
                body.open()
                var data = Data()
                let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 1024)
                defer { buffer.deallocate() }
                while body.hasBytesAvailable {
                    let n = body.read(buffer, maxLength: 1024)
                    data.append(buffer, count: n)
                }
                mutableRequest.httpBody = data
            }
            let (response, data) = try handler(mutableRequest)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
