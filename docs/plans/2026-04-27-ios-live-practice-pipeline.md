# iOS Live Practice Pipeline Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** iOS users can record piano practice and receive real teacher observations and session synthesis from the live backend pipeline, with session history persisted in SwiftData and accessible from the sidebar.
**Spec:** docs/specs/2026-04-27-ios-live-practice-pipeline-design.md
**Style:** Follow `apps/ios/CLAUDE.md` — @Observable, @MainActor, async/await, explicit error handling, no silent fallbacks

---

## Task Groups

```
Group A (parallel): Task 1 (Auth), Task 2 (ArtifactConfig + ArtifactRenderer)
Group B (parallel, depends on A): Task 3 (PracticeSessionService), Task 4 (ChatService)
Group C (sequential, depends on B): Task 5 (Wire ChatViewModel)
Group D (parallel, depends on C): Task 6 (ConversationRecord + Sidebar), Task 7 (ChatView ArtifactRenderer)
```

---

## Task 1: Auth — Cookie-Based better-auth Session

**Group:** A (parallel with Task 2)

**Behavior being verified:** `AuthService.signIn(identityToken:userId:email:)` calls `POST /api/auth/sign-in/social` with the better-auth Apple body and returns without error when the server sets a session cookie; `isAuthenticated` becomes `true` without touching the Keychain.

**Interface under test:** `AuthService.signIn(identityToken:userId:email:)` (new public method)

**Files:**
- Modify: `apps/ios/CrescendAI/Networking/APIEndpoints.swift`
- Modify: `apps/ios/CrescendAI/Networking/APIClient.swift`
- Modify: `apps/ios/CrescendAI/Services/Auth/AuthService.swift`
- Modify: `apps/ios/CrescendAI/Services/Auth/SyncService.swift`
- Modify: `apps/ios/CrescendAI/Features/Profile/ProfileView.swift`
- Test: `apps/ios/CrescendAITests/AuthServiceTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// AuthServiceTests.swift
import XCTest
@testable import CrescendAI

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
    static var requestHandler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
xcodebuild test -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing CrescendAITests/AuthServiceTests 2>&1 | tail -30
```
Expected: FAIL — `AuthService` has no `init(session:)` parameter and no `signIn(identityToken:userId:email:)` method.

- [ ] **Step 3: Implement**

**`apps/ios/CrescendAI/Networking/APIEndpoints.swift`** — replace all stale endpoints:

```swift
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
```

**`apps/ios/CrescendAI/Networking/APIClient.swift`** — remove `attachAuth`, `upload`, `analyze`, and stale response types:

```swift
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
```

**`apps/ios/CrescendAI/Services/Auth/AuthService.swift`** — replace JWT/Keychain logic with cookie-based auth:

```swift
import AuthenticationServices
import Foundation
import Sentry

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
        }
    }
}
```

**`apps/ios/CrescendAI/Services/Auth/SyncService.swift`** — remove manual Bearer header (line ~103-106). Find and delete these lines:

```swift
// DELETE these lines in SyncService:
guard let jwt = authService.jwt else {
    // ... any guard returning on missing jwt
}
// AND delete:
request.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")
```

The exact replacement depends on the surrounding code — remove only the JWT guard and header injection; leave all other request construction intact.

**`apps/ios/CrescendAI/Features/Profile/ProfileView.swift`** — replace `appleUserId != nil` with `isAuthenticated` and remove `try` from `signOut()`:

Find: any `authService.appleUserId != nil` → replace with `authService.isAuthenticated`
Find: `try authService.signOut()` → replace with `authService.signOut()`

- [ ] **Step 4: Run test — verify it PASSES**

```bash
xcodebuild test -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing CrescendAITests/AuthServiceTests 2>&1 | tail -30
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/ios/CrescendAI/Networking/APIEndpoints.swift apps/ios/CrescendAI/Networking/APIClient.swift apps/ios/CrescendAI/Services/Auth/AuthService.swift apps/ios/CrescendAI/Services/Auth/SyncService.swift apps/ios/CrescendAI/Features/Profile/ProfileView.swift apps/ios/CrescendAITests/AuthServiceTests.swift && git commit -m "feat(ios): switch auth to better-auth cookie session, remove Bearer header injection"
```

---

## Task 2: ArtifactConfig + ArtifactRenderer

**Group:** A (parallel with Task 1)

**Behavior being verified:** `ArtifactConfig` decodes the three API component JSON shapes into typed Swift cases; `.unknown` is produced for unrecognized types; `ChatMessage` carries `artifacts: [ArtifactConfig]` and a `.teacher` role.

**Interface under test:** `JSONDecoder().decode(ArtifactConfig.self, from:)` with API fixture JSON

**Files:**
- Create: `apps/ios/CrescendAI/Models/ArtifactConfig.swift`
- Create: `apps/ios/CrescendAI/Features/Chat/ArtifactRenderer.swift`
- Modify: `apps/ios/CrescendAI/Features/Chat/ChatViewModel.swift` (ChatMessage + ChatMessageRole only)
- Test: `apps/ios/CrescendAITests/ArtifactConfigTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// ArtifactConfigTests.swift
import XCTest
@testable import CrescendAI

final class ArtifactConfigTests: XCTestCase {

    func test_decodeExerciseSet() throws {
        let json = """
        {
            "type": "exercise_set",
            "config": {
                "sourcePassage": "Chopin Op.9 mm.1-4",
                "targetSkill": "legato phrasing",
                "exercises": [
                    {
                        "title": "Slow practice",
                        "instruction": "Play mm.1-4 at 60bpm",
                        "focusDimension": "phrasing",
                        "exerciseId": "ex-001"
                    }
                ]
            }
        }
        """.data(using: .utf8)!

        let artifact = try JSONDecoder().decode(ArtifactConfig.self, from: json)

        guard case .exerciseSet(let config) = artifact else {
            XCTFail("Expected .exerciseSet, got \(artifact)")
            return
        }
        XCTAssertEqual(config.sourcePassage, "Chopin Op.9 mm.1-4")
        XCTAssertEqual(config.exercises.count, 1)
        XCTAssertEqual(config.exercises[0].exerciseId, "ex-001")
    }

    func test_decodeKeyboardGuide() throws {
        let json = """
        {
            "type": "keyboard_guide",
            "config": {
                "title": "Hand positioning",
                "description": "Keep wrist level",
                "hands": "both"
            }
        }
        """.data(using: .utf8)!

        let artifact = try JSONDecoder().decode(ArtifactConfig.self, from: json)

        guard case .keyboardGuide(let config) = artifact else {
            XCTFail("Expected .keyboardGuide, got \(artifact)")
            return
        }
        XCTAssertEqual(config.title, "Hand positioning")
        XCTAssertEqual(config.hands, "both")
    }

    func test_decodeScoreHighlight() throws {
        let json = """
        {
            "type": "score_highlight",
            "config": {
                "pieceId": "chopin-op9-no2",
                "highlights": [
                    { "bars": [1, 4], "dimension": "dynamics", "annotation": "forte here" }
                ]
            }
        }
        """.data(using: .utf8)!

        let artifact = try JSONDecoder().decode(ArtifactConfig.self, from: json)

        guard case .scoreHighlight(let config) = artifact else {
            XCTFail("Expected .scoreHighlight, got \(artifact)")
            return
        }
        XCTAssertEqual(config.pieceId, "chopin-op9-no2")
        XCTAssertEqual(config.highlights[0].bars, [1, 4])
    }

    func test_decodeUnknownType() throws {
        let json = """
        {
            "type": "future_card",
            "config": { "foo": "bar" }
        }
        """.data(using: .utf8)!

        let artifact = try JSONDecoder().decode(ArtifactConfig.self, from: json)

        guard case .unknown(let typeName) = artifact else {
            XCTFail("Expected .unknown, got \(artifact)")
            return
        }
        XCTAssertEqual(typeName, "future_card")
    }

    func test_chatMessageHasArtifactsAndTeacherRole() {
        let artifact = ArtifactConfig.unknown(type: "test")
        let message = ChatMessage(
            role: .teacher,
            text: "Here is your feedback",
            artifacts: [artifact]
        )
        XCTAssertEqual(message.artifacts.count, 1)
        XCTAssertEqual(message.role, .teacher)
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
xcodebuild test -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing CrescendAITests/ArtifactConfigTests 2>&1 | tail -30
```
Expected: FAIL — `ArtifactConfig` type does not exist; `ChatMessage` has no `artifacts` property; `.teacher` role does not exist.

- [ ] **Step 3: Implement**

**`apps/ios/CrescendAI/Models/ArtifactConfig.swift`** — new file:

```swift
import Foundation

struct ExerciseItem: Codable {
    let title: String
    let instruction: String
    let focusDimension: String
    let exerciseId: String
    let hands: String?
}

struct ExerciseSetConfig: Codable {
    let sourcePassage: String
    let targetSkill: String
    let exercises: [ExerciseItem]
}

struct ScoreHighlight: Codable {
    let bars: [Int]
    let dimension: String
    let annotation: String?
}

struct ScoreHighlightConfig: Codable {
    let pieceId: String
    let highlights: [ScoreHighlight]
}

struct KeyboardGuideConfig: Codable {
    let title: String
    let description: String
    let hands: String
    let fingering: String?
}

enum ArtifactConfig: Codable {
    case exerciseSet(ExerciseSetConfig)
    case scoreHighlight(ScoreHighlightConfig)
    case keyboardGuide(KeyboardGuideConfig)
    case unknown(type: String)

    private enum CodingKeys: String, CodingKey {
        case type, config
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type_ = try container.decode(String.self, forKey: .type)

        switch type_ {
        case "exercise_set":
            let config = try container.decode(ExerciseSetConfig.self, forKey: .config)
            self = .exerciseSet(config)
        case "score_highlight":
            let config = try container.decode(ScoreHighlightConfig.self, forKey: .config)
            self = .scoreHighlight(config)
        case "keyboard_guide":
            let config = try container.decode(KeyboardGuideConfig.self, forKey: .config)
            self = .keyboardGuide(config)
        default:
            self = .unknown(type: type_)
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .exerciseSet(let config):
            try container.encode("exercise_set", forKey: .type)
            try container.encode(config, forKey: .config)
        case .scoreHighlight(let config):
            try container.encode("score_highlight", forKey: .type)
            try container.encode(config, forKey: .config)
        case .keyboardGuide(let config):
            try container.encode("keyboard_guide", forKey: .type)
            try container.encode(config, forKey: .config)
        case .unknown(let type_):
            try container.encode(type_, forKey: .type)
        }
    }
}
```

**`apps/ios/CrescendAI/Features/Chat/ChatViewModel.swift`** — add `.teacher` to `ChatMessageRole` and `artifacts` to `ChatMessage`:

Replace the existing `ChatMessageRole` enum:
```swift
enum ChatMessageRole: Equatable {
    case user
    case system
    case observation
    case teacher
}
```

Replace the existing `ChatMessage` struct:
```swift
struct ChatMessage: Identifiable {
    let id = UUID()
    let role: ChatMessageRole
    let text: String
    let dimension: String?
    let timestamp: Date
    var elaboration: String?
    var artifacts: [ArtifactConfig]

    init(
        role: ChatMessageRole,
        text: String,
        dimension: String? = nil,
        timestamp: Date = .now,
        elaboration: String? = nil,
        artifacts: [ArtifactConfig] = []
    ) {
        self.role = role
        self.text = text
        self.dimension = dimension
        self.timestamp = timestamp
        self.elaboration = elaboration
        self.artifacts = artifacts
    }
}
```

**`apps/ios/CrescendAI/Features/Chat/ArtifactRenderer.swift`** — new file:

```swift
import SwiftUI

struct ArtifactRenderer: View {
    let config: ArtifactConfig

    var body: some View {
        switch config {
        case .exerciseSet(let c):
            ExerciseSetCard(config: c)
        case .scoreHighlight:
            ScoreHighlightPlaceholder()
        case .keyboardGuide(let c):
            KeyboardGuideCard(config: c)
        case .unknown:
            EmptyView()
        }
    }
}

// Placeholder until score rendering is in scope
private struct ScoreHighlightPlaceholder: View {
    var body: some View {
        RoundedRectangle(cornerRadius: 8)
            .fill(CrescendColor.surface2)
            .frame(height: 80)
            .overlay(
                Text("Score highlight")
                    .font(CrescendFont.labelSM())
                    .foregroundStyle(CrescendColor.tertiaryText)
            )
    }
}

// Minimal card views — styled to match the existing design system

struct ExerciseSetCard: View {
    let config: ExerciseSetConfig

    var body: some View {
        VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
            Text(config.targetSkill)
                .font(CrescendFont.labelMD())
                .foregroundStyle(CrescendColor.foreground)
            ForEach(config.exercises.indices, id: \.self) { i in
                let ex = config.exercises[i]
                VStack(alignment: .leading, spacing: CrescendSpacing.space1) {
                    Text(ex.title)
                        .font(CrescendFont.bodySM())
                        .foregroundStyle(CrescendColor.foreground)
                    Text(ex.instruction)
                        .font(CrescendFont.labelSM())
                        .foregroundStyle(CrescendColor.secondaryText)
                }
            }
        }
        .padding(CrescendSpacing.space3)
        .background(CrescendColor.surface2)
        .cornerRadius(8)
    }
}

struct KeyboardGuideCard: View {
    let config: KeyboardGuideConfig

    var body: some View {
        VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
            Text(config.title)
                .font(CrescendFont.labelMD())
                .foregroundStyle(CrescendColor.foreground)
            Text(config.description)
                .font(CrescendFont.bodySM())
                .foregroundStyle(CrescendColor.secondaryText)
        }
        .padding(CrescendSpacing.space3)
        .background(CrescendColor.surface2)
        .cornerRadius(8)
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
xcodebuild test -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing CrescendAITests/ArtifactConfigTests 2>&1 | tail -30
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/ios/CrescendAI/Models/ArtifactConfig.swift apps/ios/CrescendAI/Features/Chat/ArtifactRenderer.swift apps/ios/CrescendAI/Features/Chat/ChatViewModel.swift apps/ios/CrescendAITests/ArtifactConfigTests.swift && git commit -m "feat(ios): add ArtifactConfig codable enum, ArtifactRenderer, teacher role, artifacts field"
```

---

## Task 3: PracticeSessionService + ChunkProducer silence gate

**Group:** B (parallel with Task 4, depends on Group A)

**Behavior being verified:** `ChunkProducer.rms(samples:)` returns zero for silent samples and a positive value for non-silent ones; `PracticeSessionService.eventStream` emits a `.sessionStarted(conversationId:)` event after `start()` completes with a mock HTTP response.

**Interface under test:** `ChunkProducer.rms(samples:)` (static), `PracticeSessionService.start()` with mock URLSession

**Files:**
- Modify: `apps/ios/CrescendAI/Services/AudioEngine/ChunkProducer.swift`
- Modify: `apps/ios/CrescendAI/Services/AudioEngine/PracticeSessionManager.swift`
- Create: `apps/ios/CrescendAI/Services/Practice/PracticeSessionService.swift`
- Test: `apps/ios/CrescendAITests/PracticeSessionServiceTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// PracticeSessionServiceTests.swift
import XCTest
@testable import CrescendAI

final class PracticeSessionServiceTests: XCTestCase {

    // MARK: - RMS silence gate (static method, no network)

    func test_rms_silentSignalReturnsZero() {
        let samples = [Float](repeating: 0, count: 1000)
        XCTAssertEqual(ChunkProducer.rms(samples: samples), 0)
    }

    func test_rms_fullScaleSineReturnsNearPointSeven() {
        // RMS of a full-scale sine = 1/sqrt(2) ≈ 0.7071
        let N = 1024
        let samples = (0..<N).map { Float(sin(2 * .pi * Double($0) / Double(N))) }
        let result = ChunkProducer.rms(samples: samples)
        XCTAssertEqual(result, 0.7071, accuracy: 0.01)
    }

    // MARK: - PracticeSessionService.start() emits sessionStarted

    func test_start_emitsSessionStartedWithConversationId() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: config)

        MockURLProtocol.requestHandler = { request in
            XCTAssertEqual(request.url?.path, "/api/practice/start")
            let body = ["sessionId": "session-abc", "conversationId": "conv-xyz"]
            let data = try JSONEncoder().encode(body)
            let response = HTTPURLResponse(url: request.url!, statusCode: 201, httpVersion: nil, headerFields: nil)!
            return (response, data)
        }

        let service = PracticeSessionService(session: session)
        var receivedEvents: [PracticeEvent] = []

        let eventTask = Task {
            for await event in service.eventStream {
                receivedEvents.append(event)
                if case .sessionStarted = event { break }
            }
        }

        try await service.start()

        await eventTask.value
        await service.stop()

        XCTAssertTrue(receivedEvents.contains(where: {
            if case .sessionStarted(let cid) = $0 { return cid == "conv-xyz" }
            return false
        }))
        XCTAssertEqual(service.conversationId, "conv-xyz")
    }

    func test_start_throwsNetworkErrorWhenServerUnavailable() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: config)

        MockURLProtocol.requestHandler = { _ in
            throw URLError(.notConnectedToInternet)
        }

        let service = PracticeSessionService(session: session)

        do {
            try await service.start()
            XCTFail("Expected throw")
        } catch APIError.networkError {
            // expected
        }
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
xcodebuild test -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing CrescendAITests/PracticeSessionServiceTests 2>&1 | tail -30
```
Expected: FAIL — `ChunkProducer.rms(samples:)` is not defined; `PracticeSessionService` does not exist.

- [ ] **Step 3: Implement**

**`apps/ios/CrescendAI/Services/AudioEngine/ChunkProducer.swift`** — add static `rms` method and silence gate in `produceChunk()`. Locate the `produceChunk()` method and add before the inference block:

```swift
// Add as a public static method on ChunkProducer:
static func rms(samples: [Float]) -> Float {
    guard !samples.isEmpty else { return 0 }
    let sumOfSquares = samples.reduce(0) { $0 + $1 * $1 }
    return sqrt(sumOfSquares / Float(samples.count))
}
```

In `produceChunk()`, add a silence gate immediately after the `guard samples.count >= sampleRate` line:

```swift
// Add this guard after the existing sample count check:
guard Self.rms(samples: samples) >= 0.01 else { return }
```

**`apps/ios/CrescendAI/Services/AudioEngine/PracticeSessionManager.swift`** — add `chunkStream` computed property after the `chunkProducer` property declaration:

```swift
var chunkStream: AsyncStream<AudioChunk>? {
    chunkProducer?.chunkStream
}
```

**`apps/ios/CrescendAI/Services/Practice/PracticeSessionService.swift`** — new file:

```swift
import Foundation
import SwiftData

enum PracticeEvent {
    case sessionStarted(conversationId: String)
    case chunkUploaded(index: Int)
    case observation(text: String, dimension: String?, artifacts: [ArtifactConfig])
    case synthesis(text: String, artifacts: [ArtifactConfig])
    case reconnecting(attempt: Int)
    case error(String)
    case sessionEnded
}

protocol PracticeSessionServiceProtocol: AnyObject {
    var eventStream: AsyncStream<PracticeEvent> { get }
    var state: PracticeSessionService.State { get }
    var currentLevel: Float { get }
    var elapsedSeconds: TimeInterval { get }
    var conversationId: String? { get }
    func start() async throws
    func stop() async
    func askForFeedback() async
}

@MainActor
@Observable
final class PracticeSessionService: PracticeSessionServiceProtocol {

    enum State {
        case idle
        case connecting
        case recording
        case stopping
    }

    private(set) var state: State = .idle
    private(set) var currentLevel: Float = 0
    private(set) var elapsedSeconds: TimeInterval = 0
    private(set) var conversationId: String?

    var eventStream: AsyncStream<PracticeEvent> { _stream }

    private var _stream: AsyncStream<PracticeEvent>
    private var _continuation: AsyncStream<PracticeEvent>.Continuation

    private let session: URLSession
    private var sessionManager: PracticeSessionManager?
    private var webSocketTask: URLSessionWebSocketTask?
    private var sessionId: String?
    private var timerTask: Task<Void, Never>?
    private var uploadTask: Task<Void, Never>?
    private var wsTask: Task<Void, Never>?
    private var modelContext: ModelContext?

    private struct PracticeStartResponse: Decodable {
        let sessionId: String
        let conversationId: String
    }

    init(session: URLSession = .shared) {
        self.session = session
        var cont: AsyncStream<PracticeEvent>.Continuation!
        _stream = AsyncStream { cont = $0 }
        _continuation = cont
    }

    func configure(modelContext: ModelContext) {
        self.modelContext = modelContext
    }

    func start() async throws {
        state = .connecting

        // 1. Create server session
        var request = URLRequest(url: APIEndpoints.practiceStart())
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode([String: String]())

        let client = APIClient(session: session)
        let startResp: PracticeStartResponse = try await client.perform(request)
        self.sessionId = startResp.sessionId
        self.conversationId = startResp.conversationId

        // 2. Start audio capture (no inference — cloud replaces local)
        let mc = modelContext ?? ModelContext(try! ModelContainer(for: Schema([])))
        let manager = PracticeSessionManager(modelContext: mc)
        self.sessionManager = manager
        try await manager.startSession(inferenceProvider: nil)

        // 3. Connect WebSocket
        connectWebSocket(sessionId: startResp.sessionId, conversationId: startResp.conversationId)

        // 4. Start chunk upload loop
        startChunkUploader(sessionId: startResp.sessionId)

        // 5. Start elapsed timer
        timerTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                elapsedSeconds += 1
            }
        }

        state = .recording
        _continuation.yield(.sessionStarted(conversationId: startResp.conversationId))
    }

    func stop() async {
        state = .stopping
        timerTask?.cancel()
        timerTask = nil
        uploadTask?.cancel()
        uploadTask = nil

        await sessionManager?.endSession()
        sessionManager = nil

        // Wait up to 30s for synthesis
        let synthesisTimeout = Task {
            try? await Task.sleep(for: .seconds(30))
        }
        await synthesisTimeout.value

        wsTask?.cancel()
        wsTask = nil
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil

        state = .idle
        _continuation.yield(.sessionEnded)
    }

    func askForFeedback() async {
        // Feedback is delivered via the WebSocket observation stream — nothing to call
    }

    // MARK: - WebSocket

    private func connectWebSocket(sessionId: String, conversationId: String, attempt: Int = 0) {
        let url = APIEndpoints.practiceWs(sessionId: sessionId, conversationId: conversationId)
        let task = session.webSocketTask(with: url)
        webSocketTask = task
        task.resume()

        wsTask = Task {
            do {
                try await receiveWebSocketMessages(task: task)
            } catch {
                guard !Task.isCancelled, state == .recording else { return }
                let maxAttempts = 5
                guard attempt < maxAttempts else {
                    _continuation.yield(.error("WebSocket disconnected after \(maxAttempts) attempts"))
                    return
                }
                let delay = min(pow(2.0, Double(attempt)), 30.0)
                _continuation.yield(.reconnecting(attempt: attempt + 1))
                try? await Task.sleep(for: .seconds(delay))
                guard !Task.isCancelled, state == .recording else { return }
                connectWebSocket(sessionId: sessionId, conversationId: conversationId, attempt: attempt + 1)
            }
        }
    }

    private func receiveWebSocketMessages(task: URLSessionWebSocketTask) async throws {
        while !Task.isCancelled {
            let message = try await task.receive()
            guard case .string(let text) = message,
                  let data = text.data(using: .utf8) else { continue }
            handleWebSocketData(data)
        }
    }

    private func handleWebSocketData(_ data: Data) {
        struct WSEvent: Decodable {
            let type: String
            let text: String?
            let dimension: String?
            let components: [ArtifactConfig]?
        }

        guard let event = try? JSONDecoder().decode(WSEvent.self, from: data) else { return }

        switch event.type {
        case "observation":
            _continuation.yield(.observation(
                text: event.text ?? "",
                dimension: event.dimension,
                artifacts: event.components ?? []
            ))
        case "synthesis":
            _continuation.yield(.synthesis(
                text: event.text ?? "",
                artifacts: event.components ?? []
            ))
            if let cid = conversationId, let mc = modelContext {
                let record = ConversationRecord(
                    conversationId: cid,
                    sessionId: UUID(uuidString: sessionId ?? "") ?? UUID(),
                    startedAt: Date(),
                    synthesisText: event.text
                )
                mc.insert(record)
                try? mc.save()
            }
        case "error":
            _continuation.yield(.error(event.text ?? "Unknown error"))
        default:
            break
        }
    }

    // MARK: - Chunk Upload

    private func startChunkUploader(sessionId: String) {
        guard let manager = sessionManager else { return }
        var localIndex = 0
        uploadTask = Task {
            guard let stream = manager.chunkStream else { return }
            for await chunk in stream {
                guard !Task.isCancelled else { break }
                guard let fileURL = chunk.fileURL,
                      let audioData = try? Data(contentsOf: fileURL) else { continue }

                var req = URLRequest(url: APIEndpoints.practiceChunk(sessionId: sessionId, chunkIndex: localIndex))
                req.httpMethod = "POST"
                req.setValue("audio/aac", forHTTPHeaderField: "Content-Type")
                req.httpBody = audioData

                _ = try? await session.data(for: req)
                _continuation.yield(.chunkUploaded(index: localIndex))
                localIndex += 1
            }
        }
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
xcodebuild test -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing CrescendAITests/PracticeSessionServiceTests 2>&1 | tail -30
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/ios/CrescendAI/Services/AudioEngine/ChunkProducer.swift apps/ios/CrescendAI/Services/AudioEngine/PracticeSessionManager.swift apps/ios/CrescendAI/Services/Practice/PracticeSessionService.swift apps/ios/CrescendAITests/PracticeSessionServiceTests.swift && git commit -m "feat(ios): add PracticeSessionService with WebSocket reconnect, add RMS silence gate to ChunkProducer"
```

---

## Task 4: ChatService (SSE)

**Group:** B (parallel with Task 3, depends on Group A)

**Behavior being verified:** `SSEParser.parse(line:)` returns typed `ChatEvent` values for each SSE line shape; `ChatService.send()` emits a `.delta` event with the message text from a complete SSE sequence.

**Interface under test:** `SSEParser.parse(line:)` (internal struct), `ChatService.send(message:conversationId:)` with mock URLSession

**Files:**
- Create: `apps/ios/CrescendAI/Services/Chat/ChatService.swift`
- Test: `apps/ios/CrescendAITests/ChatServiceTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// ChatServiceTests.swift
import XCTest
@testable import CrescendAI

final class ChatServiceTests: XCTestCase {

    // MARK: - SSEParser unit tests

    func test_sseParser_parsesStartEvent() throws {
        let line = #"data: {"type":"start","conversationId":"c-001"}"#
        let event = SSEParser.parse(line: line)
        guard case .start(let cid) = event else {
            XCTFail("Expected .start, got \(String(describing: event))")
            return
        }
        XCTAssertEqual(cid, "c-001")
    }

    func test_sseParser_parsesDeltaEvent() {
        let line = #"data: {"type":"delta","text":"Hello world"}"#
        let event = SSEParser.parse(line: line)
        guard case .delta(let text) = event else {
            XCTFail("Expected .delta, got \(String(describing: event))")
            return
        }
        XCTAssertEqual(text, "Hello world")
    }

    func test_sseParser_parsesDoneEvent() {
        let event = SSEParser.parse(line: "data: [DONE]")
        guard case .done = event else {
            XCTFail("Expected .done, got \(String(describing: event))")
            return
        }
    }

    func test_sseParser_parsesToolResultWithComponents() throws {
        let componentsJson = #"[{"type":"keyboard_guide","config":{"title":"t","description":"d","hands":"both"}}]"#
        let escaped = componentsJson.replacingOccurrences(of: "\"", with: "\\\"")
        let line = #"data: {"type":"tool_result","componentsJson":"\#(escaped)"}"#
        let event = SSEParser.parse(line: line)
        guard case .toolResult(let artifacts) = event else {
            XCTFail("Expected .toolResult, got \(String(describing: event))")
            return
        }
        XCTAssertEqual(artifacts.count, 1)
        guard case .keyboardGuide(let config) = artifacts[0] else {
            XCTFail("Expected .keyboardGuide")
            return
        }
        XCTAssertEqual(config.title, "t")
    }

    func test_sseParser_returnsNilForNonDataLine() {
        let event = SSEParser.parse(line: ": keep-alive")
        XCTAssertNil(event)
    }

    // MARK: - ChatService integration

    func test_send_emitsDeltaEventFromSSEStream() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: config)

        let sseBody = """
        data: {"type":"start","conversationId":"c-001"}

        data: {"type":"delta","text":"Nice playing"}

        data: [DONE]

        """
        MockURLProtocol.requestHandler = { request in
            XCTAssertEqual(request.url?.path, "/api/chat")
            let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: ["Content-Type": "text/event-stream"])!
            return (response, Data(sseBody.utf8))
        }

        let service = ChatService(session: session)
        var events: [ChatEvent] = []

        for await event in service.send(message: "How am I doing?", conversationId: nil) {
            events.append(event)
        }

        XCTAssertTrue(events.contains(where: {
            if case .delta(let t) = $0 { return t == "Nice playing" }
            return false
        }))
        XCTAssertTrue(events.contains(where: { if case .done = $0 { return true }; return false }))
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
xcodebuild test -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing CrescendAITests/ChatServiceTests 2>&1 | tail -30
```
Expected: FAIL — `SSEParser`, `ChatEvent`, and `ChatService` do not exist.

- [ ] **Step 3: Implement**

**`apps/ios/CrescendAI/Services/Chat/ChatService.swift`** — new file:

```swift
import Foundation

enum ChatEvent {
    case start(conversationId: String)
    case delta(String)
    case toolStart(toolName: String)
    case toolResult(artifacts: [ArtifactConfig])
    case done
    case error(String)
}

struct SSEParser {
    private enum EventPayload: Decodable {
        case start(conversationId: String)
        case delta(text: String)
        case toolStart(toolName: String)
        case toolResult(componentsJson: String?)
        case done
        case error(message: String)
        case unknown

        private enum CodingKeys: String, CodingKey {
            case type, text, conversationId, componentsJson, toolName, message
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let type_ = try c.decode(String.self, forKey: .type)
            switch type_ {
            case "start":
                self = .start(conversationId: (try? c.decode(String.self, forKey: .conversationId)) ?? "")
            case "delta":
                self = .delta(text: (try? c.decode(String.self, forKey: .text)) ?? "")
            case "tool_start":
                self = .toolStart(toolName: (try? c.decode(String.self, forKey: .toolName)) ?? "")
            case "tool_result":
                self = .toolResult(componentsJson: try? c.decode(String.self, forKey: .componentsJson))
            case "done":
                self = .done
            case "error":
                self = .error(message: (try? c.decode(String.self, forKey: .message)) ?? "Unknown error")
            default:
                self = .unknown
            }
        }
    }

    static func parse(line: String) -> ChatEvent? {
        guard line.hasPrefix("data: ") else { return nil }
        let payload = String(line.dropFirst(6))

        if payload == "[DONE]" { return .done }

        guard let data = payload.data(using: .utf8),
              let parsed = try? JSONDecoder().decode(EventPayload.self, from: data) else {
            return nil
        }

        switch parsed {
        case .start(let cid): return .start(conversationId: cid)
        case .delta(let text): return .delta(text)
        case .toolStart(let name): return .toolStart(toolName: name)
        case .toolResult(let json):
            guard let json,
                  let data = json.data(using: .utf8),
                  let artifacts = try? JSONDecoder().decode([ArtifactConfig].self, from: data) else {
                return .toolResult(artifacts: [])
            }
            return .toolResult(artifacts: artifacts)
        case .done: return .done
        case .error(let msg): return .error(msg)
        case .unknown: return nil
        }
    }
}

protocol ChatServiceProtocol: AnyObject {
    func send(message: String, conversationId: String?) -> AsyncStream<ChatEvent>
}

@MainActor
@Observable
final class ChatService: ChatServiceProtocol {
    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func send(message: String, conversationId: String?) -> AsyncStream<ChatEvent> {
        AsyncStream { continuation in
            Task {
                do {
                    try await streamSSE(message: message, conversationId: conversationId, continuation: continuation)
                } catch {
                    continuation.yield(.error(error.localizedDescription))
                    continuation.finish()
                }
            }
        }
    }

    private func streamSSE(
        message: String,
        conversationId: String?,
        continuation: AsyncStream<ChatEvent>.Continuation
    ) async throws {
        var request = URLRequest(url: APIEndpoints.chat())
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")

        struct ChatBody: Encodable {
            let message: String
            let conversationId: String?
        }
        request.httpBody = try JSONEncoder().encode(ChatBody(message: message, conversationId: conversationId))

        let (bytes, response) = try await session.bytes(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200..<300).contains(httpResponse.statusCode) else {
            throw APIError.invalidResponse((response as? HTTPURLResponse)?.statusCode ?? 0)
        }

        var buffer = ""
        for try await byte in bytes {
            buffer.append(Character(UnicodeScalar(byte)))
            if buffer.hasSuffix("\n\n") {
                let lines = buffer.components(separatedBy: "\n")
                for line in lines {
                    if let event = SSEParser.parse(line: line) {
                        continuation.yield(event)
                        if case .done = event {
                            continuation.finish()
                            return
                        }
                    }
                }
                buffer = ""
            }
        }
        continuation.finish()
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
xcodebuild test -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing CrescendAITests/ChatServiceTests 2>&1 | tail -30
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/ios/CrescendAI/Services/Chat/ChatService.swift apps/ios/CrescendAITests/ChatServiceTests.swift && git commit -m "feat(ios): add ChatService with SSEParser for /api/chat streaming"
```

---

## Task 5: Wire ChatViewModel to Real Services

**Group:** C (sequential, depends on Group B)

**Behavior being verified:** `ChatViewModel` appends a `.teacher` `ChatMessage` when `ChatService` emits a `.delta` event; it appends an `.observation` message when `PracticeSessionService` emits an `.observation` event; all `Task.sleep` mock code is removed.

**Interface under test:** `ChatViewModel.messages` array contents after injecting mock services via `configureForTesting`

**Files:**
- Modify: `apps/ios/CrescendAI/Features/Chat/ChatViewModel.swift`
- Test: `apps/ios/CrescendAITests/ChatViewModelTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// ChatViewModelTests.swift
import XCTest
import SwiftData
@testable import CrescendAI

// Mock practice service
final class MockPracticeService: PracticeSessionServiceProtocol {
    var eventStream: AsyncStream<PracticeEvent>
    var _continuation: AsyncStream<PracticeEvent>.Continuation

    var state: PracticeSessionService.State = .idle
    var currentLevel: Float = 0
    var elapsedSeconds: TimeInterval = 0
    var conversationId: String?

    init() {
        var cont: AsyncStream<PracticeEvent>.Continuation!
        eventStream = AsyncStream { cont = $0 }
        _continuation = cont
    }

    func start() async throws { state = .recording }
    func stop() async { state = .idle }
    func askForFeedback() async {}
}

// Mock chat service
final class MockChatService: ChatServiceProtocol {
    var stubbedEvents: [ChatEvent] = []

    func send(message: String, conversationId: String?) -> AsyncStream<ChatEvent> {
        let events = stubbedEvents
        return AsyncStream { continuation in
            Task {
                for event in events { continuation.yield(event) }
                continuation.finish()
            }
        }
    }
}

final class ChatViewModelTests: XCTestCase {

    @MainActor
    func test_observationEventAppendsObservationMessage() async throws {
        let schema = Schema([Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self, CheckInRecord.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        let context = ModelContext(container)

        let practiceService = MockPracticeService()
        let chatService = MockChatService()

        let vm = ChatViewModel()
        vm.configureForTesting(modelContext: context, practiceService: practiceService, chatService: chatService)

        // Emit an observation event
        practiceService._continuation.yield(.observation(text: "Nice phrasing", dimension: "phrasing", artifacts: []))

        // Give the subscription task time to process
        try await Task.sleep(for: .milliseconds(50))

        let obsMessages = vm.messages.filter { $0.role == .observation }
        XCTAssertEqual(obsMessages.count, 1)
        XCTAssertEqual(obsMessages[0].text, "Nice phrasing")
        XCTAssertEqual(obsMessages[0].dimension, "phrasing")
    }

    @MainActor
    func test_sendMessageAppendsDeltaAsTeacherMessage() async throws {
        let schema = Schema([Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self, CheckInRecord.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        let context = ModelContext(container)

        let practiceService = MockPracticeService()
        let chatService = MockChatService()
        chatService.stubbedEvents = [
            .start(conversationId: "c-1"),
            .delta("Great work"),
            .done,
        ]

        let vm = ChatViewModel()
        vm.configureForTesting(modelContext: context, practiceService: practiceService, chatService: chatService)

        vm.inputText = "How am I doing?"
        await vm.sendMessage()

        // Wait for stream to complete
        try await Task.sleep(for: .milliseconds(50))

        let userMessages = vm.messages.filter { $0.role == .user }
        let teacherMessages = vm.messages.filter { $0.role == .teacher }

        XCTAssertEqual(userMessages.count, 1)
        XCTAssertEqual(userMessages[0].text, "How am I doing?")
        XCTAssertFalse(teacherMessages.isEmpty)

        let fullText = teacherMessages.map(\.text).joined()
        XCTAssertTrue(fullText.contains("Great work"))
    }

    @MainActor
    func test_synthesisEventAppendsSynthesisMessage() async throws {
        let schema = Schema([Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self, CheckInRecord.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        let context = ModelContext(container)

        let practiceService = MockPracticeService()
        let chatService = MockChatService()

        let vm = ChatViewModel()
        vm.configureForTesting(modelContext: context, practiceService: practiceService, chatService: chatService)

        practiceService._continuation.yield(.synthesis(text: "Session summary text", artifacts: []))
        try await Task.sleep(for: .milliseconds(50))

        let teacherMessages = vm.messages.filter { $0.role == .teacher }
        XCTAssertEqual(teacherMessages.count, 1)
        XCTAssertEqual(teacherMessages[0].text, "Session summary text")
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
xcodebuild test -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing CrescendAITests/ChatViewModelTests 2>&1 | tail -30
```
Expected: FAIL — `ChatViewModel` has no `configureForTesting` method; `sendMessage()` is synchronous and returns `Void`, not async.

- [ ] **Step 3: Implement**

Replace the entire body of `ChatViewModel.swift` with the wired version:

```swift
import Foundation
import SwiftData
import SwiftUI

enum ChatMessageRole: Equatable {
    case user
    case system
    case observation
    case teacher
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: ChatMessageRole
    let text: String
    let dimension: String?
    let timestamp: Date
    var elaboration: String?
    var artifacts: [ArtifactConfig]

    init(
        role: ChatMessageRole,
        text: String,
        dimension: String? = nil,
        timestamp: Date = .now,
        elaboration: String? = nil,
        artifacts: [ArtifactConfig] = []
    ) {
        self.role = role
        self.text = text
        self.dimension = dimension
        self.timestamp = timestamp
        self.elaboration = elaboration
        self.artifacts = artifacts
    }
}

enum PracticeState {
    case idle
    case recording
    case processing
}

@MainActor
@Observable
final class ChatViewModel {
    var messages: [ChatMessage] = []
    var inputText = ""
    var practiceState: PracticeState = .idle

    private var practiceService: (any PracticeSessionServiceProtocol)?
    private var chatService: (any ChatServiceProtocol)?
    private var modelContext: ModelContext?
    private var practiceEventTask: Task<Void, Never>?

    var currentLevel: Float { practiceService?.currentLevel ?? 0 }
    var sessionDuration: TimeInterval { practiceService?.elapsedSeconds ?? 0 }

    var greeting: String {
        let hour = Calendar.current.component(.hour, from: .now)
        if hour < 12 { return "Good morning! Ready to practice?" }
        if hour < 17 { return "Good afternoon! Ready to practice?" }
        return "Good evening! Ready to practice?" 
    }

    func configure(modelContext: ModelContext) {
        self.modelContext = modelContext
        let practice = PracticeSessionService()
        practice.configure(modelContext: modelContext)
        self.practiceService = practice
        self.chatService = ChatService()
        subscribeToEvents()
    }

    func configureForTesting(
        modelContext: ModelContext,
        practiceService: any PracticeSessionServiceProtocol,
        chatService: any ChatServiceProtocol
    ) {
        self.modelContext = modelContext
        self.practiceService = practiceService
        self.chatService = chatService
        subscribeToEvents()
    }

    func startPractice() async {
        addSystemMessage("Session started. Play when you're ready.")
        do {
            try await practiceService?.start()
            practiceState = .recording
        } catch {
            addSystemMessage("Could not start session: \(error.localizedDescription)")
        }
    }

    func stopPractice() async {
        practiceState = .processing
        addSystemMessage("Ending session...")
        await practiceService?.stop()
        practiceState = .idle
    }

    func pausePractice() {}

    func askForFeedback() async {
        await practiceService?.askForFeedback()
    }

    func sendMessage() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        messages.append(ChatMessage(role: .user, text: text))
        inputText = ""

        guard let chatService else { return }
        let conversationId = practiceService?.conversationId

        var teacherMsgIndex: Int? = nil
        var teacherArtifacts: [ArtifactConfig] = []

        for await event in chatService.send(message: text, conversationId: conversationId) {
            switch event {
            case .start:
                break
            case .delta(let chunk):
                if let idx = teacherMsgIndex {
                    messages[idx] = ChatMessage(
                        role: .teacher,
                        text: messages[idx].text + chunk,
                        artifacts: teacherArtifacts
                    )
                } else {
                    let msg = ChatMessage(role: .teacher, text: chunk, artifacts: [])
                    messages.append(msg)
                    teacherMsgIndex = messages.count - 1
                }
            case .toolResult(let artifacts):
                teacherArtifacts.append(contentsOf: artifacts)
                if let idx = teacherMsgIndex {
                    messages[idx] = ChatMessage(
                        role: .teacher,
                        text: messages[idx].text,
                        artifacts: teacherArtifacts
                    )
                }
            case .done:
                break
            case .error(let msg):
                addSystemMessage("Error: \(msg)")
            case .toolStart:
                break
            }
        }
    }

    func addSystemMessage(_ text: String) {
        messages.append(ChatMessage(role: .system, text: text))
    }

    func requestElaboration(for messageId: UUID) {}

    private func subscribeToEvents() {
        practiceEventTask?.cancel()
        guard let practiceService else { return }
        practiceEventTask = Task {
            for await event in practiceService.eventStream {
                guard !Task.isCancelled else { break }
                handlePracticeEvent(event)
            }
        }
    }

    private func handlePracticeEvent(_ event: PracticeEvent) {
        switch event {
        case .sessionStarted:
            break
        case .observation(let text, let dimension, let artifacts):
            messages.append(ChatMessage(role: .observation, text: text, dimension: dimension, artifacts: artifacts))
        case .synthesis(let text, let artifacts):
            messages.append(ChatMessage(role: .teacher, text: text, artifacts: artifacts))
        case .reconnecting(let attempt):
            addSystemMessage("Reconnecting... (attempt \(attempt))")
        case .error(let msg):
            addSystemMessage("Error: \(msg)")
        case .chunkUploaded:
            break
        case .sessionEnded:
            break
        }
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
xcodebuild test -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing CrescendAITests/ChatViewModelTests 2>&1 | tail -30
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/ios/CrescendAI/Features/Chat/ChatViewModel.swift apps/ios/CrescendAITests/ChatViewModelTests.swift && git commit -m "feat(ios): wire ChatViewModel to PracticeSessionService and ChatService, remove all Task.sleep mocks"
```

---

## Task 6: ConversationRecord + Sidebar

**Group:** D (parallel with Task 7, depends on Group C)

**Behavior being verified:** `ConversationRecord` inserts and fetches from an in-memory `ModelContainer`; `SidebarView` renders session titles from `@Query` results instead of hardcoded data.

**Interface under test:** `ModelContext.insert` + `ModelContext.fetch` for `ConversationRecord`; `SidebarView` body renders `ConversationRecord.title`

**Files:**
- Create: `apps/ios/CrescendAI/Models/ConversationRecord.swift`
- Modify: `apps/ios/CrescendAI/App/SidebarView.swift`
- Modify: `apps/ios/CrescendAI/App/CrescendAIApp.swift`
- Test: `apps/ios/CrescendAITests/ConversationRecordTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// ConversationRecordTests.swift
import XCTest
import SwiftData
@testable import CrescendAI

final class ConversationRecordTests: XCTestCase {

    func test_insertAndFetchConversationRecord() throws {
        let schema = Schema([ConversationRecord.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        let context = ModelContext(container)

        let record = ConversationRecord(
            conversationId: "conv-001",
            sessionId: UUID(),
            startedAt: Date(),
            synthesisText: "Good session"
        )
        context.insert(record)
        try context.save()

        let descriptor = FetchDescriptor<ConversationRecord>()
        let results = try context.fetch(descriptor)

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].conversationId, "conv-001")
        XCTAssertEqual(results[0].synthesisText, "Good session")
    }

    func test_conversationRecordTitleDefaultsToDate() throws {
        let schema = Schema([ConversationRecord.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        let context = ModelContext(container)

        let now = Date()
        let record = ConversationRecord(
            conversationId: "conv-002",
            sessionId: UUID(),
            startedAt: now,
            synthesisText: nil
        )
        context.insert(record)

        XCTAssertNil(record.title)
        // displayTitle falls back to formatted date
        XCTAssertFalse(record.displayTitle.isEmpty)
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
xcodebuild test -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing CrescendAITests/ConversationRecordTests 2>&1 | tail -30
```
Expected: FAIL — `ConversationRecord` does not exist.

- [ ] **Step 3: Implement**

**`apps/ios/CrescendAI/Models/ConversationRecord.swift`** — new file:

```swift
import Foundation
import SwiftData

@Model
final class ConversationRecord {
    @Attribute(.unique) var conversationId: String
    var sessionId: UUID
    var startedAt: Date
    var title: String?
    var synthesisText: String?

    init(conversationId: String, sessionId: UUID, startedAt: Date, title: String? = nil, synthesisText: String? = nil) {
        self.conversationId = conversationId
        self.sessionId = sessionId
        self.startedAt = startedAt
        self.title = title
        self.synthesisText = synthesisText
    }

    var displayTitle: String {
        title ?? DateFormatter.localizedString(from: startedAt, dateStyle: .medium, timeStyle: .short)
    }
}
```

**`apps/ios/CrescendAI/App/CrescendAIApp.swift`** — add `ConversationRecord.self` to the schema array:

```swift
// In the Schema([...]) array, add ConversationRecord.self:
let schema = Schema([
    Student.self,
    PracticeSessionRecord.self,
    ChunkResultRecord.self,
    ObservationRecord.self,
    CheckInRecord.self,
    ConversationRecord.self,
])
```

**`apps/ios/CrescendAI/App/SidebarView.swift`** — replace hardcoded sessions with `@Query`:

```swift
import SwiftData
import SwiftUI

struct SidebarView: View {
    @Binding var isOpen: Bool
    let onNewSession: () -> Void
    let onSelectSession: (String) -> Void  // conversationId instead of UUID
    let onShowProfile: () -> Void

    @Query(sort: \ConversationRecord.startedAt, order: .reverse) private var sessions: [ConversationRecord]

    var body: some View {
        ZStack(alignment: .leading) {
            if isOpen {
                Color.black.opacity(0.4)
                    .ignoresSafeArea()
                    .onTapGesture { closeSidebar() }
                    .transition(.opacity)
            }

            if isOpen {
                drawerContent
                    .frame(width: 280)
                    .frame(maxHeight: .infinity)
                    .background(CrescendColor.sidebarBackground)
                    .ignoresSafeArea(edges: .bottom)
                    .transition(.move(edge: .leading))
            }
        }
        .animation(.easeInOut(duration: 0.25), value: isOpen)
        .gesture(
            DragGesture()
                .onEnded { value in
                    if value.translation.width < -50 { closeSidebar() }
                }
        )
    }

    private var drawerContent: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("crescend")
                    .font(CrescendFont.displayMD())
                    .foregroundStyle(CrescendColor.foreground)

                Spacer()

                Button(action: onNewSession) {
                    Image(systemName: "square.and.pencil")
                        .font(.system(size: 18, weight: .medium))
                        .foregroundStyle(CrescendColor.foreground)
                        .frame(width: 36, height: 36)
                        .contentShape(Rectangle())
                }
                .buttonStyle(CrescendPressStyle())
            }
            .padding(.horizontal, CrescendSpacing.space4)
            .padding(.top, CrescendSpacing.space2)
            .padding(.bottom, CrescendSpacing.space6)

            ScrollView {
                LazyVStack(alignment: .leading, spacing: CrescendSpacing.space1) {
                    Text("Recents")
                        .font(CrescendFont.labelMD())
                        .foregroundStyle(CrescendColor.tertiaryText)
                        .padding(.horizontal, CrescendSpacing.space4)
                        .padding(.bottom, CrescendSpacing.space2)

                    ForEach(sessions) { session in
                        Button {
                            onSelectSession(session.conversationId)
                            closeSidebar()
                        } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(session.displayTitle)
                                    .font(CrescendFont.bodyMD())
                                    .foregroundStyle(CrescendColor.foreground)
                                    .lineLimit(1)

                                Text(session.startedAt, style: .relative)
                                    .font(CrescendFont.labelSM())
                                    .foregroundStyle(CrescendColor.tertiaryText)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, CrescendSpacing.space4)
                            .padding(.vertical, CrescendSpacing.space2)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(CrescendPressStyle())
                    }
                }
            }

            Spacer()

            Button(action: {
                onShowProfile()
                closeSidebar()
            }) {
                HStack(spacing: CrescendSpacing.space3) {
                    Circle()
                        .fill(CrescendColor.surface2)
                        .frame(width: 32, height: 32)
                        .overlay {
                            Image(systemName: "person.fill")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundStyle(CrescendColor.secondaryText)
                        }

                    Text("Profile")
                        .font(CrescendFont.bodyMD())
                        .foregroundStyle(CrescendColor.foreground)

                    Spacer()
                }
                .padding(.horizontal, CrescendSpacing.space4)
                .padding(.vertical, CrescendSpacing.space3)
                .contentShape(Rectangle())
            }
            .buttonStyle(CrescendPressStyle())
            .padding(.bottom, CrescendSpacing.space4)

            Divider()
                .overlay(CrescendColor.border)
                .padding(.horizontal, CrescendSpacing.space4)
        }
    }

    private func closeSidebar() {
        isOpen = false
    }
}
```

Note: the `onSelectSession` callback type changes from `(UUID)` to `(String)` (conversationId). Update all call sites in `ContentView.swift` or wherever `SidebarView` is instantiated to match the new signature.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
xcodebuild test -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' -only-testing CrescendAITests/ConversationRecordTests 2>&1 | tail -30
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/ios/CrescendAI/Models/ConversationRecord.swift apps/ios/CrescendAI/App/SidebarView.swift apps/ios/CrescendAI/App/CrescendAIApp.swift apps/ios/CrescendAITests/ConversationRecordTests.swift && git commit -m "feat(ios): add ConversationRecord SwiftData model, wire SidebarView to @Query"
```

---

## Task 7: ChatView ArtifactRenderer Integration

**Group:** D (parallel with Task 6, depends on Group C)

**Behavior being verified:** `ChatView.chatRow()` renders `ArtifactRenderer` for messages whose `artifacts` array is non-empty; messages with empty artifacts render no `ArtifactRenderer`.

**Interface under test:** `ChatView` body — SwiftUI view structure (verified via Xcode Previews and visual inspection; no XCTest for pure view layout)

**Files:**
- Modify: `apps/ios/CrescendAI/Features/Chat/ChatView.swift`

> Note: SwiftUI view rendering cannot be meaningfully unit-tested without a third-party library. This task is verified by running the app in the simulator and confirming artifact cards appear in the chat for observation messages that carry components. The build step is the verification.

- [ ] **Step 1: Locate the chatRow function in ChatView.swift**

```bash
grep -n "chatRow\|ObservationCard\|func chat" apps/ios/CrescendAI/Features/Chat/ChatView.swift
```

- [ ] **Step 2: Add ArtifactRenderer to chatRow**

In `ChatView.swift`, locate the view builder or function that renders a single `ChatMessage`. After the message text, add artifact rendering:

```swift
// Inside the message bubble / row view, after the Text(message.text) view:
if !message.artifacts.isEmpty {
    VStack(spacing: CrescendSpacing.space2) {
        ForEach(message.artifacts.indices, id: \.self) { i in
            ArtifactRenderer(config: message.artifacts[i])
        }
    }
    .padding(.top, CrescendSpacing.space2)
}
```

The exact insertion point depends on the current `ChatView.swift` structure. The rule is: add it immediately after the text label in the same VStack that constitutes the message bubble.

- [ ] **Step 3: Verify in simulator**

```bash
xcodebuild build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -20
```
Expected: BUILD SUCCEEDED. Boot the simulator and confirm the app launches without crashes. Artifact cards will appear once a real practice session returns observation components from the backend.

- [ ] **Step 4: Commit**

```bash
git add apps/ios/CrescendAI/Features/Chat/ChatView.swift && git commit -m "feat(ios): render ArtifactRenderer in chatRow for messages with non-empty artifacts"
```

---

## Completion Criteria

All 7 tasks committed, all tests passing:

```bash
xcodebuild test -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | grep -E "PASS|FAIL|error:"
```

Expected: All test suites PASS, zero build errors.
