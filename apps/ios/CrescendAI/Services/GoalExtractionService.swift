import Foundation
import SwiftData

struct ExtractedGoals: Codable {
    let pieces: [String]
    let focus_areas: [String]
    let deadlines: [GoalDeadline]
    let raw_text: String
}

struct GoalDeadline: Codable {
    let description: String
    let date: String?
}

struct StoredGoals: Codable {
    var pieces: [String]
    var focus_areas: [String]
    var deadlines: [GoalDeadline]

    static let empty = StoredGoals(pieces: [], focus_areas: [], deadlines: [])
}

/// Sends student messages to the Workers API for LLM-based goal extraction.
/// Merges extracted goals into the student's explicit_goals JSON.
@MainActor
@Observable
final class GoalExtractionService {
    private let authService: AuthService
    private(set) var isExtracting = false

    init(authService: AuthService) {
        self.authService = authService
    }

    /// Extract goals from a student message and merge into their record.
    func extractAndStore(message: String, student: Student, in modelContext: ModelContext) async throws {
        guard authService.isAuthenticated else {
            throw AuthError.notAuthenticated
        }

        isExtracting = true
        defer { isExtracting = false }

        let extracted = try await callExtractGoals(message: message)

        // Merge into local student record
        var existing = parseStoredGoals(student.explicitGoals)

        for piece in extracted.pieces where !existing.pieces.contains(piece) {
            existing.pieces.append(piece)
        }
        for area in extracted.focus_areas where !existing.focus_areas.contains(area) {
            existing.focus_areas.append(area)
        }
        for deadline in extracted.deadlines {
            existing.deadlines.append(deadline)
        }

        student.explicitGoals = encodeGoals(existing)
        try modelContext.save()
    }

    // MARK: - Private

    private func callExtractGoals(message: String) async throws -> ExtractedGoals {
        let url = APIEndpoints.extractGoals()

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        struct GoalRequest: Encodable { let message: String }
        request.httpBody = try JSONEncoder().encode(GoalRequest(message: message))

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200..<300).contains(httpResponse.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw AuthError.serverAuthFailed("Goal extraction failed with status \(code)")
        }

        return try JSONDecoder().decode(ExtractedGoals.self, from: data)
    }

    private func parseStoredGoals(_ json: String?) -> StoredGoals {
        guard let json, let data = json.data(using: .utf8) else {
            return .empty
        }
        return (try? JSONDecoder().decode(StoredGoals.self, from: data)) ?? .empty
    }

    private func encodeGoals(_ goals: StoredGoals) -> String? {
        guard let data = try? JSONEncoder().encode(goals) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
