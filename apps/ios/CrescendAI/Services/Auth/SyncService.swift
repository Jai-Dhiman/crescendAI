import Foundation
import SwiftData

enum SyncError: LocalizedError {
    case notAuthenticated
    case encodingFailed(Error)
    case networkError(Error)
    case serverError(String)

    var errorDescription: String? {
        switch self {
        case .notAuthenticated:
            "Not signed in"
        case .encodingFailed(let error):
            "Failed to encode sync data: \(error.localizedDescription)"
        case .networkError(let error):
            "Sync network error: \(error.localizedDescription)"
        case .serverError(let message):
            "Sync failed: \(message)"
        }
    }
}

@MainActor
@Observable
final class SyncService {
    private let authService: AuthService
    private(set) var isSyncing = false
    private(set) var lastSyncError: String?

    init(authService: AuthService) {
        self.authService = authService
    }

    func syncAfterSession(student: Student, session: PracticeSessionRecord, in modelContext: ModelContext) async {
        guard authService.isAuthenticated else { return }

        isSyncing = true
        lastSyncError = nil

        do {
            try await performSync(student: student, newSessions: [session])
            session.synced = true
            try modelContext.save()
            student.lastSyncedAt = Date()
            try modelContext.save()
        } catch {
            lastSyncError = error.localizedDescription
            print("[SyncService] Sync failed: \(error)")
        }

        isSyncing = false
    }

    func syncOnLaunch(student: Student, in modelContext: ModelContext) async {
        guard authService.isAuthenticated else { return }

        // Skip if synced recently (within 1 hour)
        if let lastSync = student.lastSyncedAt,
           Date().timeIntervalSince(lastSync) < 3600 {
            return
        }

        isSyncing = true
        lastSyncError = nil

        do {
            // Find unsynced sessions
            let studentId = student.appleUserId
            let predicate = #Predicate<PracticeSessionRecord> {
                $0.student?.appleUserId == studentId && !$0.synced && $0.endedAt != nil
            }
            let descriptor = FetchDescriptor<PracticeSessionRecord>(predicate: predicate)
            let unsyncedSessions = try modelContext.fetch(descriptor)

            if !unsyncedSessions.isEmpty {
                try await performSync(student: student, newSessions: unsyncedSessions)
                for session in unsyncedSessions {
                    session.synced = true
                }
                try modelContext.save()
            }

            student.lastSyncedAt = Date()
            try modelContext.save()
        } catch {
            lastSyncError = error.localizedDescription
            print("[SyncService] Launch sync failed: \(error)")
        }

        isSyncing = false
    }

    // MARK: - Private

    private func performSync(student: Student, newSessions: [PracticeSessionRecord]) async throws {
        guard let jwt = authService.jwt else {
            throw SyncError.notAuthenticated
        }

        let url = APIEndpoints.sync()
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")

        let body = SyncRequestBody(
            student: StudentDelta(
                inferred_level: student.inferredLevel,
                baseline_dynamics: student.baselineDynamics,
                baseline_timing: student.baselineTiming,
                baseline_pedaling: student.baselinePedaling,
                baseline_articulation: student.baselineArticulation,
                baseline_phrasing: student.baselinePhrasing,
                baseline_interpretation: student.baselineInterpretation,
                baseline_session_count: student.baselineSessionCount,
                explicit_goals: student.explicitGoals
            ),
            new_sessions: newSessions.map { session in
                SessionDelta(
                    id: session.id.uuidString,
                    started_at: ISO8601DateFormatter().string(from: session.startedAt),
                    ended_at: session.endedAt.map { ISO8601DateFormatter().string(from: $0) },
                    avg_dynamics: sessionAverage(session, \.dynamics),
                    avg_timing: sessionAverage(session, \.timing),
                    avg_pedaling: sessionAverage(session, \.pedaling),
                    avg_articulation: sessionAverage(session, \.articulation),
                    avg_phrasing: sessionAverage(session, \.phrasing),
                    avg_interpretation: sessionAverage(session, \.interpretation),
                    observations_json: encodeObservations(session.observations),
                    chunks_summary_json: nil
                )
            },
            last_sync_timestamp: student.lastSyncedAt.map { ISO8601DateFormatter().string(from: $0) }
        )

        do {
            request.httpBody = try JSONEncoder().encode(body)
        } catch {
            throw SyncError.encodingFailed(error)
        }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw SyncError.networkError(error)
        }

        guard let httpResponse = response as? HTTPURLResponse,
              (200..<300).contains(httpResponse.statusCode) else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            if let errorBody = try? JSONDecoder().decode([String: String].self, from: data),
               let message = errorBody["error"] {
                throw SyncError.serverError(message)
            }
            throw SyncError.serverError("Server returned \(statusCode)")
        }
    }

    private func sessionAverage(_ session: PracticeSessionRecord, _ keyPath: KeyPath<ChunkResultRecord, Double>) -> Double? {
        let completed = session.chunks.filter { $0.inferenceStatus == .completed }
        guard !completed.isEmpty else { return nil }
        let sum = completed.reduce(0.0) { $0 + $1[keyPath: keyPath] }
        return sum / Double(completed.count)
    }

    private func encodeObservations(_ observations: [ObservationRecord]) -> String? {
        guard !observations.isEmpty else { return nil }
        let items = observations.map { obs in
            ["chunkIndex": "\(obs.chunkIndex)", "dimension": obs.dimension, "text": obs.text]
        }
        return (try? JSONEncoder().encode(items)).flatMap { String(data: $0, encoding: .utf8) }
    }
}

// MARK: - Sync DTOs

private struct SyncRequestBody: Encodable {
    let student: StudentDelta
    let new_sessions: [SessionDelta]
    let last_sync_timestamp: String?
}

private struct StudentDelta: Encodable {
    let inferred_level: String?
    let baseline_dynamics: Double?
    let baseline_timing: Double?
    let baseline_pedaling: Double?
    let baseline_articulation: Double?
    let baseline_phrasing: Double?
    let baseline_interpretation: Double?
    let baseline_session_count: Int?
    let explicit_goals: String?
}

private struct SessionDelta: Encodable {
    let id: String
    let started_at: String
    let ended_at: String?
    let avg_dynamics: Double?
    let avg_timing: Double?
    let avg_pedaling: Double?
    let avg_articulation: Double?
    let avg_phrasing: Double?
    let avg_interpretation: Double?
    let observations_json: String?
    let chunks_summary_json: String?
}
