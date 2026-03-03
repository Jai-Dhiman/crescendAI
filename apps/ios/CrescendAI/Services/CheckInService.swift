import Foundation
import SwiftData

/// Generates post-session check-in questions based on trigger rules.
/// Never interrupts practice -- only fires at session end.
@MainActor
@Observable
final class CheckInService {
    private(set) var pendingQuestion: String?
    private(set) var questionSessionId: UUID?

    /// Evaluate whether a check-in is warranted after this session.
    /// Returns a question string if triggered, nil otherwise.
    func evaluateCheckIn(student: Student, session: PracticeSessionRecord) -> String? {
        // No check-ins until session 3+
        guard student.baselineSessionCount >= 3 else { return nil }

        // Max one check-in per session (only if none pending)
        guard pendingQuestion == nil else { return nil }

        let question: String?

        if let repertoireQuestion = checkRepertoireTrigger(student: student) {
            question = repertoireQuestion
        } else if let progressQuestion = checkProgressTrigger(student: student) {
            question = progressQuestion
        } else if student.baselineSessionCount >= 5 {
            question = checkOpenEndedTrigger()
        } else {
            question = nil
        }

        if let question {
            pendingQuestion = question
            questionSessionId = session.id
        }

        return question
    }

    /// Record the student's answer to the pending check-in.
    func recordAnswer(_ answer: String?, in modelContext: ModelContext) {
        guard let question = pendingQuestion, let sessionId = questionSessionId else { return }

        // Store in a lightweight model for future D1 sync
        // For now, persist as a check-in record that the sync service can pick up
        let checkIn = CheckInRecord(
            sessionId: sessionId,
            question: question,
            answer: answer
        )
        modelContext.insert(checkIn)
        do {
            try modelContext.save()
        } catch {
            print("[CheckInService] Failed to save check-in: \(error)")
        }

        pendingQuestion = nil
        questionSessionId = nil
    }

    func dismiss() {
        pendingQuestion = nil
        questionSessionId = nil
    }

    // MARK: - Trigger Rules

    /// Triggered when the same piece appears in 3+ recent sessions.
    /// Since we don't have piece identification yet, this checks for
    /// consistent practice patterns (many sessions in a short time).
    private func checkRepertoireTrigger(student: Student) -> String? {
        let recentSessions = student.sessions
            .sorted { $0.startedAt > $1.startedAt }
            .prefix(5)

        guard recentSessions.count >= 3 else { return nil }

        // Check if sessions are clustered (all within last 3 days)
        if let oldest = recentSessions.last?.startedAt,
           Date().timeIntervalSince(oldest) < 3 * 24 * 3600 {
            return "I notice you've been practicing frequently. Are you preparing for something?"
        }

        return nil
    }

    /// Triggered when a dimension improves by >0.1 over recent sessions.
    private func checkProgressTrigger(student: Student) -> String? {
        guard student.sessions.count >= 3 else { return nil }

        let recent = student.sessions
            .sorted { $0.startedAt > $1.startedAt }
            .prefix(3)

        for dimension in StudentModelService.dimensions {
            guard let baseline = StudentModelService.baseline(for: dimension, student: student) else {
                continue
            }

            // Compare recent session averages to baseline
            var recentAvg = 0.0
            var count = 0
            for session in recent {
                if let avg = StudentModelService.sessionAverage(for: dimension, session: session) {
                    recentAvg += avg
                    count += 1
                }
            }

            guard count > 0 else { continue }
            recentAvg /= Double(count)

            let improvement = recentAvg - baseline
            if improvement > 0.1 {
                let dimName = dimension.capitalized
                return "Your \(dimName.lowercased()) has been improving over the last few sessions. Is that something you've been focusing on?"
            }
        }

        return nil
    }

    /// 10% random chance after session 5+.
    private func checkOpenEndedTrigger() -> String? {
        let roll = Double.random(in: 0...1)
        guard roll < 0.1 else { return nil }
        return "Is there anything specific you'd like me to pay attention to in your playing?"
    }
}
