import Foundation
import SwiftData

/// On-device student model updates: baseline EMA, level inference.
/// Runs after each practice session ends.
enum StudentModelService {
    private static let alpha: Double = 0.3

    static let dimensions: [String] = [
        "dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation",
    ]

    /// Update student baselines with exponential moving average after a session ends.
    static func updateBaselines(student: Student, session: PracticeSessionRecord) {
        let completed = session.chunks.filter { $0.inferenceStatus == .completed }
        guard !completed.isEmpty else { return }

        let count = Double(completed.count)

        let avgDynamics = completed.reduce(0.0) { $0 + $1.dynamics } / count
        let avgTiming = completed.reduce(0.0) { $0 + $1.timing } / count
        let avgPedaling = completed.reduce(0.0) { $0 + $1.pedaling } / count
        let avgArticulation = completed.reduce(0.0) { $0 + $1.articulation } / count
        let avgPhrasing = completed.reduce(0.0) { $0 + $1.phrasing } / count
        let avgInterpretation = completed.reduce(0.0) { $0 + $1.interpretation } / count

        student.baselineDynamics = ema(current: student.baselineDynamics, new: avgDynamics)
        student.baselineTiming = ema(current: student.baselineTiming, new: avgTiming)
        student.baselinePedaling = ema(current: student.baselinePedaling, new: avgPedaling)
        student.baselineArticulation = ema(current: student.baselineArticulation, new: avgArticulation)
        student.baselinePhrasing = ema(current: student.baselinePhrasing, new: avgPhrasing)
        student.baselineInterpretation = ema(current: student.baselineInterpretation, new: avgInterpretation)

        student.baselineSessionCount += 1

        updateInferredLevel(student: student)
    }

    /// Infer student level from average dimension scores.
    /// Simple heuristic for V1 (plan specifies rules-based approach).
    static func updateInferredLevel(student: Student) {
        let baselines = [
            student.baselineDynamics,
            student.baselineTiming,
            student.baselinePedaling,
            student.baselineArticulation,
            student.baselinePhrasing,
            student.baselineInterpretation,
        ].compactMap { $0 }

        guard !baselines.isEmpty else { return }

        let avg = baselines.reduce(0.0, +) / Double(baselines.count)

        // V1 heuristic based on score ranges (plan specifies <0.3, 0.3-0.6, >0.6)
        if avg < 0.3 {
            student.inferredLevel = "beginner"
        } else if avg < 0.6 {
            student.inferredLevel = "intermediate"
        } else {
            student.inferredLevel = "advanced"
        }
    }

    /// Get the baseline value for a named dimension.
    static func baseline(for dimension: String, student: Student) -> Double? {
        switch dimension {
        case "dynamics": return student.baselineDynamics
        case "timing": return student.baselineTiming
        case "pedaling": return student.baselinePedaling
        case "articulation": return student.baselineArticulation
        case "phrasing": return student.baselinePhrasing
        case "interpretation": return student.baselineInterpretation
        default: return nil
        }
    }

    /// Get the session average for a named dimension.
    static func sessionAverage(for dimension: String, session: PracticeSessionRecord) -> Double? {
        let completed = session.chunks.filter { $0.inferenceStatus == .completed }
        guard !completed.isEmpty else { return nil }
        let count = Double(completed.count)
        switch dimension {
        case "dynamics": return completed.reduce(0.0) { $0 + $1.dynamics } / count
        case "timing": return completed.reduce(0.0) { $0 + $1.timing } / count
        case "pedaling": return completed.reduce(0.0) { $0 + $1.pedaling } / count
        case "articulation": return completed.reduce(0.0) { $0 + $1.articulation } / count
        case "phrasing": return completed.reduce(0.0) { $0 + $1.phrasing } / count
        case "interpretation": return completed.reduce(0.0) { $0 + $1.interpretation } / count
        default: return nil
        }
    }

    // MARK: - Private

    private static func ema(current: Double?, new: Double) -> Double {
        guard let current else { return new }
        return alpha * new + (1.0 - alpha) * current
    }
}
