import Foundation
import SwiftData

@Model
final class Student {
    @Attribute(.unique) var appleUserId: String
    var email: String?
    var inferredLevel: String?

    // 6-dimension baselines (exponential moving average)
    var baselineDynamics: Double?
    var baselineTiming: Double?
    var baselinePedaling: Double?
    var baselineArticulation: Double?
    var baselinePhrasing: Double?
    var baselineInterpretation: Double?
    var baselineSessionCount: Int

    var explicitGoals: String?
    var lastSyncedAt: Date?
    var createdAt: Date

    @Relationship(deleteRule: .cascade, inverse: \PracticeSessionRecord.student)
    var sessions: [PracticeSessionRecord]

    init(
        appleUserId: String,
        email: String? = nil,
        inferredLevel: String? = nil,
        baselineSessionCount: Int = 0,
        explicitGoals: String? = nil
    ) {
        self.appleUserId = appleUserId
        self.email = email
        self.inferredLevel = inferredLevel
        self.baselineDynamics = nil
        self.baselineTiming = nil
        self.baselinePedaling = nil
        self.baselineArticulation = nil
        self.baselinePhrasing = nil
        self.baselineInterpretation = nil
        self.baselineSessionCount = baselineSessionCount
        self.explicitGoals = explicitGoals
        self.lastSyncedAt = nil
        self.createdAt = Date()
        self.sessions = []
    }
}
