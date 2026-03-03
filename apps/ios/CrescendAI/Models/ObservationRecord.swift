import Foundation
import SwiftData

@Model
final class ObservationRecord {
    var chunkIndex: Int
    var dimension: String
    var text: String
    var elaboration: String?
    var createdAt: Date

    var session: PracticeSessionRecord?

    init(
        chunkIndex: Int,
        dimension: String,
        text: String,
        elaboration: String? = nil,
        session: PracticeSessionRecord? = nil
    ) {
        self.chunkIndex = chunkIndex
        self.dimension = dimension
        self.text = text
        self.elaboration = elaboration
        self.createdAt = Date()
        self.session = session
    }
}
