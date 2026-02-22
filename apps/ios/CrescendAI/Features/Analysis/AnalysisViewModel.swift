import SwiftUI

@MainActor
@Observable
final class AnalysisViewModel {
    let result: AnalysisResult

    var labeledScores: [(String, Double)] {
        result.calibrated_dimensions.toLabeledPairs()
    }

    init(result: AnalysisResult) {
        self.result = result
    }

    func scoreLabel(_ score: Double) -> String {
        if score >= 0.7 { return "Strong" }
        if score >= 0.5 { return "Good" }
        if score >= 0.3 { return "Developing" }
        return "Needs focus"
    }
}
