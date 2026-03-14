import Foundation

/// 6-dimensional piano performance evaluation (A1-Max model).
/// Mirrors the teacher-grounded taxonomy dimensions.
struct PerformanceDimensions: Codable, Equatable {
    let dynamics: Double
    let timing: Double
    let pedaling: Double
    let articulation: Double
    let phrasing: Double
    let interpretation: Double

    func toLabeledPairs() -> [(String, Double)] {
        [
            ("Dynamics", dynamics),
            ("Timing", timing),
            ("Pedaling", pedaling),
            ("Articulation", articulation),
            ("Phrasing", phrasing),
            ("Interpretation", interpretation),
        ]
    }
}

struct PracticeTip: Codable, Equatable {
    let title: String
    let description: String
}

struct ModelResult: Codable, Equatable {
    let model_name: String
    let model_type: String
    let pairwise_accuracy: Double
    let dimensions: PerformanceDimensions
}

struct Citation: Codable, Equatable {
    let text: String
    let source: String
}

struct CitedFeedback: Codable, Equatable {
    let html: String
    let plain_text: String
    let citations: [Citation]
}

struct AnalysisResult: Codable, Equatable {
    let performance_id: String
    let dimensions: PerformanceDimensions
    let models: [ModelResult]
    let teacher_feedback: CitedFeedback
    let practice_tips: [PracticeTip]
}

/// Response from the /api/upload endpoint.
struct UploadedPerformance: Codable, Equatable {
    let id: String
    let audio_url: String
    let r2_key: String
    let title: String
    let file_size_bytes: Int
    let content_type: String
}
