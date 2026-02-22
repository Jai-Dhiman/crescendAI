import Foundation

/// 19-dimensional piano performance evaluation.
/// Mirrors the Rust `PerformanceDimensions` struct from the web backend.
struct PerformanceDimensions: Codable, Equatable {
    // Timing
    let timing: Double

    // Articulation
    let articulation_length: Double
    let articulation_touch: Double

    // Pedal
    let pedal_amount: Double
    let pedal_clarity: Double

    // Timbre
    let timbre_variety: Double
    let timbre_depth: Double
    let timbre_brightness: Double
    let timbre_loudness: Double

    // Dynamics
    let dynamics_range: Double

    // Performance qualities
    let tempo: Double
    let space: Double
    let balance: Double
    let drama: Double

    // Mood
    let mood_valence: Double
    let mood_energy: Double
    let mood_imagination: Double

    // Interpretation
    let interpretation_sophistication: Double
    let interpretation_overall: Double

    func toLabeledPairs() -> [(String, Double)] {
        [
            ("Timing", timing),
            ("Art. Length", articulation_length),
            ("Art. Touch", articulation_touch),
            ("Pedal Amt", pedal_amount),
            ("Pedal Clarity", pedal_clarity),
            ("Timbre Var.", timbre_variety),
            ("Timbre Depth", timbre_depth),
            ("Brightness", timbre_brightness),
            ("Loudness", timbre_loudness),
            ("Dyn. Range", dynamics_range),
            ("Tempo", tempo),
            ("Space", space),
            ("Balance", balance),
            ("Drama", drama),
            ("Valence", mood_valence),
            ("Energy", mood_energy),
            ("Imagination", mood_imagination),
            ("Sophistication", interpretation_sophistication),
            ("Interpretation", interpretation_overall),
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
    let r_squared: Double
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
    let calibrated_dimensions: PerformanceDimensions
    let calibration_context: String?
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
