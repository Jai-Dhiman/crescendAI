import Foundation

// MARK: - User Models

struct User: Codable {
    let id: String
    let email: String
    let name: String
    let avatar: String?
    let createdAt: String
    let updatedAt: String
}

// MARK: - Recording Models

struct Recording: Codable {
    let id: String
    let userId: String
    let title: String
    let description: String?
    let audioUrl: String
    let localPath: String?
    let duration: Double // in seconds
    let createdAt: String
    let updatedAt: String
    let status: RecordingStatus
    let analysisId: String?
}

enum RecordingStatus: String, Codable, CaseIterable {
    case recorded
    case uploading
    case processing
    case analyzed
    case error
}

// MARK: - Analysis Models

struct Analysis: Codable {
    let id: String
    let recordingId: String
    let status: AnalysisStatus
    let results: AnalysisResults?
    let createdAt: String
    let updatedAt: String
    let error: String?
}

enum AnalysisStatus: String, Codable, CaseIterable {
    case pending
    case processing
    case completed
    case failed
}

struct AnalysisResults: Codable {
    let overallScore: Double // 0-100
    let timing: TimingAnalysis
    let pitch: PitchAnalysis
    let dynamics: DynamicsAnalysis
    let technique: TechniqueAnalysis
    let recommendations: [String]
    let midiFile: String?
    let visualizations: Visualizations
}

struct TimingAnalysis: Codable {
    let accuracy: Double // 0-100
    let consistency: Double // 0-100
    let tempo: Double // BPM
    let tempoVariations: [TempoVariation]
}

struct TempoVariation: Codable {
    let timestamp: Double // seconds
    let tempo: Double // BPM
}

struct PitchAnalysis: Codable {
    let accuracy: Double // 0-100
    let intonation: Double // 0-100
    let noteErrors: [NoteError]
}

struct NoteError: Codable {
    let timestamp: Double // seconds
    let expectedNote: String
    let playedNote: String
    let severity: NoteSeverity
}

enum NoteSeverity: String, Codable, CaseIterable {
    case minor
    case major
}

struct DynamicsAnalysis: Codable {
    let range: Double // 0-100
    let control: Double // 0-100
    let expression: Double // 0-100
}

struct TechniqueAnalysis: Codable {
    let fingerwork: Double // 0-100
    let articulation: Double // 0-100
    let phrasing: Double // 0-100
}

struct Visualizations: Codable {
    let spectrogram: String? // URL to spectrogram image
    let pianoRoll: String? // URL to piano roll visualization
    let waveform: String? // URL to waveform image
}

// MARK: - Practice Session Models

struct PracticeSession: Codable {
    let id: String
    let userId: String
    let recordings: [Recording]
    let startTime: String
    let endTime: String?
    let totalDuration: Double // in seconds
    let notes: String?
}

// MARK: - Progress Models

struct Progress: Codable {
    let userId: String
    let totalPracticeTime: Double // in seconds
    let totalRecordings: Int
    let averageScore: Double
    let skillProgression: SkillProgression
    let weeklyStats: [WeeklyStats]
    let lastUpdated: String
}

struct SkillProgression: Codable {
    let timing: Double // 0-100
    let pitch: Double // 0-100
    let dynamics: Double // 0-100
    let technique: Double // 0-100
}

struct WeeklyStats: Codable {
    let week: String // ISO week format
    let practiceTime: Double // in seconds
    let recordingCount: Int
    let averageScore: Double
}

// MARK: - API Response Models

struct APIResponse<T: Codable>: Codable {
    let data: T
    let message: String?
    let error: String?
}

struct PaginatedResponse<T: Codable>: Codable {
    let data: [T]
    let pagination: PaginationInfo
}

struct PaginationInfo: Codable {
    let page: Int
    let limit: Int
    let total: Int
    let hasMore: Bool
}

// MARK: - Authentication Models

struct AuthTokens: Codable {
    let accessToken: String
    let refreshToken: String
    let expiresAt: String
}

struct AuthState {
    var user: User?
    var tokens: AuthTokens?
    var isAuthenticated: Bool
    var isLoading: Bool
    
    init() {
        self.user = nil
        self.tokens = nil
        self.isAuthenticated = false
        self.isLoading = false
    }
}

// MARK: - Upload Models

struct UploadProgress {
    let recordingId: String
    let progress: Double // 0-100
    let status: UploadStatus
    let error: String?
}

enum UploadStatus: String, CaseIterable {
    case uploading
    case completed
    case error
}

// MARK: - Settings Models

struct UserSettings: Codable {
    let notifications: NotificationSettings
    let audio: AudioSettings
    let privacy: PrivacySettings
}

struct NotificationSettings: Codable {
    let practiceReminders: Bool
    let analysisComplete: Bool
    let weeklyReports: Bool
}

struct AudioSettings: Codable {
    let sampleRate: Int
    let bitRate: Int
    let format: AudioFormat
}

enum AudioFormat: String, Codable, CaseIterable {
    case wav
    case mp3
    case m4a
}

struct PrivacySettings: Codable {
    let shareProgress: Bool
    let allowAnalytics: Bool
}

// MARK: - Error Models

struct AppError: Error {
    let code: String
    let message: String
    let details: Any?
    
    init(code: String, message: String, details: Any? = nil) {
        self.code = code
        self.message = message
        self.details = details
    }
}
