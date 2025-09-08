// User types
export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  createdAt: string;
  updatedAt: string;
}

// Recording types
export interface Recording {
  id: string;
  userId: string;
  title: string;
  description?: string;
  audioUrl: string;
  localPath?: string;
  duration: number; // in seconds
  createdAt: string;
  updatedAt: string;
  status: RecordingStatus;
  analysisId?: string;
}

export type RecordingStatus = 'recorded' | 'uploading' | 'processing' | 'analyzed' | 'error';

// Analysis types
export interface Analysis {
  id: string;
  recordingId: string;
  status: AnalysisStatus;
  results?: AnalysisResults;
  createdAt: string;
  updatedAt: string;
  error?: string;
}

export type AnalysisStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface AnalysisResults {
  overallScore: number; // 0-100
  timing: TimingAnalysis;
  pitch: PitchAnalysis;
  dynamics: DynamicsAnalysis;
  technique: TechniqueAnalysis;
  recommendations: string[];
  midiFile?: string;
  visualizations: Visualizations;
}

export interface TimingAnalysis {
  accuracy: number; // 0-100
  consistency: number; // 0-100
  tempo: number; // BPM
  tempoVariations: TempoVariation[];
}

export interface TempoVariation {
  timestamp: number; // seconds
  tempo: number; // BPM
}

export interface PitchAnalysis {
  accuracy: number; // 0-100
  intonation: number; // 0-100
  noteErrors: NoteError[];
}

export interface NoteError {
  timestamp: number; // seconds
  expectedNote: string;
  playedNote: string;
  severity: 'minor' | 'major';
}

export interface DynamicsAnalysis {
  range: number; // 0-100
  control: number; // 0-100
  expression: number; // 0-100
}

export interface TechniqueAnalysis {
  fingerwork: number; // 0-100
  articulation: number; // 0-100
  phrasing: number; // 0-100
}

export interface Visualizations {
  spectrogram?: string; // URL to spectrogram image
  pianoRoll?: string; // URL to piano roll visualization
  waveform?: string; // URL to waveform image
}

// Practice session types
export interface PracticeSession {
  id: string;
  userId: string;
  recordings: Recording[];
  startTime: string;
  endTime?: string;
  totalDuration: number; // in seconds
  notes?: string;
}

// Progress tracking
export interface Progress {
  userId: string;
  totalPracticeTime: number; // in seconds
  totalRecordings: number;
  averageScore: number;
  skillProgression: SkillProgression;
  weeklyStats: WeeklyStats[];
  lastUpdated: string;
}

export interface SkillProgression {
  timing: number; // 0-100
  pitch: number; // 0-100
  dynamics: number; // 0-100
  technique: number; // 0-100
}

export interface WeeklyStats {
  week: string; // ISO week format
  practiceTime: number; // in seconds
  recordingCount: number;
  averageScore: number;
}

// API types
export interface ApiResponse<T> {
  data: T;
  message?: string;
  error?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    hasMore: boolean;
  };
}

// Auth types
export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresAt: string;
}

export interface AuthState {
  user: User | null;
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

// Upload types
export interface UploadProgress {
  recordingId: string;
  progress: number; // 0-100
  status: 'uploading' | 'completed' | 'error';
  error?: string;
}

// Settings types
export interface UserSettings {
  notifications: NotificationSettings;
  audio: AudioSettings;
  privacy: PrivacySettings;
}

export interface NotificationSettings {
  practiceReminders: boolean;
  analysisComplete: boolean;
  weeklyReports: boolean;
}

export interface AudioSettings {
  sampleRate: number;
  bitRate: number;
  format: 'wav' | 'mp3' | 'm4a';
}

export interface PrivacySettings {
  shareProgress: boolean;
  allowAnalytics: boolean;
}

// Error types
export interface AppError {
  code: string;
  message: string;
  details?: any;
}

// Navigation types
export type RootStackParamList = {
  index: undefined;
  auth: undefined;
  '(tabs)': undefined;
};

export type TabParamList = {
  record: undefined;
  recordings: undefined;
  progress: undefined;
  profile: undefined;
};