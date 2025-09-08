// Piano Performance Analysis Types

export interface PerceptualDimensions {
  // Technical Performance Metrics (5)
  timingStability: number;
  articulationLength: number;
  articulationTouch: number;
  pedalUsage: number;
  pedalClarity: number;
  
  // Tonal Quality Metrics (4)
  timbreColorVariation: number;
  timbreRichness: number;
  timbreBrightness: number;
  dynamicVolume: number;
  
  // Musical Expression Metrics (5)
  dynamicSophistication: number;
  dynamicRange: number;
  musicalPacing: number;
  spatialQuality: number;
  musicalBalance: number;
  
  // Interpretive Qualities (5)
  expressionLevel: number;
  emotionalValence: number;
  energyLevel: number;
  interpretiveAuthenticity: number;
  overallConvincingness: number;
}

export interface AudioMetadata {
  filename: string;
  duration: number;
  sampleRate: number;
  bitDepth: number;
  fileSize: number;
  format: 'wav' | 'mp3';
  uploadedAt: Date;
  pieceName?: string;
  notes?: string;
}

export interface AnalysisResult {
  id: string;
  dimensions: PerceptualDimensions;
  metadata: AudioMetadata;
  processingTime: number;
  status: 'processing' | 'completed' | 'failed';
  error?: string;
  createdAt: Date;
}

export interface AnalysisRequest {
  audioUrl: string;
  metadata: Pick<AudioMetadata, 'filename' | 'duration' | 'sampleRate' | 'bitDepth' | 'fileSize' | 'format' | 'pieceName' | 'notes'>;
}

export interface ProcessingStatus {
  id: string;
  status: 'uploaded' | 'preprocessing' | 'inferring' | 'completed' | 'failed';
  progress: number; // 0-100
  estimatedTimeRemaining?: number; // seconds
  error?: string;
}