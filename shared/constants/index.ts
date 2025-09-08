// Audio Processing Constants
export const AUDIO_CONSTRAINTS = {
  MIN_SAMPLE_RATE: 44100,
  MIN_BIT_DEPTH: 16,
  MAX_FILE_SIZE_MB: 50,
  MIN_DURATION_SEC: 30,
  MAX_DURATION_SEC: 180, // 3 minutes
  SUPPORTED_FORMATS: ['wav', 'mp3'] as const,
} as const;

// API Endpoints
export const API_ENDPOINTS = {
  UPLOAD: '/api/upload',
  ANALYZE: '/api/analyze',
  STATUS: '/api/status',
  RESULTS: '/api/results',
} as const;

// Processing Configuration
export const PROCESSING_CONFIG = {
  CHUNK_SIZE_SEC: 30,
  MEL_SPECTROGRAM: {
    N_MELS: 128,
    HOP_LENGTH: 512,
    N_FFT: 2048,
    WINDOW: 'hann',
  },
  PATCH_SIZE: 16,
} as const;

// Dimension Labels for UI
export const DIMENSION_LABELS = {
  // Technical
  timingStability: 'Timing Stability',
  articulationLength: 'Articulation Length',
  articulationTouch: 'Articulation Touch',
  pedalUsage: 'Pedal Usage',
  pedalClarity: 'Pedal Clarity',
  
  // Tonal
  timbreColorVariation: 'Timbre Color Variation',
  timbreRichness: 'Timbre Richness',
  timbreBrightness: 'Timbre Brightness',
  dynamicVolume: 'Dynamic Volume',
  
  // Expression
  dynamicSophistication: 'Dynamic Sophistication',
  dynamicRange: 'Dynamic Range',
  musicalPacing: 'Musical Pacing',
  spatialQuality: 'Spatial Quality',
  musicalBalance: 'Musical Balance',
  
  // Interpretive
  expressionLevel: 'Expression Level',
  emotionalValence: 'Emotional Valence',
  energyLevel: 'Energy Level',
  interpretiveAuthenticity: 'Interpretive Authenticity',
  overallConvincingness: 'Overall Convincingness',
} as const;

export const DIMENSION_CATEGORIES = {
  technical: [
    'timingStability',
    'articulationLength', 
    'articulationTouch',
    'pedalUsage',
    'pedalClarity'
  ],
  tonal: [
    'timbreColorVariation',
    'timbreRichness',
    'timbreBrightness', 
    'dynamicVolume'
  ],
  expression: [
    'dynamicSophistication',
    'dynamicRange',
    'musicalPacing',
    'spatialQuality',
    'musicalBalance'
  ],
  interpretive: [
    'expressionLevel',
    'emotionalValence',
    'energyLevel',
    'interpretiveAuthenticity',
    'overallConvincingness'
  ]
} as const;