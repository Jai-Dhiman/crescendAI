# 🎹 CrescendAI

## AI-powered piano performance analysis using Audio Spectrogram Transformer technology

## Overview

CrescendAI is a platform that uses machine learning to analyze piano performances across 16 different dimensions. The platform uses Audio Spectrogram Transformer (AST) models with  to deliver comprehensive analysis of classical piano recordings. The system provides detailed insights into technical performance, tonal quality, musical expression, and interpretive qualities.

## 🚀 Key Features

### **16-Dimensional Performance Analysis**

- **Technical Metrics**: Timing stability, articulation control, pedal technique
- **Tonal Quality**: Timbre variation, harmonic richness, dynamic projection  
- **Musical Expression**: Phrasing, tempo flexibility, spatial qualities
- **Interpretive Analysis**: Emotional content, authenticity, overall convincingness

### **Advanced Technology Stack**

- Custom 86M parameter Audio Spectrogram Transformer model
- High-performance Rust backend with Cloudflare Workers
- React Native cross-platform mobile application
- GPU-optimized inference with Modal hosting
- Global edge processing for <100ms response times

### **Professional Features**

- High-quality audio recording (44.1kHz, 16-bit)
- Real-time upload and processing status
- Comprehensive visualization with radar charts
- Historical performance comparison
- Raw data export for detailed analysis

## 🏗️ Architecture

```jsonc
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  React Native   │────│ Cloudflare       │────│  Modal GPU      │
│  Mobile App     │    │ Workers (Rust)   │    │  Inference      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌──────────────────┐             │
         └──────────────│ Cloudflare R2    │─────────────┘
                        │ Storage          │
                        └──────────────────┘
```

### **Core Components**

- **Mobile Frontend**: React Native app with professional audio recording capabilities
- **Edge Backend**: Rust-powered Cloudflare Workers for high-performance preprocessing
- **ML Infrastructure**: Custom AST model deployed on Modal GPU infrastructure
- **Storage Layer**: Cloudflare R2 for global audio storage and results caching

### **Data Flow**

1. User records 30s-3min piano performance via mobile app
2. Audio uploaded to Cloudflare R2 with real-time progress tracking
3. Rust backend preprocesses audio into mel-spectrograms
4. Modal GPU infrastructure runs AST model inference (2-5 seconds)
5. Results aggregated and cached for instant retrieval
6. 19-dimensional analysis displayed in mobile interface

## 🛠️ Setup & Installation

### **Prerequisites**

- Python 3.9+ for model training/development
- Node.js 18+ for mobile development
- Rust 1.70+ for backend development
- Expo CLI for React Native deployment

### **Installation**

```bash
# Clone the repository
git clone https://github.com/Jai-Dhiman/crescendai.git
cd crescendai

# Install Python dependencies (using uv)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Setup mobile development
cd mobile
npm install

# Setup backend (Rust)
cd ../backend
cargo build --target wasm32-unknown-unknown --release

# Setup model training environment
cd ../model
uv pip install -r requirements.txt
```

### **Configuration**

1. Copy `.env.example` to `.env` and configure:
   - Cloudflare Worker credentials
   - Modal API keys
   - R2 storage configuration

2. Initialize the model:

   ```bash
   python model/colab_setup.py
   ```

3. Start development servers:

   ```bash
   # Mobile app
   cd mobile && npm run dev
   
   # Backend (if developing locally)
   cd backend && cargo test
   ```

## 📱 Usage

### **Recording Analysis**

1. Open the mobile app
2. Tap "Record" to capture a piano performance (30s-3min)
3. Upload automatically begins with progress tracking
4. View comprehensive analysis results within 10-15 seconds

### **Interpreting Results**

- **Radar Chart**: Visual overview of all 16 performance dimensions
- **Temporal Analysis**: Performance quality changes over time
- **Historical Comparison**: Track improvement across multiple recordings
- **Raw Scores**: Detailed numerical data for technical analysis

### **Example Analysis Output**

```json
{
  "technical_performance": {
    "timing_stability": 8.7,
    "articulation_length": 7.9,
    "articulation_touch": 8.2,
    "pedal_usage": 7.4,
    "pedal_clarity": 8.1
  },
  "tonal_quality": {
    "timbre_color_variation": 8.8,
    "timbre_richness": 9.1,
    "timbre_brightness": 7.6,
    "dynamic_volume": 8.4
  }
  // ... additional dimensions
}
```

## 🔧 Technical Specifications

### **Performance**

- **Analysis Time**: 10-15 seconds for 3-minute recording
- **Model Inference**: 2-5 seconds GPU processing
- **Global Latency**: <100ms upload initiation worldwide
- **Throughput**: 20+ concurrent analyses supported

### **Audio Quality**

- **Sample Rate**: 44.1 kHz minimum
- **Bit Depth**: 16-bit minimum
- **Supported Formats**: WAV, MP3
- **File Size Limit**: 50MB per recording

### **System Requirements**

- **Mobile**: iOS 12+, Android 8+ (API level 26+)
- **Storage**: 30-day retention with automatic cleanup
- **Network**: Optimized for 3G+ connections

## 🧪 Research & Development

This platform serves as a research demonstration showcasing:

- **Advanced ML Deployment**: Production-ready Audio Spectrogram Transformer models
- **High-Performance Systems**: Rust-powered edge computing with global distribution
- **Mobile-First Design**: Cross-platform audio processing and real-time feedback
- **Scalable Architecture**: Cost-efficient design scaling from research to production

### **Model Training**

The AST model is trained on diverse piano performance data with expert annotations across all 19 dimensions. Training code and experimentation notebooks are available in the `model/` directory.

### **Performance Optimization**

- GPU inference batching for cost efficiency
- Edge preprocessing to minimize latency
- Adaptive quality settings for network conditions
- Intelligent caching for repeat analysis

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
