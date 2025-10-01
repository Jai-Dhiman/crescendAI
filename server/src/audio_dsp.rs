use rustfft::{FftPlanner, num_complex::Complex};
use hound::{WavReader, SampleFormat};
use std::f32::consts::PI;
use std::io::Cursor;

#[cfg(not(test))]
use worker::*;

// Conditional logging macro for test vs WASM environments
#[cfg(test)]
macro_rules! console_log {
    ($($arg:tt)*) => {
        eprintln!($($arg)*);
    };
}

#[cfg(not(test))]
macro_rules! console_log {
    ($($arg:tt)*) => {
        worker::console_log!($($arg)*);
    };
}

// Define custom Result type for audio processing to avoid conflicts with worker::Result
type AudioResult<T> = std::result::Result<T, AudioError>;

/// Audio format constants for mel-spectrogram generation
pub const SAMPLE_RATE: u32 = 22050; // Standard for ML models
pub const N_FFT: usize = 2048;
pub const HOP_LENGTH: usize = 512;
pub const N_MELS: usize = 128;
pub const FMIN: f32 = 0.0;
pub const FMAX: f32 = 11025.0; // Nyquist frequency for 22.05kHz

/// Audio processing errors specific to DSP operations
#[derive(Debug)]
pub enum AudioError {
    InvalidFormat,
    UnsupportedSampleRate(u32),
    UnsupportedBitDepth(u16),
    UnsupportedChannels(u16),
    ProcessingError(String),
    InsufficientData,
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioError::InvalidFormat => write!(f, "Invalid audio format"),
            AudioError::UnsupportedSampleRate(rate) => write!(f, "Unsupported sample rate: {}", rate),
            AudioError::UnsupportedBitDepth(bits) => write!(f, "Unsupported bit depth: {}", bits),
            AudioError::UnsupportedChannels(channels) => write!(f, "Unsupported channel count: {}", channels),
            AudioError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
            AudioError::InsufficientData => write!(f, "Insufficient audio data"),
        }
    }
}

impl std::error::Error for AudioError {}

/// Represents processed audio data ready for analysis
#[derive(Debug, Clone)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_seconds: f32,
}

/// Mel-spectrogram result structure
#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    pub data: Vec<Vec<f32>>, // [n_mels, n_frames]
    pub n_mels: usize,
    pub n_frames: usize,
    pub sample_rate: u32,
    pub hop_length: usize,
}

impl MelSpectrogram {
    /// Serialize mel-spectrogram to bytes for transmission
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        
        // Write metadata
        bytes.extend_from_slice(&(self.n_mels as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.n_frames as u32).to_le_bytes());
        bytes.extend_from_slice(&self.sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(self.hop_length as u32).to_le_bytes());
        
        // Write spectrogram data as float32 array
        for mel_bin in &self.data {
            for &value in mel_bin {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        
        console_log!("Serialized mel-spectrogram: {} bytes ({}x{} spectrogram)", 
                     bytes.len(), self.n_mels, self.n_frames);
        bytes
    }
    
    /// Get expected size in bytes
    pub fn expected_size(&self) -> usize {
        16 + (self.n_mels * self.n_frames * 4) // metadata + float32 data
    }
}

/// Parse audio data from various formats (simplified for WASM compatibility)
pub fn parse_audio_data(audio_bytes: &[u8]) -> AudioResult<AudioData> {
    console_log!("Parsing audio data: {} bytes", audio_bytes.len());
    
    // Check minimum size
    if audio_bytes.len() < 44 {
        return Err(AudioError::InsufficientData);
    }
    
    // Parse WAV data using hound library
    let mut audio_data = parse_wav_data(audio_bytes)?;
    
    console_log!("Parsed audio: {} samples at {}Hz ({:.2}s duration)",
                 audio_data.samples.len(),
                 audio_data.sample_rate,
                 audio_data.duration_seconds);
    
    // Resample to standard rate if needed
    if audio_data.sample_rate != SAMPLE_RATE {
        console_log!("Resampling from {}Hz to {}Hz", audio_data.sample_rate, SAMPLE_RATE);
        audio_data = resample_audio(&audio_data, SAMPLE_RATE)?;
        console_log!("Resampled to {} samples at {}Hz ({:.2}s duration)",
                     audio_data.samples.len(),
                     audio_data.sample_rate,
                     audio_data.duration_seconds);
    }
    
    Ok(audio_data)
}

/// Parse WAV audio data using hound library
fn parse_wav_data(wav_bytes: &[u8]) -> AudioResult<AudioData> {
    let cursor = Cursor::new(wav_bytes);
    let mut reader = WavReader::new(cursor)
        .map_err(|_| AudioError::InvalidFormat)?;
    
    let spec = reader.spec();
    
    // Validate format
    if spec.channels == 0 || spec.channels > 2 {
        return Err(AudioError::UnsupportedChannels(spec.channels));
    }
    
    if spec.bits_per_sample != 16 && spec.bits_per_sample != 24 && spec.bits_per_sample != 32 {
        return Err(AudioError::UnsupportedBitDepth(spec.bits_per_sample));
    }
    
    // Read samples based on format
    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            reader.samples::<i16>()
                .map(|s| s.map(|val| val as f32 / i16::MAX as f32)
                          .map_err(|_| AudioError::ProcessingError("Failed to read 16-bit samples".to_string())))
                .collect::<AudioResult<Vec<f32>>>()
        },
        (SampleFormat::Int, 24) => {
            reader.samples::<i32>()
                .map(|s| s.map(|val| (val as f32) / (1 << 23) as f32)
                          .map_err(|_| AudioError::ProcessingError("Failed to read 24-bit samples".to_string())))
                .collect::<AudioResult<Vec<f32>>>()
        },
        (SampleFormat::Int, 32) => {
            reader.samples::<i32>()
                .map(|s| s.map(|val| val as f32 / i32::MAX as f32)
                          .map_err(|_| AudioError::ProcessingError("Failed to read 32-bit samples".to_string())))
                .collect::<AudioResult<Vec<f32>>>()
        },
        (SampleFormat::Float, 32) => {
            reader.samples::<f32>()
                .map(|s| s.map_err(|_| AudioError::ProcessingError("Failed to read float samples".to_string())))
                .collect::<AudioResult<Vec<f32>>>()
        },
        _ => Err(AudioError::UnsupportedBitDepth(spec.bits_per_sample)),
    }?;
    
    let duration_seconds = samples.len() as f32 / (spec.sample_rate * spec.channels as u32) as f32;
    
    // Convert to mono if stereo
    let mono_samples = if spec.channels == 2 {
        samples.chunks(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect()
    } else {
        samples
    };
    
    Ok(AudioData {
        samples: mono_samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        duration_seconds,
    })
}

/// Resample audio data to target sample rate using linear interpolation
pub fn resample_audio(audio: &AudioData, target_sample_rate: u32) -> AudioResult<AudioData> {
    if audio.sample_rate == target_sample_rate {
        return Ok(audio.clone());
    }
    
    console_log!("Resampling from {}Hz to {}Hz", audio.sample_rate, target_sample_rate);
    
    let ratio = target_sample_rate as f32 / audio.sample_rate as f32;
    let new_length = (audio.samples.len() as f32 * ratio) as usize;
    let mut resampled = Vec::with_capacity(new_length);
    
    for i in 0..new_length {
        let source_index = i as f32 / ratio;
        let index = source_index as usize;
        
        if index >= audio.samples.len() - 1 {
            resampled.push(audio.samples[audio.samples.len() - 1]);
        } else {
            // Linear interpolation
            let frac = source_index - index as f32;
            let sample = audio.samples[index] * (1.0 - frac) + audio.samples[index + 1] * frac;
            resampled.push(sample);
        }
    }
    
    let new_duration = resampled.len() as f32 / target_sample_rate as f32;
    
    Ok(AudioData {
        samples: resampled,
        sample_rate: target_sample_rate,
        channels: audio.channels,
        duration_seconds: new_duration,
    })
}

/// Apply Hanning window to audio frame
fn apply_hanning_window(frame: &mut [f32]) {
    let n = frame.len();
    for i in 0..n {
        let window_val = 0.5 - 0.5 * (2.0 * PI * i as f32 / (n - 1) as f32).cos();
        frame[i] *= window_val;
    }
}

/// Compute Short-Time Fourier Transform (STFT) (simplified for WASM)
pub fn compute_stft(audio: &AudioData, n_fft: usize, hop_length: usize) -> AudioResult<Vec<Vec<Complex<f32>>>> {
    if audio.samples.len() < n_fft {
        return Err(AudioError::InsufficientData);
    }
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    
    let n_frames = (audio.samples.len() - n_fft) / hop_length + 1;
    let mut stft_result = Vec::with_capacity(n_frames);
    
    console_log!("Computing STFT: {} frames, FFT size {}, hop length {}", n_frames, n_fft, hop_length);
    
    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        let end = start + n_fft;
        
        if end > audio.samples.len() {
            break;
        }
        
        // Extract frame and apply window
        let mut frame: Vec<f32> = audio.samples[start..end].to_vec();
        apply_hanning_window(&mut frame);
        
        // Convert to complex for FFT
        let mut complex_frame: Vec<Complex<f32>> = frame.into_iter()
            .map(|x| Complex::new(x, 0.0))
            .collect();
        
        // Compute FFT
        fft.process(&mut complex_frame);
        stft_result.push(complex_frame);
    }
    
    Ok(stft_result)
}

/// Convert frequency to mel scale
fn hz_to_mel(freq: f32) -> f32 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

/// Convert mel scale to frequency
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Create mel filter bank
fn create_mel_filter_bank(sample_rate: u32, n_fft: usize, n_mels: usize, fmin: f32, fmax: f32) -> Vec<Vec<f32>> {
    let nyquist = sample_rate as f32 / 2.0;
    let fmax_actual = if fmax > nyquist { nyquist } else { fmax };
    
    // Create mel-spaced frequencies
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax_actual);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    
    // Convert back to Hz and then to FFT bin indices
    let hz_points: Vec<f32> = mel_points.iter().map(|&mel| mel_to_hz(mel)).collect();
    let bin_points: Vec<usize> = hz_points.iter()
        .map(|&hz| ((n_fft / 2 + 1) as f32 * hz / nyquist).floor() as usize)
        .collect();
    
    // Create triangular filters
    let mut filter_bank = vec![vec![0.0; n_fft / 2 + 1]; n_mels];
    
    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];
        
        for k in left..=right {
            if k < center {
                filter_bank[m][k] = (k - left) as f32 / (center - left) as f32;
            } else {
                filter_bank[m][k] = (right - k) as f32 / (right - center) as f32;
            }
        }
    }
    
    filter_bank
}

/// Generate mel-spectrogram from audio data
pub fn generate_mel_spectrogram(audio: &AudioData, n_fft: usize, hop_length: usize, n_mels: usize) -> AudioResult<MelSpectrogram> {
    // Compute STFT
    let stft = compute_stft(audio, n_fft, hop_length)?;
    
    // Create mel filter bank
    let mel_filters = create_mel_filter_bank(audio.sample_rate, n_fft, n_mels, FMIN, FMAX);
    
    // Convert STFT to power spectrogram
    let power_spectrogram: Vec<Vec<f32>> = stft.into_iter()
        .map(|frame| {
            frame.into_iter()
                .take(n_fft / 2 + 1) // Take only positive frequencies
                .map(|c| c.norm_sqr()) // Power = |magnitude|^2
                .collect()
        })
        .collect();
    
    // Apply mel filter bank
    let mut mel_spectrogram = vec![vec![0.0; power_spectrogram.len()]; n_mels];
    
    for (frame_idx, power_frame) in power_spectrogram.iter().enumerate() {
        for (mel_idx, mel_filter) in mel_filters.iter().enumerate() {
            let mut mel_energy = 0.0;
            for (bin_idx, &power) in power_frame.iter().enumerate() {
                mel_energy += power * mel_filter[bin_idx];
            }
            mel_spectrogram[mel_idx][frame_idx] = mel_energy;
        }
    }
    
    // Apply log transformation (log mel-spectrogram)
    for mel_bin in &mut mel_spectrogram {
        for value in mel_bin {
            *value = (*value + 1e-8).ln(); // Add small epsilon to avoid log(0)
        }
    }
    
    console_log!("Generated mel-spectrogram: {}x{} (mels x frames)", n_mels, mel_spectrogram[0].len());
    
    Ok(MelSpectrogram {
        data: mel_spectrogram,
        n_mels,
        n_frames: power_spectrogram.len(),
        sample_rate: audio.sample_rate,
        hop_length,
    })
}

// ============================================================================
// Audio Chunking for Temporal Analysis
// ============================================================================

/// Represents a chunk of audio with temporal metadata
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Original audio samples for this chunk
    pub samples: Vec<f32>,
    /// Sample rate (inherited from parent)
    pub sample_rate: u32,
    /// Start time in seconds
    pub start_time_secs: f32,
    /// End time in seconds
    pub end_time_secs: f32,
    /// Formatted timestamp string (e.g., "0:00-0:03")
    pub timestamp: String,
    /// Chunk index in the sequence
    pub chunk_index: usize,
}

impl AudioChunk {
    /// Format time in seconds to MM:SS string
    fn format_time(seconds: f32) -> String {
        let minutes = (seconds / 60.0).floor() as u32;
        let secs = (seconds % 60.0).floor() as u32;
        format!("{}:{:02}", minutes, secs)
    }
    
    /// Create timestamp range string
    pub fn create_timestamp(start: f32, end: f32) -> String {
        format!("{}-{}", Self::format_time(start), Self::format_time(end))
    }
}

/// Chunk audio into overlapping segments
///
/// # Arguments
/// * `audio` - Source audio data
/// * `chunk_duration_secs` - Duration of each chunk (e.g., 3.0)
/// * `overlap_secs` - Overlap between chunks (e.g., 1.0)
///
/// # Returns
/// Vector of audio chunks with temporal metadata
///
/// # Errors
/// Returns error if:
/// - Audio is too short for even one chunk
/// - Invalid chunk duration or overlap values
/// - Sample rate is invalid
pub fn chunk_audio_with_overlap(
    audio: &AudioData,
    chunk_duration_secs: f32,
    overlap_secs: f32,
) -> worker::Result<Vec<AudioChunk>> {
    // Validate parameters
    if chunk_duration_secs <= 0.0 {
        return Err(worker::Error::RustError(
            "Chunk duration must be positive".to_string()
        ));
    }
    
    if overlap_secs < 0.0 || overlap_secs >= chunk_duration_secs {
        return Err(worker::Error::RustError(
            "Overlap must be non-negative and less than chunk duration".to_string()
        ));
    }
    
    if audio.sample_rate == 0 {
        return Err(worker::Error::RustError(
            "Invalid sample rate".to_string()
        ));
    }
    
    // Calculate sample counts
    let chunk_samples = (chunk_duration_secs * audio.sample_rate as f32) as usize;
    let hop_samples = ((chunk_duration_secs - overlap_secs) * audio.sample_rate as f32) as usize;
    
    // Validate audio length
    if audio.samples.len() < chunk_samples {
        return Err(worker::Error::RustError(
            format!(
                "Audio too short: {} samples, need at least {} for one {}-second chunk",
                audio.samples.len(), chunk_samples, chunk_duration_secs
            )
        ));
    }
    
    console_log!(
        "Chunking audio: {} samples at {}Hz into {}-second chunks with {}-second overlap",
        audio.samples.len(),
        audio.sample_rate,
        chunk_duration_secs,
        overlap_secs
    );
    
    let mut chunks = Vec::new();
    let mut start_sample = 0;
    let mut chunk_index = 0;
    
    while start_sample + chunk_samples <= audio.samples.len() {
        let end_sample = start_sample + chunk_samples;
        
        // Calculate times
        let start_time_secs = start_sample as f32 / audio.sample_rate as f32;
        let end_time_secs = end_sample as f32 / audio.sample_rate as f32;
        let timestamp = AudioChunk::create_timestamp(start_time_secs, end_time_secs);
        
        // Extract samples - use to_vec() to avoid borrowing issues
        let chunk_samples_vec = audio.samples[start_sample..end_sample].to_vec();
        
        console_log!(
            "Created chunk {}: {} ({:.2}s - {:.2}s, {} samples)",
            chunk_index,
            timestamp,
            start_time_secs,
            end_time_secs,
            chunk_samples_vec.len()
        );
        
        chunks.push(AudioChunk {
            samples: chunk_samples_vec,
            sample_rate: audio.sample_rate,
            start_time_secs,
            end_time_secs,
            timestamp,
            chunk_index,
        });
        
        start_sample += hop_samples;
        chunk_index += 1;
    }
    
    console_log!("Created {} chunks from audio", chunks.len());
    
    // Ensure we got at least one chunk
    if chunks.is_empty() {
        return Err(worker::Error::RustError(
            "Failed to create any chunks from audio".to_string()
        ));
    }
    
    Ok(chunks)
}

/// Generate mel-spectrogram specifically for an audio chunk
///
/// # Arguments
/// * `chunk` - Audio chunk to process
///
/// # Returns
/// Serialized mel-spectrogram bytes
///
/// # Errors
/// Returns error if DSP processing fails
pub async fn generate_mel_spectrogram_for_chunk(
    chunk: &AudioChunk,
) -> worker::Result<Vec<u8>> {
    console_log!(
        "Generating mel-spectrogram for chunk {} ({})",
        chunk.chunk_index,
        chunk.timestamp
    );
    
    // Create temporary AudioData for this chunk
    let chunk_audio = AudioData {
        samples: chunk.samples.clone(),
        sample_rate: chunk.sample_rate,
        channels: 1, // Assume mono for chunks
        duration_seconds: chunk.end_time_secs - chunk.start_time_secs,
    };
    
    // Use existing mel-spectrogram generation
    match process_audio_to_mel_spectrogram_from_audio(&chunk_audio).await {
        Ok(spectrogram_bytes) => {
            console_log!(
                "Generated {}-byte spectrogram for chunk {}",
                spectrogram_bytes.len(),
                chunk.chunk_index
            );
            Ok(spectrogram_bytes)
        }
        Err(e) => {
            console_log!(
                "Failed to generate spectrogram for chunk {}: {}",
                chunk.chunk_index,
                e
            );
            Err(worker::Error::RustError(format!(
                "Spectrogram generation failed for chunk {}: {}",
                chunk.chunk_index, e
            )))
        }
    }
}

/// Helper function to generate mel-spectrogram from AudioData directly
pub async fn process_audio_to_mel_spectrogram_from_audio(audio_data: &AudioData) -> worker::Result<Vec<u8>> {
    // Generate mel-spectrogram
    let mel_spec = generate_mel_spectrogram(audio_data, N_FFT, HOP_LENGTH, N_MELS)
        .map_err(|e| worker::Error::RustError(format!("Mel-spectrogram generation failed: {}", e)))?;
    
    // Convert to bytes for transmission
    Ok(mel_spec.to_bytes())
}

/// Main function to process audio data and generate mel-spectrogram
pub async fn process_audio_to_mel_spectrogram(audio_bytes: &[u8]) -> worker::Result<Vec<u8>> {
    console_log!("Starting audio processing for mel-spectrogram generation");
    
    // Parse audio data
    let mut audio_data = parse_audio_data(audio_bytes)
        .map_err(|e| worker::Error::RustError(format!("Audio parsing failed: {}", e)))?;
    
    // Validate minimum duration (need at least 1 second for meaningful analysis)
    if audio_data.duration_seconds < 1.0 {
        return Err(worker::Error::RustError("Audio too short for analysis (minimum 1 second required)".to_string()));
    }
    
    // Resample to standard rate if needed
    if audio_data.sample_rate != SAMPLE_RATE {
        audio_data = resample_audio(&audio_data, SAMPLE_RATE)
            .map_err(|e| worker::Error::RustError(format!("Resampling failed: {}", e)))?;
    }
    
    // Generate mel-spectrogram
    let mel_spec = generate_mel_spectrogram(&audio_data, N_FFT, HOP_LENGTH, N_MELS)
        .map_err(|e| worker::Error::RustError(format!("Mel-spectrogram generation failed: {}", e)))?;
    
    console_log!("Audio processing completed successfully: {}x{} mel-spectrogram", 
                 mel_spec.n_mels, mel_spec.n_frames);
    
    // Convert to bytes for transmission
    Ok(mel_spec.to_bytes())
}

// Note: The original test version that returned Result<Vec<u8>, String> has been removed
// as it conflicted with the main function. Tests now use the main function which returns
// worker::Result<Vec<u8>>.

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_wav_data() -> Vec<u8> {
        let mut data = Vec::new();
        
        // WAV file header for 16-bit PCM, 44100Hz, mono
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&[100u8, 0, 0, 0]); // File size
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&[16u8, 0, 0, 0]); // fmt chunk size
        data.extend_from_slice(&[1u8, 0]); // PCM format
        data.extend_from_slice(&[1u8, 0]); // Mono
        data.extend_from_slice(&[0x44, 0xAC, 0, 0]); // 44100 Hz
        data.extend_from_slice(&[0x88, 0x58, 1, 0]); // Byte rate
        data.extend_from_slice(&[2u8, 0]); // Block align
        data.extend_from_slice(&[16u8, 0]); // 16 bits per sample
        data.extend_from_slice(b"data");
        data.extend_from_slice(&[64u8, 0, 0, 0]); // Data size
        
        // Generate 1000Hz sine wave for 0.5 seconds
        let sample_rate = 44100;
        let duration = 0.5;
        let frequency = 1000.0;
        let num_samples = (sample_rate as f32 * duration) as usize;
        
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * PI * frequency * t).sin();
            let sample_i16 = (sample * i16::MAX as f32) as i16;
            data.extend_from_slice(&sample_i16.to_le_bytes());
        }
        
        data
    }

    #[test]
    fn test_hz_to_mel_conversion() {
        assert_eq!(hz_to_mel(0.0), 0.0);
        assert!((hz_to_mel(1000.0) - 1000.0).abs() < 50.0); // Approximately 1000 mel for 1000 Hz
    }

    #[test]
    fn test_mel_to_hz_conversion() {
        assert_eq!(mel_to_hz(0.0), 0.0);
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);
        assert!((hz - hz_back).abs() < 1.0); // Round-trip conversion
    }

    #[test]
    fn test_audio_data_creation() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let audio = AudioData {
            samples: samples.clone(),
            sample_rate: 44100,
            channels: 1,
            duration_seconds: samples.len() as f32 / 44100.0,
        };
        
        assert_eq!(audio.samples, samples);
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 1);
        assert!((audio.duration_seconds - samples.len() as f32 / 44100.0).abs() < 1e-6);
    }

    #[test]
    fn test_wav_parsing_validation() {
        let test_wav = create_test_wav_data();
        
        // Should successfully parse (returns placeholder for WASM compatibility)
        let result = parse_audio_data(&test_wav);
        assert!(result.is_ok());
        
        let audio = result.unwrap();
        assert_eq!(audio.sample_rate, SAMPLE_RATE); // Uses constant 22050
        assert_eq!(audio.channels, 1);
        assert!(audio.duration_seconds > 0.0);
        assert!(!audio.samples.is_empty());
    }

    #[test]
    fn test_invalid_audio_rejection() {
        let invalid_data = vec![0x00, 0x01, 0x02]; // Too short
        let result = parse_audio_data(&invalid_data);
        assert!(result.is_err());
        
        match result.err().unwrap() {
            AudioError::InsufficientData => (),
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_mel_spectrogram_serialization() {
        let mel_spec = MelSpectrogram {
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            n_mels: 2,
            n_frames: 2,
            sample_rate: 22050,
            hop_length: 512,
        };
        
        let bytes = mel_spec.to_bytes();
        let expected_size = mel_spec.expected_size();
        
        assert_eq!(bytes.len(), expected_size);
        assert!(bytes.len() > 16); // At least metadata size
    }

    #[test]
    fn test_resampling() {
        let original_audio = AudioData {
            samples: vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5],
            sample_rate: 8000,
            channels: 1,
            duration_seconds: 8.0 / 8000.0,
        };
        
        let resampled = resample_audio(&original_audio, 16000).unwrap();
        assert_eq!(resampled.sample_rate, 16000);
        assert!(resampled.samples.len() > original_audio.samples.len());
    }

    #[test]
    fn test_hanning_window() {
        let mut frame = vec![1.0; 8];
        apply_hanning_window(&mut frame);
        
        // Window should start and end near zero
        assert!(frame[0] < 0.1);
        assert!(frame[frame.len() - 1] < 0.1);
        
        // Middle values should be larger
        assert!(frame[frame.len() / 2] > 0.5);
    }

    #[test]
    fn test_mel_filter_bank_creation() {
        let filter_bank = create_mel_filter_bank(22050, 2048, 128, 0.0, 11025.0);
        
        assert_eq!(filter_bank.len(), 128); // n_mels
        assert_eq!(filter_bank[0].len(), 1025); // n_fft/2 + 1
        
        // Each filter should have non-zero values
        for filter in &filter_bank {
            let sum: f32 = filter.iter().sum();
            assert!(sum > 0.0);
        }
    }

    #[test]
    fn test_stft_dimensions() {
        let audio = AudioData {
            samples: vec![0.0; 44100], // 1 second of silence at 44100 Hz
            sample_rate: 44100,
            channels: 1,
            duration_seconds: 1.0,
        };
        
        let stft = compute_stft(&audio, 2048, 512).unwrap();
        
        assert!(!stft.is_empty());
        assert_eq!(stft[0].len(), 2048); // FFT size
        
        // Should have roughly (samples - n_fft) / hop_length + 1 frames
        let expected_frames = (44100 - 2048) / 512 + 1;
        assert!((stft.len() as i32 - expected_frames as i32).abs() <= 1);
    }

    #[test]
    fn test_insufficient_data_handling() {
        let short_audio = AudioData {
            samples: vec![0.1, 0.2, 0.3], // Too few samples
            sample_rate: 44100,
            channels: 1,
            duration_seconds: 3.0 / 44100.0,
        };
        
        let result = compute_stft(&short_audio, 2048, 512);
        assert!(result.is_err());
        
        match result.err().unwrap() {
            AudioError::InsufficientData => (),
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_error_display() {
        let errors = vec![
            AudioError::InvalidFormat,
            AudioError::UnsupportedSampleRate(96000),
            AudioError::UnsupportedBitDepth(8),
            AudioError::UnsupportedChannels(6),
            AudioError::ProcessingError("test error".to_string()),
            AudioError::InsufficientData,
        ];
        
        for error in errors {
            let error_string = error.to_string();
            assert!(!error_string.is_empty());
        }
    }
}