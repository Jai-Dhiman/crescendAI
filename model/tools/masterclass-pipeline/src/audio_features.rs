use crate::config::{FFT_SIZE, HOP_SIZE, SAMPLE_RATE};
use realfft::RealFftPlanner;
use std::f32::consts::PI;

/// Compute a Hann window of the given size.
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos()))
        .collect()
}

/// Short-Time Fourier Transform.
/// Returns magnitude spectrogram: Vec of frames, each frame is Vec of frequency bins.
/// Number of frequency bins = fft_size / 2 + 1.
pub fn stft(samples: &[f32], fft_size: usize, hop_size: usize) -> Vec<Vec<f32>> {
    let window = hann_window(fft_size);
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);

    let num_frames = if samples.len() >= fft_size {
        (samples.len() - fft_size) / hop_size + 1
    } else {
        0
    };

    let mut spectrogram = Vec::with_capacity(num_frames);
    let mut input = vec![0.0f32; fft_size];
    let mut spectrum = fft.make_output_vec();

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let end = start + fft_size;

        // Apply window
        for (i, sample) in input.iter_mut().enumerate() {
            if start + i < samples.len() && start + i < end {
                *sample = samples[start + i] * window[i];
            } else {
                *sample = 0.0;
            }
        }

        fft.process(&mut input, &mut spectrum)
            .expect("FFT processing failed");

        // Compute magnitude
        let magnitudes: Vec<f32> = spectrum.iter().map(|c| c.norm()).collect();
        spectrogram.push(magnitudes);
    }

    spectrogram
}

/// Compute RMS energy per frame.
pub fn rms_energy(samples: &[f32], frame_size: usize, hop_size: usize) -> Vec<f32> {
    let num_frames = if samples.len() >= frame_size {
        (samples.len() - frame_size) / hop_size + 1
    } else {
        0
    };

    let mut energies = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = i * hop_size;
        let end = (start + frame_size).min(samples.len());
        let sum_sq: f32 = samples[start..end].iter().map(|s| s * s).sum();
        let rms = (sum_sq / (end - start) as f32).sqrt();
        energies.push(rms);
    }
    energies
}

/// Convert RMS to dB scale.
pub fn rms_to_db(rms: &[f32]) -> Vec<f32> {
    let floor = 1e-10f32;
    rms.iter()
        .map(|&r| 20.0 * (r.max(floor)).log10())
        .collect()
}

/// Compute spectral centroid per frame from a magnitude spectrogram.
/// Returns frequency in Hz.
pub fn spectral_centroid(spectrogram: &[Vec<f32>], sample_rate: u32, fft_size: usize) -> Vec<f32> {
    let freq_resolution = sample_rate as f32 / fft_size as f32;

    spectrogram
        .iter()
        .map(|frame| {
            let total_magnitude: f32 = frame.iter().sum();
            if total_magnitude < 1e-10 {
                return 0.0;
            }
            let weighted_sum: f32 = frame
                .iter()
                .enumerate()
                .map(|(i, &mag)| i as f32 * freq_resolution * mag)
                .sum();
            weighted_sum / total_magnitude
        })
        .collect()
}

/// Compute zero-crossing rate per frame.
pub fn zero_crossing_rate(samples: &[f32], frame_size: usize, hop_size: usize) -> Vec<f32> {
    let num_frames = if samples.len() >= frame_size {
        (samples.len() - frame_size) / hop_size + 1
    } else {
        0
    };

    let mut zcrs = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = i * hop_size;
        let end = (start + frame_size).min(samples.len());
        let window = &samples[start..end];

        let mut crossings = 0u32;
        for j in 1..window.len() {
            if window[j] * window[j - 1] < 0.0 {
                crossings += 1;
            }
        }
        zcrs.push(crossings as f32 / (window.len() - 1).max(1) as f32);
    }
    zcrs
}

/// Apply a 1D median filter to a slice.
pub fn median_filter_1d(data: &[f32], kernel_size: usize) -> Vec<f32> {
    let half = kernel_size / 2;
    let len = data.len();
    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(len);
        let mut window: Vec<f32> = data[start..end].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = window.len() / 2;
        result.push(window[mid]);
    }

    result
}

/// Harmonic-Percussive Source Separation (HPSS).
/// Returns (harmonic_energy_ratio, percussive_energy_ratio) per frame.
///
/// - Median filter across time (horizontal) -> harmonic component
/// - Median filter across frequency (vertical) -> percussive component
/// - Soft masks: H_mask = H^2 / (H^2 + P^2 + eps)
pub fn hpss(spectrogram: &[Vec<f32>], time_kernel: usize, freq_kernel: usize) -> HpssResult {
    let num_frames = spectrogram.len();
    if num_frames == 0 {
        return HpssResult {
            harmonic_ratio: Vec::new(),
            percussive_ratio: Vec::new(),
        };
    }
    let num_bins = spectrogram[0].len();

    // Harmonic: median filter across time axis for each frequency bin
    let mut harmonic_spec = vec![vec![0.0f32; num_bins]; num_frames];
    for bin in 0..num_bins {
        let time_slice: Vec<f32> = spectrogram.iter().map(|frame| frame[bin]).collect();
        let filtered = median_filter_1d(&time_slice, time_kernel);
        for (frame_idx, &val) in filtered.iter().enumerate() {
            harmonic_spec[frame_idx][bin] = val;
        }
    }

    // Percussive: median filter across frequency axis for each time frame
    let mut percussive_spec = vec![vec![0.0f32; num_bins]; num_frames];
    for frame_idx in 0..num_frames {
        let freq_slice = &spectrogram[frame_idx];
        let filtered = median_filter_1d(freq_slice, freq_kernel);
        percussive_spec[frame_idx] = filtered;
    }

    // Compute soft masks and energy ratios per frame
    let eps = 1e-10f32;
    let mut harmonic_ratio = Vec::with_capacity(num_frames);
    let mut percussive_ratio = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let mut h_energy = 0.0f32;
        let mut p_energy = 0.0f32;
        let mut total_energy = 0.0f32;

        for bin in 0..num_bins {
            let orig = spectrogram[frame_idx][bin];
            let h = harmonic_spec[frame_idx][bin];
            let p = percussive_spec[frame_idx][bin];

            let h2 = h * h;
            let p2 = p * p;
            let denom = h2 + p2 + eps;

            let h_mask = h2 / denom;
            let p_mask = p2 / denom;

            h_energy += h_mask * orig * orig;
            p_energy += p_mask * orig * orig;
            total_energy += orig * orig;
        }

        let total = total_energy + eps;
        harmonic_ratio.push(h_energy / total);
        percussive_ratio.push(p_energy / total);
    }

    HpssResult {
        harmonic_ratio,
        percussive_ratio,
    }
}

pub struct HpssResult {
    pub harmonic_ratio: Vec<f32>,
    pub percussive_ratio: Vec<f32>,
}

/// Compute all audio features for a set of samples at the default settings.
/// Returns features aligned by frame index (all vectors have the same length).
#[allow(dead_code)]
pub struct AudioFeatures {
    pub rms_db: Vec<f32>,
    pub spectral_centroid: Vec<f32>,
    pub zcr: Vec<f32>,
    pub harmonic_ratio: Vec<f32>,
    pub percussive_ratio: Vec<f32>,
    pub frame_times: Vec<f64>,
}

pub fn compute_features(samples: &[f32]) -> AudioFeatures {
    let spec = stft(samples, FFT_SIZE, HOP_SIZE);
    let rms = rms_energy(samples, FFT_SIZE, HOP_SIZE);
    let rms_db = rms_to_db(&rms);
    let centroid = spectral_centroid(&spec, SAMPLE_RATE, FFT_SIZE);
    let zcr = zero_crossing_rate(samples, FFT_SIZE, HOP_SIZE);
    let hpss_result = hpss(&spec, 31, 31);

    // Compute frame center times
    let frame_times: Vec<f64> = (0..rms_db.len())
        .map(|i| (i * HOP_SIZE + FFT_SIZE / 2) as f64 / SAMPLE_RATE as f64)
        .collect();

    // Ensure all vectors are the same length (use shortest)
    let len = rms_db
        .len()
        .min(centroid.len())
        .min(zcr.len())
        .min(hpss_result.harmonic_ratio.len())
        .min(frame_times.len());

    AudioFeatures {
        rms_db: rms_db[..len].to_vec(),
        spectral_centroid: centroid[..len].to_vec(),
        zcr: zcr[..len].to_vec(),
        harmonic_ratio: hpss_result.harmonic_ratio[..len].to_vec(),
        percussive_ratio: hpss_result.percussive_ratio[..len].to_vec(),
        frame_times: frame_times[..len].to_vec(),
    }
}
