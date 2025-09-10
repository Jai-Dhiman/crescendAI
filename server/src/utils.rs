use worker::*;

/// Utility functions for the CrescendAI backend

/// Generate a random UUID for job IDs
pub fn generate_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Format timestamp for logging
pub fn format_timestamp() -> String {
    js_sys::Date::new_0().to_iso_string().as_string().unwrap()
}

/// Validate file size limits
pub fn validate_file_size(size: usize, max_size: usize) -> Result<()> {
    if size > max_size {
        return Err(worker::Error::RustError(format!(
            "File size {} exceeds maximum allowed size {}",
            size, max_size
        )));
    }
    Ok(())
}

/// Extract file extension from filename
pub fn get_file_extension(filename: &str) -> Option<&str> {
    filename.split('.').last()
}

/// Check if file extension is supported audio format
pub fn is_supported_audio_format(extension: &str) -> bool {
    matches!(extension.to_lowercase().as_str(), "wav" | "mp3" | "m4a" | "flac")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_uuid() {
        let uuid1 = generate_uuid();
        let uuid2 = generate_uuid();
        
        assert!(!uuid1.is_empty());
        assert!(!uuid2.is_empty());
        assert_ne!(uuid1, uuid2); // Should be different
        
        // Check UUID format (roughly)
        assert!(uuid1.len() >= 32);
        assert!(uuid1.contains('-'));
    }

    #[test]
    fn test_file_size_validation() {
        assert!(validate_file_size(100, 1000).is_ok());
        assert!(validate_file_size(1000, 1000).is_ok());
        assert!(validate_file_size(1001, 1000).is_err());
    }

    #[test]
    fn test_file_extension_extraction() {
        assert_eq!(get_file_extension("test.wav"), Some("wav"));
        assert_eq!(get_file_extension("song.mp3"), Some("mp3"));
        assert_eq!(get_file_extension("no_extension"), Some("no_extension"));
        assert_eq!(get_file_extension(""), Some(""));
    }

    #[test]
    fn test_supported_audio_formats() {
        assert!(is_supported_audio_format("wav"));
        assert!(is_supported_audio_format("WAV"));
        assert!(is_supported_audio_format("mp3"));
        assert!(is_supported_audio_format("MP3"));
        assert!(is_supported_audio_format("m4a"));
        assert!(is_supported_audio_format("flac"));
        
        assert!(!is_supported_audio_format("txt"));
        assert!(!is_supported_audio_format("jpg"));
        assert!(!is_supported_audio_format(""));
    }
}