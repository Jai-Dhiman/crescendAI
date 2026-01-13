use serde::{Deserialize, Serialize};

/// A piano performance (demo or user-uploaded)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Performance {
    pub id: String,
    pub composer: String,
    pub piece_title: String,
    pub performer: String,
    pub thumbnail_url: String,
    pub audio_url: String,
    pub duration_seconds: u32,
    pub year_recorded: Option<u32>,
    pub description: Option<String>,
}

/// Response from audio upload endpoint
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UploadedPerformance {
    pub id: String,
    pub audio_url: String,
    pub r2_key: String,
    pub title: String,
    pub file_size_bytes: usize,
    pub content_type: String,
}

impl UploadedPerformance {
    /// Convert to a Performance for analysis
    pub fn to_performance(&self) -> Performance {
        Performance {
            id: self.id.clone(),
            composer: "Unknown".to_string(),
            piece_title: self.title.clone(),
            performer: "Your Recording".to_string(),
            thumbnail_url: "/images/upload-placeholder.svg".to_string(),
            audio_url: self.audio_url.clone(),
            duration_seconds: 0, // Will be determined during analysis
            year_recorded: None,
            description: Some("Your uploaded recording".to_string()),
        }
    }
}

#[cfg(feature = "ssr")]
impl Performance {
    /// Get the 3 demo performances (Horowitz, Argerich, Gould)
    pub fn get_demo_performances() -> Vec<Performance> {
        vec![
            Performance {
                id: "horowitz-chopin-ballade-1".to_string(),
                composer: "Frederic Chopin".to_string(),
                piece_title: "Ballade No. 1 in G minor, Op. 23".to_string(),
                performer: "Vladimir Horowitz".to_string(),
                thumbnail_url: "/images/horowitz.jpg".to_string(),
                audio_url: "/audio/horowitz-chopin-ballade-1.mp3".to_string(),
                duration_seconds: 540,
                year_recorded: Some(1968),
                description: Some("A legendary interpretation showcasing Horowitz's unparalleled virtuosity and dramatic intensity.".to_string()),
            },
            Performance {
                id: "argerich-prokofiev-toccata".to_string(),
                composer: "Sergei Prokofiev".to_string(),
                piece_title: "Toccata in D minor, Op. 11".to_string(),
                performer: "Martha Argerich".to_string(),
                thumbnail_url: "/images/argerich.jpg".to_string(),
                audio_url: "/audio/argerich-prokofiev-toccata.mp3".to_string(),
                duration_seconds: 300,
                year_recorded: Some(1975),
                description: Some("Argerich's fiery temperament perfectly matches Prokofiev's motoristic brilliance.".to_string()),
            },
            Performance {
                id: "gould-bach-goldberg-aria".to_string(),
                composer: "Johann Sebastian Bach".to_string(),
                piece_title: "Goldberg Variations - Aria".to_string(),
                performer: "Glenn Gould".to_string(),
                thumbnail_url: "/images/gould.jpg".to_string(),
                audio_url: "/audio/gould-bach-goldberg-aria.mp3".to_string(),
                duration_seconds: 180,
                year_recorded: Some(1981),
                description: Some("Gould's contemplative 1981 recording, a meditation on musical structure and time.".to_string()),
            },
        ]
    }

    pub fn find_by_id(id: &str) -> Option<Performance> {
        Self::get_demo_performances()
            .into_iter()
            .find(|p| p.id == id)
    }
}
