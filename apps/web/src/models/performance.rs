use serde::{Deserialize, Serialize};

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

#[cfg(feature = "ssr")]
impl Performance {
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
            Performance {
                id: "zimerman-chopin-ballade-4".to_string(),
                composer: "Frederic Chopin".to_string(),
                piece_title: "Ballade No. 4 in F minor, Op. 52".to_string(),
                performer: "Krystian Zimerman".to_string(),
                thumbnail_url: "/images/zimerman.jpg".to_string(),
                audio_url: "/audio/zimerman-chopin-ballade-4.mp3".to_string(),
                duration_seconds: 660,
                year_recorded: Some(1988),
                description: Some("Zimerman's perfectionist approach yields a reading of extraordinary polish and depth.".to_string()),
            },
            Performance {
                id: "kissin-rachmaninoff-prelude".to_string(),
                composer: "Sergei Rachmaninoff".to_string(),
                piece_title: "Prelude in G minor, Op. 23 No. 5".to_string(),
                performer: "Evgeny Kissin".to_string(),
                thumbnail_url: "/images/kissin.jpg".to_string(),
                audio_url: "/audio/kissin-rachmaninoff-prelude.mp3".to_string(),
                duration_seconds: 240,
                year_recorded: Some(1993),
                description: Some("Kissin's youthful passion and technical command in one of Rachmaninoff's most beloved preludes.".to_string()),
            },
            Performance {
                id: "pollini-beethoven-appassionata".to_string(),
                composer: "Ludwig van Beethoven".to_string(),
                piece_title: "Piano Sonata No. 23 'Appassionata'".to_string(),
                performer: "Maurizio Pollini".to_string(),
                thumbnail_url: "/images/pollini.jpg".to_string(),
                audio_url: "/audio/pollini-beethoven-appassionata.mp3".to_string(),
                duration_seconds: 720,
                year_recorded: Some(1975),
                description: Some("Pollini's intellectual clarity combined with volcanic intensity in Beethoven's tempestuous masterpiece.".to_string()),
            },
        ]
    }

    pub fn find_by_id(id: &str) -> Option<Performance> {
        Self::get_demo_performances()
            .into_iter()
            .find(|p| p.id == id)
    }
}
