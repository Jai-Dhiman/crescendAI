use crate::models::{AnalysisResult, PerformanceDimensions, Performance, PracticeTip};

pub fn get_performances() -> Vec<Performance> {
    vec![
        Performance {
            id: "horowitz-chopin-ballade-1".into(),
            composer: "Frederic Chopin".into(),
            piece_title: "Ballade No. 1 in G minor, Op. 23".into(),
            performer: "Vladimir Horowitz".into(),
            thumbnail_url: "/images/placeholder-1.svg".into(),
            audio_url: "/audio/sample.mp3".into(),
            duration_seconds: 560,
            year_recorded: Some(1968),
            description: Some(
                "A legendary interpretation showcasing Horowitz's unparalleled command of dynamics and color."
                    .into(),
            ),
        },
        Performance {
            id: "argerich-prokofiev-toccata".into(),
            composer: "Sergei Prokofiev".into(),
            piece_title: "Toccata in D minor, Op. 11".into(),
            performer: "Martha Argerich".into(),
            thumbnail_url: "/images/placeholder-2.svg".into(),
            audio_url: "/audio/sample.mp3".into(),
            duration_seconds: 310,
            year_recorded: Some(1972),
            description: Some("Breathtaking virtuosity and precision in this early recording.".into()),
        },
        Performance {
            id: "gould-bach-goldberg".into(),
            composer: "Johann Sebastian Bach".into(),
            piece_title: "Goldberg Variations, BWV 988 (Aria)".into(),
            performer: "Glenn Gould".into(),
            thumbnail_url: "/images/placeholder-3.svg".into(),
            audio_url: "/audio/sample.mp3".into(),
            duration_seconds: 180,
            year_recorded: Some(1981),
            description: Some("The contemplative 1981 recording of the opening Aria.".into()),
        },
        Performance {
            id: "zimerman-chopin-ballade-4".into(),
            composer: "Frederic Chopin".into(),
            piece_title: "Ballade No. 4 in F minor, Op. 52".into(),
            performer: "Krystian Zimerman".into(),
            thumbnail_url: "/images/placeholder-4.svg".into(),
            audio_url: "/audio/sample.mp3".into(),
            duration_seconds: 720,
            year_recorded: Some(1988),
            description: Some("A masterful reading of Chopin's most complex Ballade.".into()),
        },
        Performance {
            id: "kissin-rachmaninoff-prelude".into(),
            composer: "Sergei Rachmaninoff".into(),
            piece_title: "Prelude in G minor, Op. 23 No. 5".into(),
            performer: "Evgeny Kissin".into(),
            thumbnail_url: "/images/placeholder-5.svg".into(),
            audio_url: "/audio/sample.mp3".into(),
            duration_seconds: 240,
            year_recorded: Some(1992),
            description: Some("Powerful and dramatic interpretation.".into()),
        },
        Performance {
            id: "pollini-beethoven-sonata".into(),
            composer: "Ludwig van Beethoven".into(),
            piece_title: "Piano Sonata No. 23 'Appassionata'".into(),
            performer: "Maurizio Pollini".into(),
            thumbnail_url: "/images/placeholder-6.svg".into(),
            audio_url: "/audio/sample.mp3".into(),
            duration_seconds: 1380,
            year_recorded: Some(1975),
            description: Some("Intellectually rigorous yet emotionally compelling.".into()),
        },
    ]
}

pub fn get_performance_by_id(id: &str) -> Option<Performance> {
    get_performances().into_iter().find(|p| p.id == id)
}

pub fn mock_analyze_performance(_performance_id: &str) -> AnalysisResult {
    AnalysisResult {
        performance_id: _performance_id.to_string(),
        dimensions: PerformanceDimensions {
            timing: 0.85,
            articulation_length: 0.78,
            articulation_touch: 0.82,
            pedal_amount: 0.75,
            pedal_clarity: 0.88,
            timbre_variety: 0.92,
            timbre_depth: 0.87,
            timbre_brightness: 0.73,
            timbre_loudness: 0.68,
            dynamics_range: 0.95,
            tempo: 0.80,
            space: 0.85,
            balance: 0.90,
            drama: 0.88,
            mood_valence: 0.72,
            mood_energy: 0.85,
            mood_imagination: 0.91,
            interpretation_sophistication: 0.89,
            interpretation_overall: 0.87,
        },
        teacher_feedback: "Your sense of timing shows wonderful musical intuition - the subtle rubato in the second theme creates a natural, breathing quality that draws the listener into the narrative arc of the piece. The timbral variety you achieve, particularly in the softer passages, demonstrates a sophisticated understanding of the piano's coloristic possibilities.\n\nTo further develop your dynamic palette, try practicing the opening phrase at three different dynamic levels while maintaining the same emotional intensity. Your pedal technique is already quite clean, but exploring half-pedaling in the transitional passages could add even more clarity to the harmonic progressions.\n\nThe dramatic arc of your performance builds convincingly toward the climax. Consider experimenting with slightly more space before the recapitulation to heighten the sense of return.".into(),
        practice_tips: vec![
            PracticeTip {
                title: "Dynamic Layering Exercise".into(),
                description: "Practice the main theme at pp, mp, and mf while keeping the same expressive character. This builds control and awareness of your dynamic range.".into(),
            },
            PracticeTip {
                title: "Pedal Clarity Check".into(),
                description: "Record yourself playing the transitional passages with and without pedal. Compare the harmonic clarity and find the optimal balance.".into(),
            },
            PracticeTip {
                title: "Structural Breathing".into(),
                description: "Mark the major structural points in your score. Practice adding micro-pauses at these moments to help listeners follow the musical architecture.".into(),
            },
        ],
    }
}

/// Loading messages to cycle through during analysis
pub fn get_loading_messages() -> Vec<&'static str> {
    vec![
        "Analyzing articulation patterns...",
        "Evaluating pedal technique...",
        "Measuring dynamic range...",
        "Assessing timbral qualities...",
        "Examining phrasing and tempo...",
        "Detecting expressive nuances...",
        "Analyzing harmonic balance...",
        "Evaluating interpretive choices...",
        "Generating personalized feedback...",
    ]
}
