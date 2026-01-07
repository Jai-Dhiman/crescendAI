use crate::models::{Performance, PerformanceDimensions, PracticeTip};

/// Mock Vectorize RAG service that returns relevant practice tips.
/// In production, this will query Cloudflare Vectorize with the performance context.
pub async fn get_practice_tips(
    performance: &Performance,
    dimensions: &PerformanceDimensions,
) -> Vec<PracticeTip> {
    // Find areas that could use improvement (lower scores)
    let mut areas: Vec<(&str, f64, &str, &str)> = vec![
        (
            "timing",
            dimensions.timing,
            "Rhythmic Precision",
            "Practice with a metronome at slower tempos, then gradually increase speed. \
            Focus on internalizing the pulse before adding expressive rubato.",
        ),
        (
            "articulation",
            (dimensions.articulation_length + dimensions.articulation_touch) / 2.0,
            "Touch and Articulation",
            "Experiment with different attack speeds and release times. \
            Practice scales and arpeggios focusing on evenness and control of each finger.",
        ),
        (
            "pedaling",
            (dimensions.pedal_amount + dimensions.pedal_clarity) / 2.0,
            "Pedal Technique",
            "Listen carefully to the harmonic clarity when pedaling. \
            Practice half-pedaling and flutter pedaling for more nuanced sound colors.",
        ),
        (
            "dynamics",
            dimensions.dynamics_range,
            "Dynamic Range",
            "Work on expanding your dynamic palette from pppp to ffff. \
            Practice crescendos and diminuendos over long phrases for greater control.",
        ),
        (
            "balance",
            dimensions.balance,
            "Voicing and Balance",
            "Practice bringing out individual voices in chordal passages. \
            Record yourself and listen for whether the melody projects clearly over accompaniment.",
        ),
    ];

    // Sort by score (lowest first - most room for improvement)
    areas.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Add composer-specific tip
    let composer_tip = match performance.composer.as_str() {
        "Frederic Chopin" => PracticeTip {
            title: "Chopin's Singing Tone".to_string(),
            description: "Chopin was inspired by bel canto opera singers. \
                Practice making the piano 'sing' by shaping phrases like a vocalist would, \
                with natural breathing points and expressive portamento-like connections."
                .to_string(),
        },
        "Johann Sebastian Bach" => PracticeTip {
            title: "Polyphonic Clarity".to_string(),
            description: "Practice each voice separately before combining them. \
                Bach's music requires independence of hands and fingers - \
                work on articulating each line with its own character and direction."
                .to_string(),
        },
        "Ludwig van Beethoven" => PracticeTip {
            title: "Beethoven's Architecture".to_string(),
            description: "Study the overall structure before diving into details. \
                Beethoven's music often features dramatic contrasts and developmental passages - \
                understanding the form helps communicate the musical narrative."
                .to_string(),
        },
        "Sergei Rachmaninoff" => PracticeTip {
            title: "Rachmaninoff's Lyricism".to_string(),
            description: "Despite the technical demands, Rachmaninoff's music is deeply lyrical. \
                Find the singing melody even in virtuosic passages, \
                and use the bass line to create a rich harmonic foundation."
                .to_string(),
        },
        "Sergei Prokofiev" => PracticeTip {
            title: "Prokofiev's Precision".to_string(),
            description: "Prokofiev's music demands rhythmic exactness and percussive clarity. \
                Practice with sharp, precise attacks while maintaining musical flow. \
                The 'wrong notes' are intentional - commit to them fully."
                .to_string(),
        },
        _ => PracticeTip {
            title: "General Practice Strategy".to_string(),
            description: "Break difficult passages into small sections. \
                Practice hands separately, then combine at a slower tempo. \
                Gradually increase speed only when accuracy is consistent."
                .to_string(),
        },
    };

    // Return top 2 improvement tips plus composer-specific tip
    vec![
        PracticeTip {
            title: areas[0].2.to_string(),
            description: areas[0].3.to_string(),
        },
        PracticeTip {
            title: areas[1].2.to_string(),
            description: areas[1].3.to_string(),
        },
        composer_tip,
    ]
}
