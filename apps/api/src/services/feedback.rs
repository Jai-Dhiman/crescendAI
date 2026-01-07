use crate::models::{Performance, PerformanceDimensions};

/// Mock feedback generation that returns teacher-style commentary.
/// In production, this will call Workers AI with the dimensions and RAG context.
pub async fn generate_teacher_feedback(
    performance: &Performance,
    dimensions: &PerformanceDimensions,
) -> String {
    // Find the strongest dimensions
    let mut scores: Vec<(&str, f64)> = vec![
        ("timing precision", dimensions.timing),
        ("articulation control", (dimensions.articulation_length + dimensions.articulation_touch) / 2.0),
        ("pedaling technique", (dimensions.pedal_amount + dimensions.pedal_clarity) / 2.0),
        ("tonal variety", (dimensions.timbre_variety + dimensions.timbre_depth) / 2.0),
        ("dynamic expression", dimensions.dynamics_range),
        ("dramatic intensity", dimensions.drama),
        ("musical imagination", dimensions.mood_imagination),
        ("interpretive depth", dimensions.interpretation_overall),
    ];
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_strength = scores[0].0;
    let second_strength = scores[1].0;
    let growth_area = scores.last().unwrap().0;

    // Generate feedback based on performer and piece
    let performer_style = match performance.performer.as_str() {
        "Vladimir Horowitz" => "legendary virtuosity and dramatic flair",
        "Martha Argerich" => "fiery temperament and electrifying energy",
        "Glenn Gould" => "intellectual clarity and unique artistic vision",
        "Krystian Zimerman" => "meticulous attention to detail and tonal refinement",
        "Evgeny Kissin" => "passionate intensity and technical brilliance",
        "Maurizio Pollini" => "structural clarity and controlled power",
        _ => "distinctive artistic voice",
    };

    let overall_score = dimensions.interpretation_overall;
    let quality_descriptor = if overall_score >= 0.90 {
        "exceptional"
    } else if overall_score >= 0.80 {
        "impressive"
    } else if overall_score >= 0.70 {
        "solid"
    } else {
        "developing"
    };

    format!(
        "This {} interpretation of {} by {} demonstrates {}. \
        The performance shows particular strength in {}, which brings out the \
        emotional depth of the piece beautifully. The {} also contributes significantly \
        to the overall musical narrative.\n\n\
        The recording captures {}'s {}, particularly evident in the way phrases \
        are shaped and the natural ebb and flow of the musical line. \
        For continued growth, focusing on {} could add even more nuance \
        to this already compelling interpretation.\n\n\
        Overall, this is a performance that rewards careful listening and \
        demonstrates a deep understanding of {}'s musical language.",
        quality_descriptor,
        performance.piece_title,
        performance.performer,
        performer_style,
        top_strength,
        second_strength,
        performance.performer,
        performer_style,
        growth_area,
        performance.composer
    )
}
