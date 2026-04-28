import SwiftUI

struct ArtifactRenderer: View {
    let config: ArtifactConfig

    var body: some View {
        switch config {
        case .exerciseSet(let c):
            ArtifactExerciseSetCard(config: c)
        case .scoreHighlight:
            ArtifactScoreHighlightPlaceholder()
        case .keyboardGuide(let c):
            ArtifactKeyboardGuideCard(config: c)
        case .unknown:
            EmptyView()
        }
    }
}

private struct ArtifactScoreHighlightPlaceholder: View {
    var body: some View {
        RoundedRectangle(cornerRadius: 8)
            .fill(CrescendColor.surface2)
            .frame(height: 80)
            .overlay(
                Text("Score highlight")
                    .font(CrescendFont.labelSM())
                    .foregroundStyle(CrescendColor.tertiaryText)
            )
    }
}

private struct ArtifactExerciseSetCard: View {
    let config: ExerciseSetConfig

    var body: some View {
        VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
            Text(config.targetSkill)
                .font(CrescendFont.labelMD())
                .foregroundStyle(CrescendColor.foreground)
            ForEach(config.exercises.indices, id: \.self) { i in
                let ex = config.exercises[i]
                VStack(alignment: .leading, spacing: CrescendSpacing.space1) {
                    Text(ex.title)
                        .font(CrescendFont.bodySM())
                        .foregroundStyle(CrescendColor.foreground)
                    Text(ex.instruction)
                        .font(CrescendFont.labelSM())
                        .foregroundStyle(CrescendColor.secondaryText)
                }
            }
        }
        .padding(CrescendSpacing.space3)
        .background(CrescendColor.surface2)
        .cornerRadius(8)
    }
}

private struct ArtifactKeyboardGuideCard: View {
    let config: KeyboardGuideConfig

    var body: some View {
        VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
            Text(config.title)
                .font(CrescendFont.labelMD())
                .foregroundStyle(CrescendColor.foreground)
            Text(config.description)
                .font(CrescendFont.bodySM())
                .foregroundStyle(CrescendColor.secondaryText)
        }
        .padding(CrescendSpacing.space3)
        .background(CrescendColor.surface2)
        .cornerRadius(8)
    }
}

#Preview("ExerciseSet") {
    ArtifactRenderer(config: .exerciseSet(ExerciseSetConfig(
        sourcePassage: "Chopin Op.9 mm.1-4",
        targetSkill: "legato phrasing",
        exercises: [
            ExerciseItem(title: "Slow practice", instruction: "Play mm.1-4 at 60bpm", focusDimension: "phrasing", exerciseId: "ex-001", hands: nil),
            ExerciseItem(title: "Phrase shaping", instruction: "Build the phrase peak at beat 3", focusDimension: "phrasing", exerciseId: "ex-002", hands: nil),
        ]
    )))
    .padding()
    .background(CrescendColor.background)
    .crescendTheme()
}

#Preview("KeyboardGuide") {
    ArtifactRenderer(config: .keyboardGuide(KeyboardGuideConfig(
        title: "Hand positioning",
        description: "Keep your wrist level with the keys and let arm weight do the work.",
        hands: "both",
        fingering: nil
    )))
    .padding()
    .background(CrescendColor.background)
    .crescendTheme()
}

#Preview("ScoreHighlight") {
    ArtifactRenderer(config: .scoreHighlight(ScoreHighlightConfig(
        pieceId: "chopin-op9-no2",
        highlights: [ScoreHighlight(bars: [1, 4], dimension: "dynamics", annotation: "Build to forte here")]
    )))
    .padding()
    .background(CrescendColor.background)
    .crescendTheme()
}
