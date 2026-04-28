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
