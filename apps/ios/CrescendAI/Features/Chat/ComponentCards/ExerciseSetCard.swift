import SwiftUI

struct ExerciseItem: Identifiable {
    let id = UUID()
    let title: String
    let description: String
}

struct ExerciseSetCard: View {
    let dimension: String
    let exercises: [ExerciseItem]
    let onStart: () -> Void

    var body: some View {
        CrescendCard {
            VStack(alignment: .leading, spacing: CrescendSpacing.space3) {
                HStack {
                    DimensionPill(dimension: dimension)
                    Spacer()
                    Text("\(exercises.count) exercises")
                        .font(CrescendFont.labelSM())
                        .foregroundStyle(CrescendColor.tertiaryText)
                }

                ForEach(Array(exercises.enumerated()), id: \.element.id) { index, exercise in
                    HStack(alignment: .top, spacing: CrescendSpacing.space3) {
                        Text("\(index + 1)")
                            .font(CrescendFont.labelMD())
                            .foregroundStyle(CrescendColor.tertiaryText)
                            .frame(width: 20)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(exercise.title)
                                .font(CrescendFont.bodySM(.medium))
                                .foregroundStyle(CrescendColor.foreground)
                            Text(exercise.description)
                                .font(CrescendFont.bodySM())
                                .foregroundStyle(CrescendColor.secondaryText)
                                .lineLimit(2)
                        }
                    }

                    if index < exercises.count - 1 {
                        Divider()
                            .background(CrescendColor.border)
                    }
                }

                Button(action: onStart) {
                    HStack(spacing: CrescendSpacing.space2) {
                        Image(systemName: "play.fill")
                            .font(.system(size: 12, weight: .medium))
                        Text("Start Focus Mode")
                            .font(CrescendFont.labelLG())
                    }
                    .foregroundStyle(CrescendColor.foreground)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, CrescendSpacing.space2)
                    .background(CrescendColor.surface2)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(CrescendPressStyle())
            }
        }
    }
}

#Preview {
    ExerciseSetCard(
        dimension: "dynamics",
        exercises: [
            ExerciseItem(title: "Dynamic Range", description: "Play bars 1-4 building from p to f"),
            ExerciseItem(title: "Sudden Contrast", description: "Forte passage then immediate pianissimo"),
            ExerciseItem(title: "Gradual Decay", description: "Sustain a chord and let it diminuendo naturally"),
        ],
        onStart: {}
    )
    .padding()
    .background(CrescendColor.background)
    .crescendTheme()
}
