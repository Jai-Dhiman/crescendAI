import SwiftUI

struct FocusModeExercise: Identifiable {
    let id = UUID()
    let title: String
    let instructions: String
    let dimension: String
}

struct FocusModeView: View {
    @Environment(\.dismiss) private var dismiss
    let exercises: [FocusModeExercise]
    let dimension: String

    @State private var currentIndex = 0
    @State private var isRecording = false
    @State private var feedbackText: String?

    /// Near-black background for deeper immersion
    private let focusBackground = Color(red: 0x1A / 255.0, green: 0x18 / 255.0, blue: 0x16 / 255.0)

    var body: some View {
        NavigationStack {
            ZStack {
                focusBackground.ignoresSafeArea()

                VStack(spacing: CrescendSpacing.space6) {
                    // Header
                    VStack(spacing: CrescendSpacing.space2) {
                        DimensionPill(dimension: dimension)

                        if !exercises.isEmpty {
                            Text(exercises[currentIndex].title)
                                .font(CrescendFont.headingLG())
                                .foregroundStyle(CrescendColor.foreground)

                            Text("\(currentIndex + 1) of \(exercises.count)")
                                .font(CrescendFont.labelMD())
                                .foregroundStyle(CrescendColor.tertiaryText)
                        }
                    }
                    .padding(.top, CrescendSpacing.space6)

                    Spacer()

                    // Instructions or recording state
                    if let feedbackText {
                        CrescendCard {
                            VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
                                DimensionPill(dimension: dimension)
                                Text(feedbackText)
                                    .font(CrescendFont.bodyLG())
                                    .foregroundStyle(CrescendColor.foreground)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                        .padding(.horizontal, CrescendSpacing.space4)
                    } else if isRecording {
                        WaveformView(level: 0.2, duration: 0)
                            .padding(.horizontal, CrescendSpacing.space6)
                    } else if !exercises.isEmpty {
                        Text(exercises[currentIndex].instructions)
                            .font(CrescendFont.bodyLG())
                            .foregroundStyle(CrescendColor.foreground)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, CrescendSpacing.space6)
                    }

                    Spacer()

                    // Controls
                    VStack(spacing: CrescendSpacing.space4) {
                        if feedbackText != nil {
                            if currentIndex < exercises.count - 1 {
                                CrescendButton("Next Exercise", style: .primary) {
                                    currentIndex += 1
                                    feedbackText = nil
                                }
                            }
                            CrescendButton("Back to Practice", style: .secondary) {
                                dismiss()
                            }
                        } else if isRecording {
                            Button(action: {
                                isRecording = false
                                // Placeholder feedback
                                feedbackText = "Good work on the dynamic control. Try emphasizing the contrast even more on the next attempt."
                            }) {
                                Circle()
                                    .fill(CrescendColor.foreground)
                                    .frame(width: 56, height: 56)
                                    .overlay {
                                        RoundedRectangle(cornerRadius: 4)
                                            .fill(focusBackground)
                                            .frame(width: 20, height: 20)
                                    }
                            }
                            .buttonStyle(CrescendPressStyle())
                        } else {
                            CrescendButton("I'm Ready", style: .primary) {
                                isRecording = true
                            }
                        }
                    }
                    .padding(.horizontal, CrescendSpacing.space6)
                    .padding(.bottom, CrescendSpacing.space8)
                }
            }
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark")
                            .foregroundStyle(CrescendColor.secondaryText)
                    }
                }
            }
            .toolbarBackground(.hidden, for: .navigationBar)
        }
    }
}

#Preview {
    FocusModeView(
        exercises: [
            FocusModeExercise(
                title: "Dynamic Range",
                instructions: "Play the opening 4 bars, gradually building from piano to forte. Focus on smooth, continuous growth.",
                dimension: "dynamics"
            ),
            FocusModeExercise(
                title: "Sudden Contrast",
                instructions: "Play the forte passage, then immediately drop to pianissimo on the repeat.",
                dimension: "dynamics"
            ),
        ],
        dimension: "dynamics"
    )
    .crescendTheme()
}
