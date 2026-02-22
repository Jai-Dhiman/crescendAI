import SwiftUI

struct AnalysisView: View {
    let result: AnalysisResult

    @State private var viewModel: AnalysisViewModel?

    var body: some View {
        ZStack {
            CrescendColor.background
                .ignoresSafeArea()

            ScrollView {
                VStack(spacing: CrescendSpacing.space6) {
                    headerSection

                    if let vm = viewModel {
                        dimensionsList(vm)

                        if let context = result.calibration_context {
                            calibrationNote(context)
                        }
                    }
                }
                .padding(.horizontal, CrescendSpacing.space4)
                .padding(.vertical, CrescendSpacing.space6)
            }
        }
        .navigationTitle("Analysis")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            viewModel = AnalysisViewModel(result: result)
        }
    }

    private var headerSection: some View {
        CrescendCard(style: .elevated) {
            VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
                Text("Performance Analysis")
                    .font(CrescendFont.headingLG())
                    .foregroundStyle(CrescendColor.foreground)

                Text("19-dimensional evaluation of your piano performance, calibrated against professional benchmarks.")
                    .font(CrescendFont.bodySM())
                    .foregroundStyle(CrescendColor.secondaryText)
            }
        }
    }

    private func dimensionsList(_ vm: AnalysisViewModel) -> some View {
        CrescendCard {
            VStack(alignment: .leading, spacing: CrescendSpacing.space4) {
                Text("Dimension Scores")
                    .font(CrescendFont.headingMD())
                    .foregroundStyle(CrescendColor.foreground)

                ForEach(Array(vm.labeledScores.enumerated()), id: \.offset) { _, pair in
                    dimensionRow(label: pair.0, score: pair.1, vm: vm)
                }
            }
        }
    }

    private func dimensionRow(label: String, score: Double, vm: AnalysisViewModel) -> some View {
        VStack(alignment: .leading, spacing: CrescendSpacing.space1) {
            HStack {
                Text(label)
                    .font(CrescendFont.labelLG())
                    .foregroundStyle(CrescendColor.foreground)

                Spacer()

                Text(String(format: "%.2f", score))
                    .font(CrescendFont.labelMD())
                    .foregroundStyle(CrescendColor.secondaryText)
                    .monospacedDigit()

                Text(vm.scoreLabel(score))
                    .font(CrescendFont.labelSM())
                    .foregroundStyle(CrescendColor.background)
                    .padding(.horizontal, CrescendSpacing.space2)
                    .padding(.vertical, 2)
                    .background(CrescendColor.foreground.opacity(score >= 0.5 ? 0.8 : 0.4))
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            }

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(CrescendColor.subtleFill)

                    RoundedRectangle(cornerRadius: 2)
                        .fill(CrescendColor.foreground.opacity(0.7))
                        .frame(width: max(0, geometry.size.width * min(1, score)))
                }
            }
            .frame(height: 6)
        }
    }

    private func calibrationNote(_ context: String) -> some View {
        CrescendCard {
            VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
                Text("Calibration")
                    .font(CrescendFont.labelLG())
                    .foregroundStyle(CrescendColor.foreground)

                Text(context)
                    .font(CrescendFont.bodySM())
                    .foregroundStyle(CrescendColor.secondaryText)
            }
        }
    }
}

#Preview {
    let sampleDimensions = PerformanceDimensions(
        timing: 0.65, articulation_length: 0.58, articulation_touch: 0.72,
        pedal_amount: 0.45, pedal_clarity: 0.51, timbre_variety: 0.68,
        timbre_depth: 0.74, timbre_brightness: 0.61, timbre_loudness: 0.55,
        dynamics_range: 0.70, tempo: 0.62, space: 0.48, balance: 0.66,
        drama: 0.53, mood_valence: 0.60, mood_energy: 0.57,
        mood_imagination: 0.42, interpretation_sophistication: 0.38,
        interpretation_overall: 0.55
    )

    let result = AnalysisResult(
        performance_id: "preview",
        dimensions: sampleDimensions,
        calibrated_dimensions: sampleDimensions,
        calibration_context: "Scores are calibrated relative to MAESTRO professional benchmarks. A score of ~0.5 represents an average professional level.",
        models: [],
        teacher_feedback: CitedFeedback(html: "", plain_text: "", citations: []),
        practice_tips: []
    )

    NavigationStack {
        AnalysisView(result: result)
    }
    .crescendTheme()
}
