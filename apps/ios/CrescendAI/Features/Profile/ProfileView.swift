import SwiftData
import SwiftUI

struct ProfileView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(AuthService.self) private var authService

    @Query(sort: \PracticeSessionRecord.startedAt) private var sessions: [PracticeSessionRecord]
    @Query private var students: [Student]

    private let dimensionMeta: [(name: String, color: Color, key: String)] = [
        ("Dynamics", CrescendColor.dimDynamics, "dynamics"),
        ("Timing", CrescendColor.dimTiming, "timing"),
        ("Pedaling", CrescendColor.dimPedaling, "pedaling"),
        ("Articulation", CrescendColor.dimArticulation, "articulation"),
        ("Phrasing", CrescendColor.dimPhrasing, "phrasing"),
        ("Interpretation", CrescendColor.dimInterpretation, "interpretation"),
    ]

    private var student: Student? { students.first }

    /// True once at least one dimension has enough history (>= 2 scored sessions) to plot.
    private var hasTrendData: Bool {
        dimensionMeta.contains { StudentModelService.dimensionSeries(for: $0.key, sessions: sessions).count >= 2 }
    }

    private var progressSubtitle: String {
        guard let student, student.baselineSessionCount > 0 else { return "New pianist" }
        let level = student.inferredLevel?.capitalized ?? "Pianist"
        let count = student.baselineSessionCount
        return "\(level) · \(count) practice session\(count == 1 ? "" : "s")"
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: CrescendSpacing.space6) {
                    // User info
                    VStack(alignment: .leading, spacing: CrescendSpacing.space1) {
                        Text(authService.isAuthenticated ? "Pianist" : "Guest")
                            .font(CrescendFont.headingLG())
                            .foregroundStyle(CrescendColor.foreground)

                        if authService.isAuthenticated {
                            Text(progressSubtitle)
                                .font(CrescendFont.bodySM())
                                .foregroundStyle(CrescendColor.secondaryText)
                        }
                    }
                    .padding(.horizontal, CrescendSpacing.space4)

                    // Trends
                    VStack(alignment: .leading, spacing: CrescendSpacing.space4) {
                        Text("Trends")
                            .font(CrescendFont.headingMD())
                            .foregroundStyle(CrescendColor.foreground)
                            .padding(.horizontal, CrescendSpacing.space4)

                        if hasTrendData {
                            ForEach(dimensionMeta, id: \.key) { meta in
                                HStack(spacing: CrescendSpacing.space3) {
                                    Text(meta.name)
                                        .font(CrescendFont.labelMD())
                                        .foregroundStyle(CrescendColor.secondaryText)
                                        .frame(width: 90, alignment: .trailing)

                                    SparklineView(
                                        values: StudentModelService.dimensionSeries(for: meta.key, sessions: sessions),
                                        color: meta.color
                                    )
                                }
                                .padding(.horizontal, CrescendSpacing.space4)
                            }
                        } else {
                            Text("Practice a few sessions to see your dimension trends.")
                                .font(CrescendFont.bodySM())
                                .foregroundStyle(CrescendColor.tertiaryText)
                                .padding(.horizontal, CrescendSpacing.space4)
                        }
                    }

                    // Settings
                    VStack(alignment: .leading, spacing: CrescendSpacing.space3) {
                        Text("Settings")
                            .font(CrescendFont.headingMD())
                            .foregroundStyle(CrescendColor.foreground)

                        HStack {
                            Text("Version")
                                .font(CrescendFont.bodyMD())
                                .foregroundStyle(CrescendColor.foreground)
                            Spacer()
                            Text("1.0.0")
                                .font(CrescendFont.bodyMD())
                                .foregroundStyle(CrescendColor.secondaryText)
                        }
                    }
                    .padding(.horizontal, CrescendSpacing.space4)

                    // Sign out
                    Button {
                        authService.signOut()
                        dismiss()
                    } label: {
                        Text("Sign Out")
                            .font(CrescendFont.labelLG())
                            .foregroundStyle(.red.opacity(0.8))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, CrescendSpacing.space3)
                    }
                    .buttonStyle(CrescendPressStyle())
                    .padding(.horizontal, CrescendSpacing.space4)
                    .padding(.bottom, CrescendSpacing.space6)
                }
                .padding(.top, CrescendSpacing.space4)
            }
            .background(CrescendColor.background)
            .navigationTitle("Profile")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                        .font(CrescendFont.labelLG())
                        .foregroundStyle(CrescendColor.foreground)
                }
            }
            .toolbarBackground(CrescendColor.surface, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
        }
    }
}

#Preview {
    ProfileView()
        .crescendTheme()
        .environment(AuthService())
        .modelContainer(
            for: [Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self, ConversationRecord.self],
            inMemory: true
        )
}
