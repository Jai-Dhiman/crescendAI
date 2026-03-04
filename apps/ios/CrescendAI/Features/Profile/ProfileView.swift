import SwiftUI

struct ProfileView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(AuthService.self) private var authService
    @State private var signOutError: String?

    private let dimensions = [
        ("Dynamics", CrescendColor.dimDynamics, [0.4, 0.5, 0.45, 0.6, 0.55, 0.7]),
        ("Timing", CrescendColor.dimTiming, [0.5, 0.52, 0.48, 0.55, 0.6]),
        ("Pedaling", CrescendColor.dimPedaling, [0.3, 0.35, 0.4, 0.38, 0.45]),
        ("Articulation", CrescendColor.dimArticulation, [0.6, 0.58, 0.62, 0.65]),
        ("Phrasing", CrescendColor.dimPhrasing, [0.45, 0.5, 0.48, 0.55, 0.52]),
        ("Interpretation", CrescendColor.dimInterpretation, [0.35, 0.4, 0.42, 0.45]),
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: CrescendSpacing.space6) {
                    // User info
                    VStack(alignment: .leading, spacing: CrescendSpacing.space1) {
                        Text(authService.appleUserId != nil ? "Pianist" : "Guest")
                            .font(CrescendFont.headingLG())
                            .foregroundStyle(CrescendColor.foreground)

                        if authService.appleUserId != nil {
                            Text("Signed in with Apple")
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

                        ForEach(dimensions, id: \.0) { name, color, values in
                            HStack(spacing: CrescendSpacing.space3) {
                                Text(name)
                                    .font(CrescendFont.labelMD())
                                    .foregroundStyle(CrescendColor.secondaryText)
                                    .frame(width: 90, alignment: .trailing)

                                SparklineView(values: values, color: color)
                            }
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

                    if let signOutError {
                        Text(signOutError)
                            .font(CrescendFont.bodySM())
                            .foregroundStyle(.red.opacity(0.8))
                            .padding(.horizontal, CrescendSpacing.space4)
                    }

                    // Sign out
                    Button {
                        do {
                            try authService.signOut()
                            dismiss()
                        } catch {
                            signOutError = error.localizedDescription
                        }
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
}
