import SwiftUI

struct HomeView: View {
    var body: some View {
        ZStack {
            CrescendColor.background
                .ignoresSafeArea()

            ScrollView {
                VStack(spacing: CrescendSpacing.space12) {
                    // Hero
                    VStack(spacing: CrescendSpacing.space4) {
                        Text("CrescendAI")
                            .font(CrescendFont.displayXL())
                            .foregroundStyle(CrescendColor.foreground)

                        Text("A teacher for every pianist.")
                            .font(CrescendFont.bodyLG())
                            .foregroundStyle(CrescendColor.secondaryText)
                    }
                    .padding(.top, CrescendSpacing.space20)

                    // Record CTA
                    VStack(spacing: CrescendSpacing.space3) {
                        NavigationLink {
                            RecordingView()
                        } label: {
                            HStack(spacing: CrescendSpacing.space2) {
                                Image(systemName: "mic.fill")
                                    .font(.system(size: 16, weight: .medium))
                                Text("Record Performance")
                                    .font(CrescendFont.labelLG())
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.horizontal, CrescendSpacing.space6)
                            .padding(.vertical, CrescendSpacing.space3)
                            .foregroundStyle(CrescendColor.background)
                            .background(CrescendColor.foreground)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                        .buttonStyle(CrescendPressNavigationStyle())

                        Text("Record a piano performance to receive AI-powered feedback on your playing.")
                            .font(CrescendFont.bodySM())
                            .foregroundStyle(CrescendColor.secondaryText)
                            .multilineTextAlignment(.center)
                    }
                    .padding(.horizontal, CrescendSpacing.space6)

                    // MIDI Player entry
                    CrescendCard(style: .interactive) {
                        NavigationLink {
                            MIDIPlayerView()
                        } label: {
                            HStack(spacing: CrescendSpacing.space3) {
                                Image(systemName: "pianokeys")
                                    .font(.system(size: 24, weight: .medium))
                                    .foregroundStyle(CrescendColor.foreground)
                                    .frame(width: 48, height: 48)
                                    .background(CrescendColor.subtleFill)
                                    .clipShape(RoundedRectangle(cornerRadius: 8))

                                VStack(alignment: .leading, spacing: CrescendSpacing.space1) {
                                    Text("MIDI Player")
                                        .font(CrescendFont.headingMD())
                                        .foregroundStyle(CrescendColor.foreground)
                                    Text("Open and play MIDI files with the built-in piano sound.")
                                        .font(CrescendFont.bodySM())
                                        .foregroundStyle(CrescendColor.secondaryText)
                                }

                                Spacer()

                                Image(systemName: "chevron.right")
                                    .font(.system(size: 14, weight: .semibold))
                                    .foregroundStyle(CrescendColor.secondaryText)
                            }
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.horizontal, CrescendSpacing.space4)
                }
                .padding(.bottom, CrescendSpacing.space12)
            }
        }
        .navigationBarHidden(true)
    }
}

private struct CrescendPressNavigationStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
            .opacity(configuration.isPressed ? 0.85 : 1.0)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

#Preview {
    NavigationStack {
        HomeView()
    }
    .crescendTheme()
}
