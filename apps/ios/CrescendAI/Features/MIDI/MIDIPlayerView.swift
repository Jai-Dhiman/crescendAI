import SwiftUI
import UniformTypeIdentifiers

struct MIDIPlayerView: View {
    @State private var viewModel = MIDIPlayerViewModel()

    var body: some View {
        ZStack {
            CrescendColor.background
                .ignoresSafeArea()

            VStack(spacing: CrescendSpacing.space8) {
                Spacer()

                contentSection

                Spacer()
            }
            .padding(.horizontal, CrescendSpacing.space6)
        }
        .navigationTitle("MIDI Player")
        .navigationBarTitleDisplayMode(.inline)
        .fileImporter(
            isPresented: $viewModel.showFilePicker,
            allowedContentTypes: [UTType(filenameExtension: "mid") ?? .data, UTType(filenameExtension: "midi") ?? .data],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    viewModel.loadFile(url: url)
                }
            case .failure(let error):
                viewModel.setError(error.localizedDescription)
            }
        }
    }

    @ViewBuilder
    private var contentSection: some View {
        switch viewModel.state {
        case .empty:
            emptyState

        case .loaded:
            playerSection

        case .error(let message):
            VStack(spacing: CrescendSpacing.space4) {
                Text(message)
                    .font(CrescendFont.bodySM())
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)

                CrescendButton("Try Again", style: .secondary) {
                    viewModel.showFilePicker = true
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: CrescendSpacing.space6) {
            Image(systemName: "pianokeys")
                .font(.system(size: 48, weight: .light))
                .foregroundStyle(CrescendColor.secondaryText)

            VStack(spacing: CrescendSpacing.space2) {
                Text("No MIDI file loaded")
                    .font(CrescendFont.headingMD())
                    .foregroundStyle(CrescendColor.foreground)

                Text("Open a .mid file to play it with the built-in piano sound.")
                    .font(CrescendFont.bodySM())
                    .foregroundStyle(CrescendColor.secondaryText)
                    .multilineTextAlignment(.center)
            }

            CrescendButton("Open MIDI File", style: .primary, icon: "folder") {
                viewModel.showFilePicker = true
            }
        }
    }

    private var playerSection: some View {
        VStack(spacing: CrescendSpacing.space6) {
            // File info
            if let name = viewModel.engine.loadedFileName {
                Text(name)
                    .font(CrescendFont.headingLG())
                    .foregroundStyle(CrescendColor.foreground)
                    .lineLimit(2)
                    .multilineTextAlignment(.center)
            }

            // Transport
            CrescendCard {
                VStack(spacing: CrescendSpacing.space4) {
                    // Progress bar
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 2)
                                .fill(CrescendColor.subtleFill)
                                .frame(height: 4)

                            RoundedRectangle(cornerRadius: 2)
                                .fill(CrescendColor.foreground)
                                .frame(
                                    width: viewModel.engine.duration > 0
                                        ? geometry.size.width * (viewModel.engine.currentPosition / viewModel.engine.duration)
                                        : 0,
                                    height: 4
                                )
                        }
                    }
                    .frame(height: 4)

                    // Time labels
                    HStack {
                        Text(formatTime(viewModel.engine.currentPosition))
                            .font(CrescendFont.labelSM())
                            .foregroundStyle(CrescendColor.secondaryText)
                            .monospacedDigit()

                        Spacer()

                        Text(formatTime(viewModel.engine.duration))
                            .font(CrescendFont.labelSM())
                            .foregroundStyle(CrescendColor.secondaryText)
                            .monospacedDigit()
                    }

                    // Controls
                    HStack(spacing: CrescendSpacing.space6) {
                        CrescendIconButton(icon: "stop.fill") {
                            viewModel.engine.stop()
                        }

                        Button(action: {
                            if viewModel.engine.isPlaying {
                                viewModel.engine.pause()
                            } else {
                                viewModel.engine.play()
                            }
                        }) {
                            Circle()
                                .fill(CrescendColor.foreground)
                                .frame(width: 56, height: 56)
                                .overlay(
                                    Image(systemName: viewModel.engine.isPlaying ? "pause.fill" : "play.fill")
                                        .font(.system(size: 22, weight: .medium))
                                        .foregroundStyle(CrescendColor.background)
                                )
                        }
                        .buttonStyle(MIDIPressStyle())

                        CrescendIconButton(icon: "folder") {
                            viewModel.showFilePicker = true
                        }
                    }
                }
            }
        }
    }

    private func formatTime(_ seconds: TimeInterval) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", mins, secs)
    }
}

private struct MIDIPressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.92 : 1.0)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

#Preview {
    NavigationStack {
        MIDIPlayerView()
    }
    .crescendTheme()
}
