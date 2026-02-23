import SwiftUI

struct TextBlockView: View {
    let text: String

    var body: some View {
        Text(text)
            .font(CrescendFont.bodyMD())
            .foregroundStyle(CrescendColor.foreground)
            .fixedSize(horizontal: false, vertical: true)
    }
}

struct DimensionCardView: View {
    let label: String
    let score: Double
    let interpretation: String

    var body: some View {
        VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
            HStack {
                Text(label)
                    .font(CrescendFont.labelLG())
                    .foregroundStyle(CrescendColor.foreground)

                Spacer()

                Text(String(format: "%.1f / 10", score))
                    .font(CrescendFont.headingMD())
                    .foregroundStyle(CrescendColor.foreground)
                    .monospacedDigit()
            }

            // Score bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3)
                        .fill(CrescendColor.subtleFill)

                    RoundedRectangle(cornerRadius: 3)
                        .fill(CrescendColor.foreground.opacity(0.7))
                        .frame(width: max(0, geometry.size.width * min(1, score / 10.0)))
                }
            }
            .frame(height: 8)

            Text(interpretation)
                .font(CrescendFont.bodySM())
                .foregroundStyle(CrescendColor.secondaryText)
        }
        .padding(CrescendSpacing.space3)
        .background(CrescendColor.subtleFill)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct ReferencePlaybackView: View {
    let label: String
    let audioFileName: String

    @StateObject private var player = AudioPlayer()
    @State private var loadError = false

    var body: some View {
        VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
            HStack(spacing: CrescendSpacing.space3) {
                Button(action: togglePlayback) {
                    Image(systemName: player.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 36, weight: .medium))
                        .foregroundStyle(CrescendColor.foreground)
                }
                .buttonStyle(.plain)

                VStack(alignment: .leading, spacing: CrescendSpacing.space1) {
                    Text(label)
                        .font(CrescendFont.labelLG())
                        .foregroundStyle(CrescendColor.foreground)

                    if player.duration > 0 {
                        Text(formatTime(player.currentTime) + " / " + formatTime(player.duration))
                            .font(CrescendFont.labelSM())
                            .foregroundStyle(CrescendColor.secondaryText)
                            .monospacedDigit()
                    }
                }

                Spacer()
            }

            if player.duration > 0 {
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(CrescendColor.subtleFill)

                        RoundedRectangle(cornerRadius: 2)
                            .fill(CrescendColor.foreground)
                            .frame(width: max(0, geometry.size.width * (player.currentTime / player.duration)))
                    }
                }
                .frame(height: 4)
            }
        }
        .padding(CrescendSpacing.space3)
        .background(CrescendColor.subtleFill)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .onAppear {
            loadAudio()
        }
    }

    private func togglePlayback() {
        if player.isPlaying {
            player.pause()
        } else {
            player.play()
        }
    }

    private func loadAudio() {
        guard let url = Bundle.main.url(forResource: audioFileName, withExtension: "m4a")
            ?? Bundle.main.url(forResource: audioFileName, withExtension: "mp3") else {
            loadError = true
            return
        }
        do {
            try player.load(url: url)
        } catch {
            loadError = true
        }
    }

    private func formatTime(_ seconds: TimeInterval) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", mins, secs)
    }
}

struct MusicSnippetView: View {
    let imageAssetName: String
    let caption: String

    var body: some View {
        VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
            Image(imageAssetName)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: 6))

            Text(caption)
                .font(CrescendFont.labelSM())
                .foregroundStyle(CrescendColor.secondaryText)
        }
    }
}
