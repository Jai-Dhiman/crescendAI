import SwiftUI

struct ChatMessageView: View {
    let message: ChatMessage

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if message.sender == .user {
                Spacer(minLength: 60)
            }

            VStack(alignment: message.sender == .ai ? .leading : .trailing, spacing: CrescendSpacing.space3) {
                ForEach(message.blocks) { block in
                    blockView(for: block)
                }
            }
            .padding(CrescendSpacing.space4)
            .background(message.sender == .ai ? Color.clear : CrescendColor.subtleFill)
            .clipShape(RoundedRectangle(cornerRadius: 12))

            if message.sender == .ai {
                Spacer(minLength: 40)
            }
        }
    }

    @ViewBuilder
    private func blockView(for block: ChatBlock) -> some View {
        switch block {
        case .text(let text):
            TextBlockView(text: text)
        case .dimensionCard(let label, let score, let interpretation):
            DimensionCardView(label: label, score: score, interpretation: interpretation)
        case .referencePlayback(let label, let audioFileName):
            ReferencePlaybackView(label: label, audioFileName: audioFileName)
        case .musicSnippet(let imageAssetName, let caption):
            MusicSnippetView(imageAssetName: imageAssetName, caption: caption)
        }
    }
}
