import SwiftUI

struct FeedbackChatView: View {
    @Bindable var viewModel: FeedbackChatViewModel

    var body: some View {
        VStack(spacing: 0) {
            // Header with drag indicator and optional focus mode label
            chatHeader

            Divider()
                .foregroundStyle(CrescendColor.border)

            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: CrescendSpacing.space4) {
                        ForEach(viewModel.messages) { message in
                            ChatMessageView(message: message)
                                .id(message.id)
                        }
                    }
                    .padding(.horizontal, CrescendSpacing.space4)
                    .padding(.vertical, CrescendSpacing.space4)
                }
                .onChange(of: viewModel.messages.count) { _, _ in
                    if let lastMessage = viewModel.messages.last {
                        withAnimation {
                            proxy.scrollTo(lastMessage.id, anchor: .bottom)
                        }
                    }
                }
            }

            Divider()
                .foregroundStyle(CrescendColor.border)

            // Suggestion chips
            if !viewModel.chips.isEmpty {
                SuggestionChipsView(chips: viewModel.chips) { chip in
                    viewModel.handleChip(chip)
                }
                .padding(.vertical, CrescendSpacing.space3)
            }
        }
        .background(CrescendColor.background)
    }

    private var chatHeader: some View {
        VStack(spacing: CrescendSpacing.space2) {
            // Drag indicator
            RoundedRectangle(cornerRadius: 2.5)
                .fill(CrescendColor.border)
                .frame(width: 36, height: 5)
                .padding(.top, CrescendSpacing.space2)

            if let focus = viewModel.focusDimension {
                HStack(spacing: CrescendSpacing.space2) {
                    Circle()
                        .fill(CrescendColor.foreground)
                        .frame(width: 8, height: 8)

                    Text("Focus: \(focus.rawValue)")
                        .font(CrescendFont.labelLG())
                        .foregroundStyle(CrescendColor.foreground)
                }
                .padding(.bottom, CrescendSpacing.space2)
            }
        }
    }
}
