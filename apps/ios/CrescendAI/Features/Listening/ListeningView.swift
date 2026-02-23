import SwiftUI

struct ListeningView: View {
    @State private var viewModel = ListeningViewModel(demoMode: .firstTime)

    var body: some View {
        ZStack {
            CrescendColor.background
                .ignoresSafeArea()

            VStack(spacing: 0) {
                // Top: Wordmark + contextual prompt
                VStack(spacing: CrescendSpacing.space3) {
                    Text("CrescendAI")
                        .font(CrescendFont.headingLG())
                        .foregroundStyle(CrescendColor.foreground)

                    Text(viewModel.contextPrompt)
                        .font(CrescendFont.bodyMD())
                        .foregroundStyle(CrescendColor.secondaryText)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, CrescendSpacing.space8)
                }
                .padding(.top, CrescendSpacing.space16)

                Spacer()

                // Center: Visual pulse
                PulseView(
                    level: viewModel.audioMonitor.currentLevel,
                    isActive: !viewModel.audioMonitor.isSilent
                )
                .frame(maxWidth: .infinity)
                .frame(height: UIScreen.main.bounds.height * 0.4)

                Spacer()

                // Bottom: Listening indicator
                HStack(spacing: CrescendSpacing.space2) {
                    Image(systemName: "mic.fill")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(CrescendColor.secondaryText)

                    Text("CrescendAI is listening")
                        .font(CrescendFont.labelMD())
                        .foregroundStyle(CrescendColor.secondaryText)
                }
                .padding(.bottom, CrescendSpacing.space8)
            }
        }
        .sheet(isPresented: $viewModel.showChat) {
            FeedbackChatView(viewModel: viewModel.chatViewModel)
                .presentationDetents([.fraction(0.85)])
                .presentationDragIndicator(.hidden)
                .presentationBackground(CrescendColor.background)
                .onChange(of: viewModel.chatViewModel.shouldDismiss) { _, shouldDismiss in
                    if shouldDismiss {
                        viewModel.dismissChat()
                    }
                }
        }
        .onAppear {
            viewModel.startListening()
        }
        .onDisappear {
            viewModel.stopListening()
        }
    }
}

#Preview {
    ListeningView()
        .crescendTheme()
}
