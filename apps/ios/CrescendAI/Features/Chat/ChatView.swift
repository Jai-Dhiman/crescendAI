import SwiftData
import SwiftUI

struct ChatView: View {
    @Environment(\.modelContext) private var modelContext
    @State private var viewModel = ChatViewModel()

    var body: some View {
        VStack(spacing: 0) {
            if viewModel.isIdle {
                idleGreeting
            } else {
                chatStream
            }

            DualModeBottomBar(
                text: $viewModel.inputText,
                isRecording: viewModel.practiceState == .recording,
                waveformLevel: viewModel.currentLevel,
                sessionDuration: viewModel.sessionDuration,
                onSend: { viewModel.sendMessage() },
                onStartPractice: { Task { await viewModel.startPractice() } },
                onPause: { viewModel.pausePractice() },
                onStop: { Task { await viewModel.stopPractice() } },
                onAsk: { viewModel.askForFeedback() }
            )
        }
        .background(CrescendColor.background)
        .onAppear { viewModel.configure(modelContext: modelContext) }
        .navigationDestination(isPresented: $viewModel.showSessionReview) {
            SessionReviewView(messages: viewModel.messages.filter { $0.role == .observation })
        }
    }

    private var idleGreeting: some View {
        VStack(spacing: CrescendSpacing.space6) {
            Spacer()

            VStack(spacing: CrescendSpacing.space2) {
                Text("How's practice going?")
                    .font(CrescendFont.headingXL())
                    .foregroundStyle(CrescendColor.foreground)
                    .multilineTextAlignment(.center)

                Text(viewModel.greeting)
                    .font(CrescendFont.bodyMD())
                    .foregroundStyle(CrescendColor.secondaryText)
            }

            VStack(spacing: CrescendSpacing.space3) {
                CrescendButton("Start practicing", style: .primary, icon: "music.note") {
                    Task { await viewModel.startPractice() }
                }

                CrescendButton("Review last session", style: .secondary, icon: "clock") {
                    viewModel.showSessionReview = true
                }

                CrescendButton("Ask a question", style: .secondary, icon: "bubble.left") {
                    // Focus the text field -- handled by scrolling to bottom
                }
            }
            .padding(.horizontal, CrescendSpacing.space8)

            Spacer()
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var chatStream: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: CrescendSpacing.space4) {
                    ForEach(viewModel.messages) { message in
                        chatRow(for: message)
                            .id(message.id)
                            .transition(.asymmetric(
                                insertion: .move(edge: .bottom).combined(with: .opacity),
                                removal: .opacity
                            ))
                    }

                    if viewModel.practiceState == .processing {
                        processingIndicator
                    }
                }
                .padding(.horizontal, CrescendSpacing.space4)
                .padding(.vertical, CrescendSpacing.space4)
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: viewModel.messages.count) {
                withAnimation(.easeOut(duration: 0.3)) {
                    proxy.scrollTo(viewModel.messages.last?.id, anchor: .bottom)
                }
            }
        }
    }

    @ViewBuilder
    private func chatRow(for message: ChatMessage) -> some View {
        switch message.role {
        case .user:
            MessageBubble(text: message.text, isUser: true)
        case .system:
            MessageBubble(text: message.text, isUser: false)
        case .observation:
            ObservationCard(
                text: message.text,
                dimension: message.dimension ?? "dynamics",
                timestamp: message.timestamp,
                elaboration: message.elaboration,
                onTellMeMore: { viewModel.requestElaboration(for: message.id) }
            )
        }
    }

    private var processingIndicator: some View {
        HStack(spacing: CrescendSpacing.space2) {
            ProgressView()
                .tint(CrescendColor.secondaryText)
                .scaleEffect(0.8)
            Text("Thinking...")
                .font(CrescendFont.bodySM())
                .foregroundStyle(CrescendColor.secondaryText)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, CrescendSpacing.space2)
    }
}

#Preview {
    NavigationStack {
        ChatView()
    }
    .crescendTheme()
    .modelContainer(
        for: [Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self],
        inMemory: true
    )
}
