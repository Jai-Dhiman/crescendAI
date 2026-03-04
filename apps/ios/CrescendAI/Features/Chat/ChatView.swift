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

            if viewModel.practiceState == .recording {
                PracticeControlBar(
                    onPause: { viewModel.pausePractice() },
                    onStop: { Task { await viewModel.stopPractice() } },
                    onAsk: { viewModel.askForFeedback() }
                )
            }

            ChatInputBar(
                text: $viewModel.inputText,
                onSend: { viewModel.sendMessage() },
                onMicTap: { Task { await viewModel.startPractice() } }
            )
        }
        .background(CrescendColor.background)
        .onAppear { viewModel.configure(modelContext: modelContext) }
        .sheet(isPresented: $viewModel.showSessionReview) {
            SessionReviewView(messages: viewModel.messages.filter { $0.role == .observation })
                .crescendTheme()
        }
    }

    private var idleGreeting: some View {
        VStack(spacing: CrescendSpacing.space6) {
            Spacer()

            Text(viewModel.greeting)
                .font(CrescendFont.displayLG())
                .foregroundStyle(CrescendColor.foreground)

            CrescendButton("Start Practice", style: .primary, icon: "music.note") {
                Task { await viewModel.startPractice() }
            }
            .padding(.horizontal, CrescendSpacing.space16)

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

                    if viewModel.practiceState == .recording {
                        WaveformView(level: viewModel.currentLevel, duration: viewModel.sessionDuration)
                            .padding(.vertical, CrescendSpacing.space6)
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
    ChatView()
        .crescendTheme()
        .modelContainer(
            for: [Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self],
            inMemory: true
        )
}
