import SwiftData
import SwiftUI

struct ChatView: View {
    @Environment(\.modelContext) private var modelContext
    @State private var viewModel = ChatViewModel()
    @FocusState private var isInputFocused: Bool
    
    var onOpenSidebar: () -> Void = {}

    var body: some View {
        VStack(spacing: 0) {
            // Chat content
            if viewModel.messages.isEmpty {
                emptyState
            } else {
                chatStream
            }
            
            // Input area
            inputArea
        }
        .background(CrescendColor.background)
        .onAppear { viewModel.configure(modelContext: modelContext) }
    }
    
    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: CrescendSpacing.space4) {
            Spacer()
            
            VStack(spacing: CrescendSpacing.space2) {
                Image("AppLogo")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 64, height: 64)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                
                Text(viewModel.greeting)
                    .font(CrescendFont.headingLG())
                    .foregroundStyle(CrescendColor.foreground)
                    .multilineTextAlignment(.center)
                
                Text("Start practicing or ask me anything")
                    .font(CrescendFont.bodyMD())
                    .foregroundStyle(CrescendColor.secondaryText)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal, CrescendSpacing.space6)
            
            Spacer()
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Chat Stream
    
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
            VStack(alignment: .leading, spacing: 0) {
                ObservationCard(
                    text: message.text,
                    dimension: message.dimension ?? "dynamics",
                    timestamp: message.timestamp,
                    elaboration: message.elaboration,
                    onTellMeMore: { viewModel.requestElaboration(for: message.id) }
                )
                if !message.artifacts.isEmpty {
                    VStack(spacing: CrescendSpacing.space2) {
                        ForEach(message.artifacts.indices, id: \.self) { i in
                            ArtifactRenderer(config: message.artifacts[i])
                        }
                    }
                    .padding(.top, CrescendSpacing.space2)
                }
            }
        case .teacher:
            VStack(alignment: .leading, spacing: 0) {
                MessageBubble(text: message.text, isUser: false)
                if !message.artifacts.isEmpty {
                    VStack(spacing: CrescendSpacing.space2) {
                        ForEach(message.artifacts.indices, id: \.self) { i in
                            ArtifactRenderer(config: message.artifacts[i])
                        }
                    }
                    .padding(.top, CrescendSpacing.space2)
                }
            }
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
    
    // MARK: - Input Area
    
    private var inputArea: some View {
        VStack(spacing: 0) {
            // Recording controls (when active)
            if viewModel.practiceState == .recording {
                recordingControls
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }

            // Main input container
            VStack(spacing: 0) {
                // Text input field
                TextField("Ask your teacher anything...", text: $viewModel.inputText, axis: .vertical)
                    .font(CrescendFont.bodyMD())
                    .foregroundStyle(CrescendColor.foreground)
                    .lineLimit(1...6)
                    .padding(.horizontal, CrescendSpacing.space4)
                    .padding(.top, CrescendSpacing.space4)
                    .frame(minHeight: 80, alignment: .top)
                    .focused($isInputFocused)
                    .disabled(viewModel.practiceState == .recording)
                    .onSubmit {
                        if !viewModel.inputText.isEmpty {
                            Task { await viewModel.sendMessage() }
                        }
                    }

                // Bottom row with action buttons
                HStack {
                    Spacer()

                    // Send button (only when there's text)
                    if !viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        Button(action: { Task { await viewModel.sendMessage() } }) {
                            Image(systemName: "arrow.up")
                                .font(.system(size: 16, weight: .semibold))
                                .foregroundStyle(CrescendColor.background)
                                .frame(width: 32, height: 32)
                                .background(CrescendColor.foreground)
                                .clipShape(Circle())
                        }
                        .buttonStyle(.plain)
                        .transition(.scale.combined(with: .opacity))
                    }
                }
                .padding(.horizontal, CrescendSpacing.space3)
                .padding(.bottom, CrescendSpacing.space3)
                .animation(.easeInOut(duration: 0.15), value: viewModel.inputText.isEmpty)
            }
            .background(CrescendColor.surface)
            .clipShape(RoundedRectangle(cornerRadius: 24))
            .overlay(
                RoundedRectangle(cornerRadius: 24)
                    .stroke(isInputFocused ? CrescendColor.foreground.opacity(0.3) : CrescendColor.border, lineWidth: 1)
            )
            .padding(.horizontal, CrescendSpacing.space4)
            .padding(.top, CrescendSpacing.space3)
            .padding(.trailing, 60) // Room for the practice button
            .overlay(alignment: .bottomTrailing) {
                // Practice / Stop button (always visible, separate from input)
                practiceButton
                    .padding(.trailing, CrescendSpacing.space4)
            }
            .padding(.bottom, CrescendSpacing.space4)
            .background(CrescendColor.background)
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: viewModel.practiceState)
    }

    @ViewBuilder
    private var practiceButton: some View {
        if viewModel.practiceState == .recording {
            Button(action: { Task { await viewModel.stopPractice() } }) {
                Image(systemName: "stop.fill")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(CrescendColor.background)
                    .frame(width: 48, height: 48)
                    .background(CrescendColor.foreground)
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
        } else {
            Button(action: { Task { await viewModel.startPractice() } }) {
                Image(systemName: "waveform")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(CrescendColor.background)
                    .frame(width: 48, height: 48)
                    .background(CrescendColor.accent)
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
        }
    }
    
    
    private var recordingControls: some View {
        VStack(spacing: CrescendSpacing.space3) {
            // Waveform visualization
            WaveformView(level: viewModel.currentLevel, duration: viewModel.sessionDuration)
                .frame(height: 60)
                .padding(.horizontal, CrescendSpacing.space4)
            
            HStack(spacing: CrescendSpacing.space4) {
                // Duration
                HStack(spacing: CrescendSpacing.space1) {
                    Image(systemName: "circle.fill")
                        .font(.system(size: 8))
                        .foregroundStyle(.red)
                    
                    Text(formatDuration(viewModel.sessionDuration))
                        .font(CrescendFont.labelMD())
                        .foregroundStyle(CrescendColor.foreground)
                        .monospacedDigit()
                }
                
                Spacer()
                
                // Ask for feedback button
                Button(action: { Task { await viewModel.askForFeedback() } }) {
                    HStack(spacing: CrescendSpacing.space2) {
                        Image(systemName: "sparkles")
                            .font(.system(size: 14, weight: .medium))
                        Text("Get Feedback")
                            .font(CrescendFont.labelMD())
                    }
                    .foregroundStyle(CrescendColor.accent)
                    .padding(.horizontal, CrescendSpacing.space3)
                    .padding(.vertical, CrescendSpacing.space2)
                    .background(CrescendColor.surface)
                    .clipShape(Capsule())
                    .overlay(
                        Capsule().stroke(CrescendColor.border, lineWidth: 1)
                    )
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, CrescendSpacing.space4)
        }
        .padding(.vertical, CrescendSpacing.space3)
        .background(CrescendColor.surface.opacity(0.5))
    }
    
    private func formatDuration(_ seconds: TimeInterval) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", mins, secs)
    }
}

#Preview {
    NavigationStack {
        ChatView()
    }
    .crescendTheme()
    .modelContainer(
        for: [Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self, ConversationRecord.self],
        inMemory: true
    )
}
