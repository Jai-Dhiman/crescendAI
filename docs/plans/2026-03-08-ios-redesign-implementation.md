# iOS Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the iOS app shell (navigation, sidebar, login, bottom bar) with a Claude-style drawer sidebar, real photo login, dual-mode bottom bar, and NavigationStack navigation -- while reusing existing inner views.

**Architecture:** Hybrid rebuild. New navigation shell wraps existing feature views. Typography tokens updated globally (Lora headings, SF Pro body). Image5.jpg from web app added to asset catalog.

**Tech Stack:** SwiftUI (iOS 17+), @Observable, NavigationStack, SwiftData, AVAudioEngine

---

### Task 1: Update Typography Tokens (Lora headings, SF Pro body)

**Files:**
- Modify: `apps/ios/CrescendAI/DesignSystem/Tokens/Typography.swift`

**Step 1: Update body and label font definitions to use system font**

Replace the Body and Label sections. Display and Heading stay as Lora.

```swift
// MARK: - Body

/// 18pt SF Pro Regular
static func bodyLG(_ weight: Font.Weight = .regular) -> Font {
    .system(size: 18, weight: weight)
}

/// 16pt SF Pro Regular
static func bodyMD(_ weight: Font.Weight = .regular) -> Font {
    .system(size: 16, weight: weight)
}

/// 14pt SF Pro Regular
static func bodySM(_ weight: Font.Weight = .regular) -> Font {
    .system(size: 14, weight: weight)
}

// MARK: - Label

/// 14pt SF Pro Medium
static func labelLG(_ weight: Font.Weight = .medium) -> Font {
    .system(size: 14, weight: weight)
}

/// 12pt SF Pro Medium
static func labelMD(_ weight: Font.Weight = .medium) -> Font {
    .system(size: 12, weight: weight)
}

/// 11pt SF Pro Medium
static func labelSM(_ weight: Font.Weight = .medium) -> Font {
    .system(size: 11, weight: weight)
}
```

**Step 2: Build the project to verify no compile errors**

Run: `xcodebuild -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add apps/ios/CrescendAI/DesignSystem/Tokens/Typography.swift
git commit -m "refactor(ios): split typography -- Lora headings, SF Pro body/labels"
```

---

### Task 2: Add Image5.jpg to iOS Asset Catalog

**Files:**
- Source: `apps/web/public/Image5.jpg`
- Create: `apps/ios/CrescendAI/Resources/Assets.xcassets/Image5.imageset/` (Contents.json + image file)

**Step 1: Create the asset catalog imageset**

```bash
mkdir -p apps/ios/CrescendAI/Resources/Assets.xcassets/Image5.imageset
cp apps/web/public/Image5.jpg apps/ios/CrescendAI/Resources/Assets.xcassets/Image5.imageset/Image5.jpg
```

**Step 2: Create Contents.json for the imageset**

Write `apps/ios/CrescendAI/Resources/Assets.xcassets/Image5.imageset/Contents.json`:

```json
{
  "images": [
    {
      "filename": "Image5.jpg",
      "idiom": "universal",
      "scale": "1x"
    },
    {
      "idiom": "universal",
      "scale": "2x"
    },
    {
      "idiom": "universal",
      "scale": "3x"
    }
  ],
  "info": {
    "author": "xcode",
    "version": 1
  }
}
```

**Step 3: Verify the image loads in Swift**

The image should be accessible as `Image("Image5")` in SwiftUI. Build to verify no asset catalog errors.

Run: `xcodebuild -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 4: Commit**

```bash
git add apps/ios/CrescendAI/Resources/Assets.xcassets/Image5.imageset/
git commit -m "asset(ios): add Image5.jpg piano background for login screen"
```

---

### Task 3: Rewrite SignInView with Real Photo Background

**Files:**
- Modify: `apps/ios/CrescendAI/Features/Auth/SignInView.swift`

**Step 1: Replace the RadialGradient with Image5.jpg + gradient overlay**

Rewrite `SignInView.swift`:

```swift
import AuthenticationServices
import SwiftUI

struct SignInView: View {
    let authService: AuthService
    @State private var error: String?
    @State private var isLoading = false
    @State private var cardOpacity = 0.0

    var body: some View {
        ZStack {
            // Full-bleed photo background
            Image("Image5")
                .resizable()
                .aspectRatio(contentMode: .fill)
                .ignoresSafeArea()

            // Gradient overlay matching web app treatment
            RadialGradient(
                colors: [
                    Color(red: 0x2D / 255.0, green: 0x29 / 255.0, blue: 0x26 / 255.0, opacity: 0.4),
                    Color(red: 0x2D / 255.0, green: 0x29 / 255.0, blue: 0x26 / 255.0, opacity: 0.85),
                ],
                center: .center,
                startRadius: 50,
                endRadius: 400
            )
            .ignoresSafeArea()

            // Floating sign-in card
            VStack(spacing: CrescendSpacing.space6) {
                VStack(spacing: CrescendSpacing.space3) {
                    Text("CrescendAI")
                        .font(CrescendFont.displayXL())
                        .foregroundStyle(CrescendColor.foreground)

                    Text("A teacher for every pianist.")
                        .font(CrescendFont.bodyLG())
                        .foregroundStyle(CrescendColor.secondaryText)
                }

                VStack(spacing: CrescendSpacing.space3) {
                    SignInWithAppleButton(.signIn) { request in
                        request.requestedScopes = [.email]
                    } onCompletion: { result in
                        Task {
                            isLoading = true
                            error = nil
                            do {
                                try await authService.handleAuthorization(result: result)
                            } catch {
                                self.error = error.localizedDescription
                            }
                            isLoading = false
                        }
                    }
                    .signInWithAppleButtonStyle(.white)
                    .frame(height: 50)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .disabled(isLoading)

                    if isLoading {
                        ProgressView()
                            .tint(CrescendColor.foreground)
                    }

                    if let error {
                        Text(error)
                            .font(CrescendFont.bodySM())
                            .foregroundStyle(.red.opacity(0.8))
                            .multilineTextAlignment(.center)
                    }
                }
            }
            .padding(CrescendSpacing.space8)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(CrescendColor.border, lineWidth: 1)
            )
            .padding(.horizontal, CrescendSpacing.space6)
            .opacity(cardOpacity)
        }
        .onAppear {
            withAnimation(.easeOut(duration: 0.6).delay(0.2)) {
                cardOpacity = 1.0
            }
        }
    }
}

#Preview {
    SignInView(authService: AuthService())
        .crescendTheme()
}
```

**Step 2: Build and verify**

Run: `xcodebuild -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add apps/ios/CrescendAI/Features/Auth/SignInView.swift
git commit -m "feat(ios): real photo background on login screen with gradient overlay"
```

---

### Task 4: Build the Sidebar Drawer

**Files:**
- Rewrite: `apps/ios/CrescendAI/App/SidebarView.swift`

**Step 1: Rewrite SidebarView as a full drawer overlay**

The new SidebarView is a 280pt drawer with session list, profile footer, and overlay dismiss. It receives bindings for open/close state and callbacks for navigation.

```swift
import SwiftUI

struct SidebarView: View {
    @Binding var isOpen: Bool
    let onNewSession: () -> Void
    let onSelectSession: (UUID) -> Void
    let onShowProfile: () -> Void

    // Placeholder session data (will connect to SwiftData later)
    private let sessions: [(id: UUID, title: String, date: String)] = [
        (UUID(), "Chopin Nocturne Op.9 No.2", "Today"),
        (UUID(), "Bach Prelude in C", "Yesterday"),
        (UUID(), "Debussy Clair de Lune", "Mar 5"),
    ]

    var body: some View {
        ZStack(alignment: .leading) {
            // Dimmed background overlay
            if isOpen {
                Color.black.opacity(0.4)
                    .ignoresSafeArea()
                    .onTapGesture { closeSidebar() }
                    .transition(.opacity)
            }

            // Drawer panel
            if isOpen {
                drawerContent
                    .frame(width: 280)
                    .frame(maxHeight: .infinity)
                    .background(CrescendColor.sidebarBackground)
                    .transition(.move(edge: .leading))
            }
        }
        .animation(.easeInOut(duration: 0.25), value: isOpen)
        .gesture(
            DragGesture()
                .onEnded { value in
                    if value.translation.width < -50 {
                        closeSidebar()
                    }
                }
        )
    }

    private var drawerContent: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Text("CrescendAI")
                    .font(CrescendFont.displayMD())
                    .foregroundStyle(CrescendColor.foreground)

                Spacer()

                Button(action: onNewSession) {
                    Image(systemName: "square.and.pencil")
                        .font(.system(size: 18, weight: .medium))
                        .foregroundStyle(CrescendColor.foreground)
                        .frame(width: 36, height: 36)
                        .contentShape(Rectangle())
                }
                .buttonStyle(CrescendPressStyle())
            }
            .padding(.horizontal, CrescendSpacing.space4)
            .padding(.top, CrescendSpacing.space4)
            .padding(.bottom, CrescendSpacing.space6)

            // Session list
            ScrollView {
                LazyVStack(alignment: .leading, spacing: CrescendSpacing.space1) {
                    Text("Recents")
                        .font(CrescendFont.labelMD())
                        .foregroundStyle(CrescendColor.tertiaryText)
                        .padding(.horizontal, CrescendSpacing.space4)
                        .padding(.bottom, CrescendSpacing.space2)

                    ForEach(sessions, id: \.id) { session in
                        Button {
                            onSelectSession(session.id)
                            closeSidebar()
                        } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(session.title)
                                    .font(CrescendFont.bodyMD())
                                    .foregroundStyle(CrescendColor.foreground)
                                    .lineLimit(1)

                                Text(session.date)
                                    .font(CrescendFont.labelSM())
                                    .foregroundStyle(CrescendColor.tertiaryText)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, CrescendSpacing.space4)
                            .padding(.vertical, CrescendSpacing.space2)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(CrescendPressStyle())
                    }
                }
            }

            Spacer()

            // Profile footer
            Button(action: {
                onShowProfile()
                closeSidebar()
            }) {
                HStack(spacing: CrescendSpacing.space3) {
                    Circle()
                        .fill(CrescendColor.surface2)
                        .frame(width: 32, height: 32)
                        .overlay {
                            Image(systemName: "person.fill")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundStyle(CrescendColor.secondaryText)
                        }

                    Text("Profile")
                        .font(CrescendFont.bodyMD())
                        .foregroundStyle(CrescendColor.foreground)

                    Spacer()
                }
                .padding(.horizontal, CrescendSpacing.space4)
                .padding(.vertical, CrescendSpacing.space3)
                .contentShape(Rectangle())
            }
            .buttonStyle(CrescendPressStyle())

            Divider()
                .overlay(CrescendColor.border)
                .padding(.horizontal, CrescendSpacing.space4)
        }
    }

    private func closeSidebar() {
        isOpen = false
    }
}

#Preview {
    ZStack {
        CrescendColor.background.ignoresSafeArea()
        SidebarView(
            isOpen: .constant(true),
            onNewSession: {},
            onSelectSession: { _ in },
            onShowProfile: {}
        )
    }
    .crescendTheme()
}
```

**Step 2: Build and verify**

Run: `xcodebuild -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add apps/ios/CrescendAI/App/SidebarView.swift
git commit -m "feat(ios): Claude-style sidebar drawer with session list and swipe dismiss"
```

---

### Task 5: Build the Dual-Mode Bottom Bar

**Files:**
- Create: `apps/ios/CrescendAI/Features/Chat/DualModeBottomBar.swift`

**Step 1: Create DualModeBottomBar with idle and recording modes**

This replaces both ChatInputBar and PracticeControlBar.

```swift
import SwiftUI

struct DualModeBottomBar: View {
    @Binding var text: String
    let isRecording: Bool
    let waveformLevel: Float
    let sessionDuration: TimeInterval
    let onSend: () -> Void
    let onStartPractice: () -> Void
    let onPause: () -> Void
    let onStop: () -> Void
    let onAsk: () -> Void

    var body: some View {
        VStack(spacing: CrescendSpacing.space2) {
            if isRecording {
                recordingMode
            } else {
                idleMode
            }
        }
        .padding(.horizontal, CrescendSpacing.space4)
        .padding(.vertical, CrescendSpacing.space2)
        .background(CrescendColor.background)
        .animation(.easeInOut(duration: 0.3), value: isRecording)
    }

    // MARK: - Idle Mode

    private var idleMode: some View {
        VStack(spacing: CrescendSpacing.space3) {
            // Start Practicing button
            Button(action: onStartPractice) {
                HStack(spacing: CrescendSpacing.space2) {
                    Image(systemName: "mic.fill")
                        .font(.system(size: 16, weight: .medium))
                    Text("Start Practicing")
                        .font(CrescendFont.labelLG())
                }
                .foregroundStyle(CrescendColor.background)
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(CrescendColor.foreground)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .buttonStyle(CrescendPressStyle())

            // Text input
            HStack(spacing: CrescendSpacing.space2) {
                TextField("Ask your teacher...", text: $text, axis: .vertical)
                    .font(CrescendFont.bodyMD())
                    .foregroundStyle(CrescendColor.foreground)
                    .lineLimit(1...4)
                    .padding(.horizontal, CrescendSpacing.space3)
                    .padding(.vertical, CrescendSpacing.space2)
                    .background(CrescendColor.inputBackground)
                    .clipShape(RoundedRectangle(cornerRadius: 20))
                    .overlay(
                        RoundedRectangle(cornerRadius: 20)
                            .stroke(CrescendColor.border, lineWidth: 1)
                    )
                    .onSubmit { onSend() }

                if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Button(action: onSend) {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundStyle(CrescendColor.background)
                            .frame(width: 36, height: 36)
                            .background(CrescendColor.foreground)
                            .clipShape(Circle())
                    }
                    .buttonStyle(CrescendPressStyle())
                }
            }
        }
    }

    // MARK: - Recording Mode

    private var recordingMode: some View {
        VStack(spacing: CrescendSpacing.space3) {
            // Duration label
            Text(formatDuration(sessionDuration))
                .font(CrescendFont.labelMD())
                .foregroundStyle(CrescendColor.secondaryText)

            // Waveform
            WaveformView(level: waveformLevel, duration: sessionDuration)
                .frame(height: 48)

            // Controls
            HStack(spacing: CrescendSpacing.space6) {
                // Pause
                Button(action: onPause) {
                    Circle()
                        .stroke(CrescendColor.foreground, lineWidth: 1.5)
                        .frame(width: 44, height: 44)
                        .overlay {
                            Image(systemName: "pause.fill")
                                .font(.system(size: 16, weight: .medium))
                                .foregroundStyle(CrescendColor.foreground)
                        }
                }
                .buttonStyle(CrescendPressStyle())

                // Stop
                Button(action: onStop) {
                    Circle()
                        .fill(CrescendColor.foreground)
                        .frame(width: 44, height: 44)
                        .overlay {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(CrescendColor.background)
                                .frame(width: 16, height: 16)
                        }
                }
                .buttonStyle(CrescendPressStyle())

                // How was that?
                Button(action: onAsk) {
                    Text("How was that?")
                        .font(CrescendFont.labelLG())
                        .foregroundStyle(CrescendColor.foreground)
                        .padding(.horizontal, CrescendSpacing.space4)
                        .padding(.vertical, CrescendSpacing.space2)
                        .background(CrescendColor.surface)
                        .clipShape(Capsule())
                        .overlay(
                            Capsule().stroke(CrescendColor.border, lineWidth: 1)
                        )
                }
                .buttonStyle(CrescendPressStyle())
            }
        }
        .padding(.vertical, CrescendSpacing.space2)
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", mins, secs)
    }
}

#Preview("Idle") {
    VStack {
        Spacer()
        DualModeBottomBar(
            text: .constant(""),
            isRecording: false,
            waveformLevel: 0,
            sessionDuration: 0,
            onSend: {},
            onStartPractice: {},
            onPause: {},
            onStop: {},
            onAsk: {}
        )
    }
    .background(CrescendColor.background)
    .crescendTheme()
}

#Preview("Recording") {
    VStack {
        Spacer()
        DualModeBottomBar(
            text: .constant(""),
            isRecording: true,
            waveformLevel: 0.6,
            sessionDuration: 127,
            onSend: {},
            onStartPractice: {},
            onPause: {},
            onStop: {},
            onAsk: {}
        )
    }
    .background(CrescendColor.background)
    .crescendTheme()
}
```

**Step 2: Build and verify**

Run: `xcodebuild -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add apps/ios/CrescendAI/Features/Chat/DualModeBottomBar.swift
git commit -m "feat(ios): dual-mode bottom bar with practice button and text input"
```

---

### Task 6: Update ChatView to Use DualModeBottomBar and New Empty State

**Files:**
- Modify: `apps/ios/CrescendAI/Features/Chat/ChatView.swift`

**Step 1: Rewrite ChatView**

Replace ChatInputBar + PracticeControlBar usage with DualModeBottomBar. Update the idle greeting to the new design (centered greeting + time-aware sub-greeting + quick-action chips).

```swift
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
```

**Step 2: Update ChatViewModel.greeting to be time-aware with name**

In `apps/ios/CrescendAI/Features/Chat/ChatViewModel.swift`, update the `greeting` computed property:

```swift
var greeting: String {
    let hour = Calendar.current.component(.hour, from: .now)
    let period: String
    if hour < 12 {
        period = "morning"
    } else if hour < 17 {
        period = "afternoon"
    } else {
        period = "evening"
    }
    return "Good \(period)."
}
```

This stays the same for now -- the personalized name ("Good evening, Jai") will come when the Student model is wired up. The main greeting text "How's practice going?" is now in the view itself.

**Step 3: Build and verify**

Run: `xcodebuild -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 4: Commit**

```bash
git add apps/ios/CrescendAI/Features/Chat/ChatView.swift
git commit -m "feat(ios): new empty state with greeting and quick-action chips, dual-mode bottom bar"
```

---

### Task 7: Rewrite MainView with NavigationStack + Sidebar Drawer

**Files:**
- Rewrite: `apps/ios/CrescendAI/App/MainView.swift`

**Step 1: Rewrite MainView**

Replace the HStack layout with NavigationStack + sidebar overlay. The hamburger button in the toolbar opens the sidebar.

```swift
import SwiftData
import SwiftUI

struct MainView: View {
    @Environment(AuthService.self) private var authService
    @Environment(\.modelContext) private var modelContext
    @State private var sidebarOpen = false

    var body: some View {
        NavigationStack {
            ZStack {
                ChatView()

                SidebarView(
                    isOpen: $sidebarOpen,
                    onNewSession: {
                        // TODO: reset chat to new session
                    },
                    onSelectSession: { _ in
                        // TODO: load selected session
                    },
                    onShowProfile: {
                        // NavigationStack push handled via NavigationLink or path
                    }
                )
            }
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        sidebarOpen.toggle()
                    } label: {
                        Image(systemName: "line.3.horizontal")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(CrescendColor.foreground)
                    }
                }
            }
            .toolbarBackground(CrescendColor.background, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(for: String.self) { destination in
                switch destination {
                case "profile":
                    ProfileView()
                default:
                    EmptyView()
                }
            }
        }
    }
}

#Preview {
    MainView()
        .crescendTheme()
        .environment(AuthService())
        .modelContainer(
            for: [Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self],
            inMemory: true
        )
}
```

**Step 2: Remove the SessionsListView placeholder**

Delete the `SessionsListView` struct from MainView.swift -- session history now lives in the sidebar drawer.

**Step 3: Build and verify**

Run: `xcodebuild -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 4: Commit**

```bash
git add apps/ios/CrescendAI/App/MainView.swift
git commit -m "feat(ios): NavigationStack shell with sidebar drawer and hamburger button"
```

---

### Task 8: Clean Up Old Files

**Files:**
- Delete or mark unused: `apps/ios/CrescendAI/Features/Chat/ChatInputBar.swift`
- Delete or mark unused: `apps/ios/CrescendAI/Features/Practice/PracticeControlBar.swift`

**Step 1: Remove ChatInputBar.swift**

This is fully replaced by DualModeBottomBar. Delete the file.

```bash
git rm apps/ios/CrescendAI/Features/Chat/ChatInputBar.swift
```

**Step 2: Remove PracticeControlBar.swift**

Recording controls are now in DualModeBottomBar. Delete the file.

```bash
git rm apps/ios/CrescendAI/Features/Practice/PracticeControlBar.swift
```

**Step 3: Verify no remaining references**

Search for `ChatInputBar` and `PracticeControlBar` in the codebase. They should only appear in git history, not in any Swift files.

```bash
grep -r "ChatInputBar\|PracticeControlBar" apps/ios/CrescendAI/ --include="*.swift"
```
Expected: no output

**Step 4: Build and verify**

Run: `xcodebuild -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 5: Commit**

```bash
git commit -m "chore(ios): remove ChatInputBar and PracticeControlBar, replaced by DualModeBottomBar"
```

---

### Task 9: Verify End-to-End

**Step 1: Full clean build**

```bash
xcodebuild clean build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -10
```
Expected: BUILD SUCCEEDED

**Step 2: Launch in Simulator and manually verify**

Run the app in Simulator and check:
- [ ] Login screen shows Image5.jpg with frosted card overlay
- [ ] After sign-in, main screen shows "How's practice going?" greeting with chips
- [ ] Hamburger button (top-left) opens sidebar drawer
- [ ] Sidebar shows "CrescendAI" header, session list, profile footer
- [ ] Swipe left on sidebar dismisses it
- [ ] Tap dimmed area dismisses sidebar
- [ ] "Start Practicing" button in bottom bar works
- [ ] Recording mode shows waveform + controls
- [ ] Stop button ends recording
- [ ] Body text uses SF Pro, headings use Lora
- [ ] All existing chat message bubbles and observation cards display correctly

**Step 3: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix(ios): address issues found during end-to-end verification"
```
