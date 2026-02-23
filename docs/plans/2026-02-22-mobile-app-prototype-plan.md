# Mobile App Prototype Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a SwiftUI prototype with two states (Listening + Feedback Chat), real audio input driving an organic visual pulse, and mocked AI feedback across 3 demo scenarios.

**Architecture:** Replace the current tab/navigation-based iOS scaffold with a two-state app: a full-screen Listening State (always-on microphone with visual pulse) that automatically transitions to a Feedback Chat (rich messages with inline elements) after 3 seconds of silence. All AI feedback is hardcoded for the prototype. Reuse the existing `AudioRecorder` for mic input and `AudioPlayer` for reference playback.

**Tech Stack:** SwiftUI (iOS 16+), AVAudioEngine (mic), AVAudioPlayer (reference playback), existing CrescendAI design system tokens (CrescendColor, CrescendFont, CrescendSpacing)

---

## Task 1: Audio Monitor -- Always-Listening Audio Engine

Refactor the existing `AudioRecorder` into a lighter `AudioMonitor` that captures audio levels continuously without writing to disk. The Listening State needs real-time amplitude, not file recording.

**Files:**
- Create: `apps/ios/CrescendAI/Features/Listening/AudioMonitor.swift`

**Step 1: Create AudioMonitor**

This is a simplified version of the existing `AudioRecorder` (at `apps/ios/CrescendAI/Features/Recording/AudioRecorder.swift`). It only calculates RMS level and detects silence -- no file writing.

```swift
import AVFoundation

@MainActor
final class AudioMonitor: ObservableObject {
    @Published private(set) var currentLevel: Float = 0
    @Published private(set) var isSilent: Bool = true
    @Published private(set) var isRunning: Bool = false

    private var audioEngine: AVAudioEngine?
    private var silenceTimer: Timer?

    /// How many seconds of silence before `isSilent` becomes true.
    var silenceThreshold: TimeInterval = 3.0

    /// Level below which audio is considered silence.
    private let silenceLevel: Float = 0.02

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
        try session.setActive(true)

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let format = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            guard let channelData = buffer.floatChannelData?[0] else { return }
            let frameLength = Int(buffer.frameLength)
            var sum: Float = 0
            for i in 0..<frameLength {
                sum += channelData[i] * channelData[i]
            }
            let rms = sqrtf(sum / Float(frameLength))
            let level = max(0, min(1, rms * 5))

            Task { @MainActor [weak self] in
                self?.updateLevel(level)
            }
        }

        try engine.start()
        self.audioEngine = engine
        self.isRunning = true
        self.isSilent = true
    }

    func stop() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        silenceTimer?.invalidate()
        silenceTimer = nil
        isRunning = false
        currentLevel = 0
        isSilent = true
    }

    private func updateLevel(_ level: Float) {
        currentLevel = level

        if level > silenceLevel {
            // Sound detected -- reset silence timer
            isSilent = false
            silenceTimer?.invalidate()
            silenceTimer = Timer.scheduledTimer(withTimeInterval: silenceThreshold, repeats: false) { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.isSilent = true
                }
            }
        }
    }
}
```

**Step 2: Verify it builds**

Run: `xcodebuild build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add apps/ios/CrescendAI/Features/Listening/AudioMonitor.swift
git commit -m "feat(ios): add AudioMonitor for always-listening audio level detection"
```

---

## Task 2: Visual Pulse Animation

Build the organic, breathing pulse that responds to audio amplitude. This replaces the existing `LevelMeterView` bar visualization.

**Files:**
- Create: `apps/ios/CrescendAI/Features/Listening/PulseView.swift`

**Step 1: Create PulseView**

The pulse is a set of concentric, organic shapes that breathe with the audio. When silent, it's a calm thin ring. When audio is detected, the shapes expand and undulate based on amplitude.

```swift
import SwiftUI

struct PulseView: View {
    /// Audio level from 0 to 1
    let level: Float
    /// Whether the monitor is currently hearing sound
    let isActive: Bool

    @State private var phase: Double = 0

    private let timer = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()

    var body: some View {
        Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let baseRadius = min(size.width, size.height) * 0.15
            let maxRadius = min(size.width, size.height) * 0.4
            let amplitude = CGFloat(level)

            // Draw 4 concentric rings with decreasing opacity
            for i in (0..<4).reversed() {
                let layerFraction = CGFloat(i) / 3.0
                let layerAmplitude = amplitude * (1.0 - layerFraction * 0.3)
                let radius = baseRadius + (maxRadius - baseRadius) * layerAmplitude
                let opacity = isActive ? (0.6 - layerFraction * 0.15) : 0.08

                var path = Path()
                let points = 120
                for j in 0..<points {
                    let angle = (Double(j) / Double(points)) * .pi * 2
                    // Organic undulation: combine multiple sine waves
                    let wave1 = sin(angle * 3 + phase + Double(i) * 0.5) * Double(layerAmplitude) * 0.12
                    let wave2 = sin(angle * 5 - phase * 0.7 + Double(i)) * Double(layerAmplitude) * 0.06
                    let r = radius * (1.0 + wave1 + wave2)

                    let x = center.x + r * cos(angle)
                    let y = center.y + r * sin(angle)

                    if j == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
                path.closeSubpath()

                context.fill(path, with: .color(CrescendColor.foreground.opacity(opacity)))
            }
        }
        .onReceive(timer) { _ in
            let speed = isActive ? 0.03 + Double(level) * 0.02 : 0.008
            phase += speed
        }
        .animation(.easeOut(duration: 0.15), value: level)
    }
}

#Preview("Pulse - Silent") {
    PulseView(level: 0, isActive: false)
        .frame(height: 300)
        .background(CrescendColor.background)
}

#Preview("Pulse - Active") {
    PulseView(level: 0.6, isActive: true)
        .frame(height: 300)
        .background(CrescendColor.background)
}
```

**Step 2: Verify it builds**

Run: `xcodebuild build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add apps/ios/CrescendAI/Features/Listening/PulseView.swift
git commit -m "feat(ios): add organic visual pulse animation driven by audio level"
```

---

## Task 3: Chat Message Models and Demo Data

Define the data models for the rich chat and pre-populate the 3 demo scenarios.

**Files:**
- Create: `apps/ios/CrescendAI/Features/Chat/ChatModels.swift`
- Create: `apps/ios/CrescendAI/Features/Chat/DemoScenarios.swift`

**Step 1: Create ChatModels**

```swift
import Foundation

enum ChatMessageSender {
    case ai
    case user
}

/// A single element within a chat message.
/// Messages are composed of one or more of these blocks.
enum ChatBlock: Identifiable {
    case text(String)
    case dimensionCard(label: String, score: Double, interpretation: String)
    case referencePlayback(label: String, audioFileName: String)
    case musicSnippet(imageAssetName: String, caption: String)

    var id: String {
        switch self {
        case .text(let s): return "text-\(s.prefix(20).hashValue)"
        case .dimensionCard(let label, _, _): return "dim-\(label)"
        case .referencePlayback(let label, _): return "ref-\(label)"
        case .musicSnippet(let name, _): return "img-\(name)"
        }
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let sender: ChatMessageSender
    let blocks: [ChatBlock]
    let timestamp = Date()
}

/// Suggestion chips shown at the bottom of the chat.
struct SuggestionChip: Identifiable, Hashable {
    let id = UUID()
    let label: String
    /// Key used to look up the action in the scenario handler.
    let actionKey: String
}

/// Focus mode targets a specific performance dimension.
enum FocusDimension: String, CaseIterable {
    case dynamics = "Dynamics"
    case articulation = "Articulation"
    case pedaling = "Pedaling"
    case timing = "Timing"
    case tone = "Tone"
}

/// Which demo scenario the app is currently running.
enum DemoMode {
    case firstTime
    case returning
}
```

**Step 2: Create DemoScenarios with hardcoded feedback**

```swift
import Foundation

enum DemoScenarios {

    // MARK: - Scenario 1: First-time user, general feedback

    static let firstTimeGreeting = ChatMessage(
        sender: .ai,
        blocks: [
            .text("Nice -- I heard something lyrical, with a gentle touch in the upper register. Here's what stood out to me:"),
            .text("Your legato phrasing is naturally musical -- the way you connected the melodic line showed real sensitivity. The dynamics could use more contrast though. The quiet sections are beautiful, but the crescendo moments don't quite arrive where they want to go."),
            .dimensionCard(label: "Dynamics", score: 5.8, interpretation: "Room to grow -- your quiet playing is expressive, but the louder passages need more commitment."),
            .text("Try exaggerating the dynamic range for a few run-throughs. It often feels like too much at the piano, but from the audience's perspective, it reads as just right."),
        ]
    )

    static let firstTimeChips: [SuggestionChip] = [
        SuggestionChip(label: "Focus on dynamics", actionKey: "start_focus_dynamics"),
        SuggestionChip(label: "What should I work on?", actionKey: "what_to_work_on"),
        SuggestionChip(label: "How does this work?", actionKey: "explain_app"),
    ]

    static let howItWorks = ChatMessage(
        sender: .ai,
        blocks: [
            .text("I listen to how you play -- not just the notes, but how they sound. Things like your dynamics, tone quality, phrasing, pedaling, and articulation."),
            .text("I can focus on one thing at a time (like just dynamics), or give you a general picture. When it helps, I can play a reference so you can hear what I'm describing."),
            .text("Just play whenever you're ready. I'm always listening."),
        ]
    )

    static let howItWorksChips: [SuggestionChip] = [
        SuggestionChip(label: "Focus on dynamics", actionKey: "start_focus_dynamics"),
        SuggestionChip(label: "Play it again", actionKey: "dismiss_chat"),
    ]

    static let whatToWorkOn = ChatMessage(
        sender: .ai,
        blocks: [
            .text("Based on what I just heard, I'd suggest focusing on dynamics first. Your phrasing instincts are strong, but the dynamic palette is narrow -- opening that up will make everything else shine more."),
            .text("Want to try a focused session on dynamics? I'll only give feedback on dynamic contrast and shaping."),
        ]
    )

    static let whatToWorkOnChips: [SuggestionChip] = [
        SuggestionChip(label: "Start focus mode", actionKey: "start_focus_dynamics"),
        SuggestionChip(label: "Try something else", actionKey: "dismiss_chat"),
    ]

    // MARK: - Scenario 2: Returning user, focus mode (dynamics)

    static let returningGreeting = "Last time: Chopin Nocturne Op. 9 No. 2 -- Bars 12-18 dynamics. Pick up where you left off?"

    static let focusDynamicsFeedback = ChatMessage(
        sender: .ai,
        blocks: [
            .text("Better! The crescendo into bar 14 has more shape now. But you're pulling back right at the peak -- try sustaining the forte for one more beat before the diminuendo."),
            .dimensionCard(label: "Dynamics", score: 6.4, interpretation: "Improving. The crescendo shape is there, but the peak needs more sustain."),
            .text("Listen to how this passage can build -- notice how the loudest moment lingers before pulling back:"),
            .referencePlayback(label: "Bars 12-18: crescendo with sustained peak", audioFileName: "reference_nocturne_bars12_18"),
        ]
    )

    static let focusDynamicsChips: [SuggestionChip] = [
        SuggestionChip(label: "Play it again", actionKey: "dismiss_chat"),
        SuggestionChip(label: "Show me another reference", actionKey: "another_reference"),
        SuggestionChip(label: "Exit focus mode", actionKey: "exit_focus"),
        SuggestionChip(label: "I'm done for today", actionKey: "done"),
    ]

    // MARK: - Scenario 3: Reference playback follow-up

    static let referenceFollowUp = ChatMessage(
        sender: .ai,
        blocks: [
            .text("Notice two things in that reference: the crescendo builds gradually over three beats (not all at once), and the peak is held with a slight tenuto before the release. That sustained moment is what gives the phrase its emotional weight."),
            .text("Try matching that shape. Gradual build, hold the peak, then release."),
        ]
    )

    static let referenceFollowUpChips: [SuggestionChip] = [
        SuggestionChip(label: "Play it again", actionKey: "dismiss_chat"),
        SuggestionChip(label: "I'm done for today", actionKey: "done"),
    ]

    // MARK: - Done

    static let doneMessage = ChatMessage(
        sender: .ai,
        blocks: [
            .text("Good session. Your dynamics are moving in the right direction -- the crescendo shaping is noticeably better than when we started. Next time, we can pick up right where you left off."),
        ]
    )

    static let doneChips: [SuggestionChip] = [
        SuggestionChip(label: "Play one more time", actionKey: "dismiss_chat"),
    ]
}
```

**Step 3: Verify it builds**

Run: `xcodebuild build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 4: Commit**

```bash
git add apps/ios/CrescendAI/Features/Chat/ChatModels.swift apps/ios/CrescendAI/Features/Chat/DemoScenarios.swift
git commit -m "feat(ios): add chat message models and hardcoded demo scenario data"
```

---

## Task 4: Chat View -- Message Rendering

Build the chat UI that renders rich messages with inline elements.

**Files:**
- Create: `apps/ios/CrescendAI/Features/Chat/ChatMessageView.swift`
- Create: `apps/ios/CrescendAI/Features/Chat/ChatBlockViews.swift`

**Step 1: Create ChatBlockViews -- individual block renderers**

```swift
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
```

**Step 2: Create ChatMessageView -- composes blocks into a message bubble**

```swift
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
```

**Step 3: Verify it builds**

Run: `xcodebuild build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 4: Commit**

```bash
git add apps/ios/CrescendAI/Features/Chat/ChatBlockViews.swift apps/ios/CrescendAI/Features/Chat/ChatMessageView.swift
git commit -m "feat(ios): add rich chat message rendering with inline dimension cards and reference playback"
```

---

## Task 5: Suggestion Chips Component

Build the horizontally scrollable suggestion chip bar.

**Files:**
- Create: `apps/ios/CrescendAI/Features/Chat/SuggestionChipsView.swift`

**Step 1: Create SuggestionChipsView**

```swift
import SwiftUI

struct SuggestionChipsView: View {
    let chips: [SuggestionChip]
    let onTap: (SuggestionChip) -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: CrescendSpacing.space2) {
                ForEach(chips) { chip in
                    Button(action: { onTap(chip) }) {
                        Text(chip.label)
                            .font(CrescendFont.labelLG())
                            .foregroundStyle(CrescendColor.foreground)
                            .padding(.horizontal, CrescendSpacing.space4)
                            .padding(.vertical, CrescendSpacing.space2)
                            .background(CrescendColor.subtleFill)
                            .clipShape(Capsule())
                            .overlay(
                                Capsule()
                                    .stroke(CrescendColor.border, lineWidth: 1)
                            )
                    }
                    .buttonStyle(ChipPressStyle())
                }
            }
            .padding(.horizontal, CrescendSpacing.space4)
        }
    }
}

private struct ChipPressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            .opacity(configuration.isPressed ? 0.7 : 1.0)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

#Preview {
    VStack {
        Spacer()
        SuggestionChipsView(
            chips: [
                SuggestionChip(label: "Focus on dynamics", actionKey: "focus"),
                SuggestionChip(label: "Play it again", actionKey: "play"),
                SuggestionChip(label: "How does this work?", actionKey: "help"),
            ],
            onTap: { _ in }
        )
    }
    .background(CrescendColor.background)
}
```

**Step 2: Verify it builds**

Run: `xcodebuild build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add apps/ios/CrescendAI/Features/Chat/SuggestionChipsView.swift
git commit -m "feat(ios): add horizontally scrollable suggestion chips component"
```

---

## Task 6: Chat View Model -- Scenario State Machine

Build the view model that drives the chat, manages demo scenario state, and handles chip actions.

**Files:**
- Create: `apps/ios/CrescendAI/Features/Chat/FeedbackChatViewModel.swift`

**Step 1: Create FeedbackChatViewModel**

```swift
import SwiftUI

@MainActor
@Observable
final class FeedbackChatViewModel {
    private(set) var messages: [ChatMessage] = []
    private(set) var chips: [SuggestionChip] = []
    private(set) var focusDimension: FocusDimension?
    private(set) var shouldDismiss = false

    var isInFocusMode: Bool { focusDimension != nil }
    let demoMode: DemoMode

    /// Tracks which responses have been shown to avoid repeats.
    private var shownResponses: Set<String> = []

    init(demoMode: DemoMode) {
        self.demoMode = demoMode
    }

    /// Called when the user stops playing and the chat appears.
    func onPerformanceEnded() {
        if isInFocusMode {
            appendMessage(DemoScenarios.focusDynamicsFeedback)
            chips = DemoScenarios.focusDynamicsChips
        } else if !shownResponses.contains("firstGreeting") {
            appendMessage(DemoScenarios.firstTimeGreeting)
            chips = DemoScenarios.firstTimeChips
            shownResponses.insert("firstGreeting")
        } else {
            // Subsequent general plays after the first -- re-show first greeting
            appendMessage(DemoScenarios.firstTimeGreeting)
            chips = DemoScenarios.firstTimeChips
        }
    }

    func handleChip(_ chip: SuggestionChip) {
        // Show user's selection as a message
        appendMessage(ChatMessage(sender: .user, blocks: [.text(chip.label)]))

        switch chip.actionKey {
        case "start_focus_dynamics":
            focusDimension = .dynamics
            appendMessage(ChatMessage(sender: .ai, blocks: [
                .text("Focus mode: Dynamics. I'll only give feedback on dynamic contrast and shaping. Play when you're ready."),
            ]))
            chips = [SuggestionChip(label: "Play now", actionKey: "dismiss_chat")]

        case "explain_app":
            appendMessage(DemoScenarios.howItWorks)
            chips = DemoScenarios.howItWorksChips

        case "what_to_work_on":
            appendMessage(DemoScenarios.whatToWorkOn)
            chips = DemoScenarios.whatToWorkOnChips

        case "another_reference":
            appendMessage(DemoScenarios.referenceFollowUp)
            chips = DemoScenarios.referenceFollowUpChips

        case "exit_focus":
            focusDimension = nil
            appendMessage(ChatMessage(sender: .ai, blocks: [
                .text("Exiting focus mode. I'll give general feedback on your next play-through."),
            ]))
            chips = [SuggestionChip(label: "Play now", actionKey: "dismiss_chat")]

        case "done":
            appendMessage(DemoScenarios.doneMessage)
            chips = DemoScenarios.doneChips

        case "dismiss_chat":
            shouldDismiss = true

        default:
            break
        }
    }

    func resetDismiss() {
        shouldDismiss = false
    }

    func clearMessages() {
        messages.removeAll()
        chips = []
    }

    private func appendMessage(_ message: ChatMessage) {
        messages.append(message)
    }
}
```

**Step 2: Verify it builds**

Run: `xcodebuild build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add apps/ios/CrescendAI/Features/Chat/FeedbackChatViewModel.swift
git commit -m "feat(ios): add chat view model with demo scenario state machine"
```

---

## Task 7: Feedback Chat Sheet View

Build the chat sheet that slides up from the bottom.

**Files:**
- Create: `apps/ios/CrescendAI/Features/Chat/FeedbackChatView.swift`

**Step 1: Create FeedbackChatView**

```swift
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
```

**Step 2: Verify it builds**

Run: `xcodebuild build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add apps/ios/CrescendAI/Features/Chat/FeedbackChatView.swift
git commit -m "feat(ios): add feedback chat sheet with message list and suggestion chips"
```

---

## Task 8: Listening View -- The Main Screen

Build the full-screen listening state that wires together the audio monitor, visual pulse, and chat sheet.

**Files:**
- Create: `apps/ios/CrescendAI/Features/Listening/ListeningView.swift`
- Create: `apps/ios/CrescendAI/Features/Listening/ListeningViewModel.swift`

**Step 1: Create ListeningViewModel**

```swift
import SwiftUI
import Combine

@MainActor
@Observable
final class ListeningViewModel {
    let audioMonitor = AudioMonitor()
    let chatViewModel: FeedbackChatViewModel

    private(set) var showChat = false
    private(set) var hasHeardAudio = false

    /// Tracks whether the user has played at least once before the chat triggers.
    private var wasPlaying = false

    private var cancellables = Set<AnyCancellable>()

    let demoMode: DemoMode

    var contextPrompt: String {
        if let focus = chatViewModel.focusDimension {
            return "Focusing on: \(focus.rawValue)"
        }
        switch demoMode {
        case .firstTime:
            return "Play anything -- I'm listening."
        case .returning:
            return DemoScenarios.returningGreeting
        }
    }

    init(demoMode: DemoMode = .firstTime) {
        self.demoMode = demoMode
        self.chatViewModel = FeedbackChatViewModel(demoMode: demoMode)
        observeAudioState()
    }

    func startListening() {
        do {
            try audioMonitor.start()
        } catch {
            // In prototype, we log but don't block the UI
            print("AudioMonitor failed to start: \(error)")
        }
    }

    func stopListening() {
        audioMonitor.stop()
    }

    func dismissChat() {
        showChat = false
        chatViewModel.resetDismiss()
    }

    private func observeAudioState() {
        // Watch for silence transitions: playing -> silent = trigger chat
        audioMonitor.$isSilent
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isSilent in
                guard let self else { return }
                if !isSilent {
                    self.wasPlaying = true
                    self.hasHeardAudio = true
                } else if isSilent && self.wasPlaying {
                    // Silence after playing -- show feedback
                    self.wasPlaying = false
                    self.triggerFeedback()
                }
            }
            .store(in: &cancellables)

        // Watch for chat dismiss requests from chip actions
        chatViewModel.$shouldDismiss
            .receive(on: DispatchQueue.main)
            .sink { [weak self] shouldDismiss in
                guard let self, shouldDismiss else { return }
                self.dismissChat()
            }
            .store(in: &cancellables)
    }

    private func triggerFeedback() {
        chatViewModel.clearMessages()
        chatViewModel.onPerformanceEnded()
        showChat = true
    }
}
```

Note: `ListeningViewModel` uses Combine to observe `AudioMonitor`'s `@Published` properties. The `$isSilent` publisher fires when the user transitions from playing to silent, which triggers the chat.

There is an issue -- `@Observable` and `$` publisher syntax: `AudioMonitor` uses `ObservableObject`/`@Published`, so `$isSilent` works. If there are compile issues with mixing `@Observable` and `ObservableObject`, change `ListeningViewModel` to also use `ObservableObject`/`@Published`. The existing codebase uses both patterns.

Also note: `FeedbackChatViewModel` uses `@Observable`, so `$shouldDismiss` won't produce a Combine publisher. To fix this, either:
- Make `shouldDismiss` on `FeedbackChatViewModel` a manual Combine publisher, OR
- Use a simpler polling approach with `onChange` in the view

The simpler fix: use `onChange` in the view instead. Adjust `ListeningViewModel` to remove the `shouldDismiss` observation and handle it in the view layer. See Task 8 Step 2 for the view approach.

**Step 2: Create ListeningView**

```swift
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
```

**Step 3: Verify it builds**

Run: `xcodebuild build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 4: Commit**

```bash
git add apps/ios/CrescendAI/Features/Listening/ListeningView.swift apps/ios/CrescendAI/Features/Listening/ListeningViewModel.swift
git commit -m "feat(ios): add listening view with pulse animation and chat sheet integration"
```

---

## Task 9: Wire Up App Entry Point

Replace the current HomeView-based navigation with the new ListeningView as the app's root.

**Files:**
- Modify: `apps/ios/CrescendAI/App/ContentView.swift`

**Step 1: Update ContentView to use ListeningView**

Replace the entire contents of `ContentView.swift`:

```swift
import SwiftUI

struct ContentView: View {
    var body: some View {
        ListeningView()
    }
}

#Preview {
    ContentView()
        .crescendTheme()
}
```

The old `HomeView`, `RecordingView`, and navigation stack are no longer the entry point. They remain in the codebase for reference but are not loaded.

**Step 2: Verify it builds**

Run: `xcodebuild build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add apps/ios/CrescendAI/App/ContentView.swift
git commit -m "feat(ios): replace home/nav entry point with listening view as app root"
```

---

## Task 10: Fix Compile Issues and Integration Test

With all files in place, do a full build, fix any compile errors, and verify the prototype runs in the simulator.

**Files:**
- May need to adjust: Any of the files from Tasks 1-9

**Step 1: Full build**

Run: `xcodebuild build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -30`

Fix any compile errors. Common issues to watch for:
- `@Observable` + `@Bindable` requires iOS 17+ -- check the deployment target in the Xcode project. If it's iOS 16, either change to iOS 17 or switch to `ObservableObject`/`@StateObject` pattern.
- `AudioMonitor.$isSilent` in `ListeningViewModel` -- if mixing `@Observable` and Combine causes issues, switch to `onChange` in the view.
- Missing file references in Xcode project -- new Swift files may need to be added to the Xcode project's `project.pbxproj`. If building from the command line and files aren't found, they may need manual Xcode project membership.

**Step 2: Run in simulator**

Run: `xcodebuild build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -10`

Verify: App launches, shows the Listening State with "Play anything -- I'm listening" and the pulse view.

**Step 3: Commit any fixes**

```bash
git add -A apps/ios/
git commit -m "fix(ios): resolve compile issues from prototype integration"
```

---

## Task 11: Add Demo Mode Switcher

Add a way to toggle between the "first-time user" and "returning user" demo scenarios. This is a small settings control for demo purposes.

**Files:**
- Modify: `apps/ios/CrescendAI/Features/Listening/ListeningView.swift`

**Step 1: Add a long-press gesture on the wordmark to toggle demo mode**

Add a `@State private var demoMode: DemoMode = .firstTime` to `ListeningView` and recreate the view model when it changes. Add a long-press on the "CrescendAI" text to cycle between modes:

In `ListeningView`, modify the "CrescendAI" text:

```swift
Text("CrescendAI")
    .font(CrescendFont.headingLG())
    .foregroundStyle(CrescendColor.foreground)
    .onLongPressGesture {
        demoMode = demoMode == .firstTime ? .returning : .firstTime
        viewModel = ListeningViewModel(demoMode: demoMode)
        viewModel.startListening()
    }
```

Also add a brief visual indicator so demo presenters know which mode they're in:

```swift
Text(demoMode == .firstTime ? "Demo: First Time" : "Demo: Returning")
    .font(CrescendFont.labelSM())
    .foregroundStyle(CrescendColor.secondaryText.opacity(0.5))
```

Place this below the wordmark. It's subtle enough not to confuse real users but visible enough for demo presenters.

**Step 2: Verify it builds**

Run: `xcodebuild build -project apps/ios/CrescendAI.xcodeproj -scheme CrescendAI -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add apps/ios/CrescendAI/Features/Listening/ListeningView.swift
git commit -m "feat(ios): add long-press demo mode switcher on wordmark"
```

---

## File Summary

### New files (10):
```
apps/ios/CrescendAI/Features/Listening/AudioMonitor.swift
apps/ios/CrescendAI/Features/Listening/PulseView.swift
apps/ios/CrescendAI/Features/Listening/ListeningView.swift
apps/ios/CrescendAI/Features/Listening/ListeningViewModel.swift
apps/ios/CrescendAI/Features/Chat/ChatModels.swift
apps/ios/CrescendAI/Features/Chat/DemoScenarios.swift
apps/ios/CrescendAI/Features/Chat/ChatBlockViews.swift
apps/ios/CrescendAI/Features/Chat/ChatMessageView.swift
apps/ios/CrescendAI/Features/Chat/SuggestionChipsView.swift
apps/ios/CrescendAI/Features/Chat/FeedbackChatView.swift
apps/ios/CrescendAI/Features/Chat/FeedbackChatViewModel.swift
```

### Modified files (1):
```
apps/ios/CrescendAI/App/ContentView.swift
```

### Existing files reused (not modified):
```
apps/ios/CrescendAI/Features/Recording/AudioRecorder.swift (pattern reference)
apps/ios/CrescendAI/Features/Recording/AudioPlayer.swift (reused for reference playback)
apps/ios/CrescendAI/Networking/APIModels.swift (PerformanceDimensions for dimension cards)
apps/ios/CrescendAI/DesignSystem/* (all tokens and components)
```

### Asset files needed (add to bundle manually or create placeholders):
```
reference_nocturne_bars12_18.m4a (or .mp3) -- reference audio clip for demo scenario 2
```

If no reference audio clip is available, the `ReferencePlaybackView` will silently handle the missing file (the play button won't crash, it just won't play anything). For the demo, any short piano clip can be used.
