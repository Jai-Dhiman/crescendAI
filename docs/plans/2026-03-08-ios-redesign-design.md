# iOS App Redesign -- Design Document

**Date:** 2026-03-08
**Approach:** Hybrid -- new shell, reuse internals

## Problem

The current iOS app has a fixed 56pt icon-only sidebar, procedural gradient login screen, serif font everywhere, and sheet-based navigation. It looks flat and unpolished. The redesign replaces the shell (navigation, sidebar, login, bottom bar) while keeping the solid inner views (chat, cards, waveform, profile).

## Design Decisions

- Dark espresso/cream palette (unchanged)
- Claude-style collapsible sidebar drawer (hamburger button, swipe to dismiss)
- Flat chronological session list in sidebar, profile at bottom
- Real photo background on login (Image5.jpg from web app)
- Centered frosted-glass card on login
- Lora for display/headings, SF Pro for body/labels
- Dual-mode bottom bar: text input + "Start Practicing" button (idle) / waveform + controls (recording)
- Centered greeting + quick-action chips on empty state
- All navigation via NavigationStack pushes (no sheets)

## Section 1: App Shell & Navigation

Root structure changes from `HStack(SidebarView, ChatView)` to a `NavigationStack` root with a sidebar drawer overlay.

**Sidebar drawer:**
- Hidden by default. Opens via hamburger button (top-left).
- Slides in from left, dims main content.
- Dismissible by: swipe left, tap dimmed area, tap hamburger.
- Width: ~280pt.
- Background: sidebarBackground (#252220).

**Sidebar contents (top to bottom):**
- Header: "CrescendAI" wordmark (Lora displayMD) + "New Session" button
- Body: flat chronological session list (piece name + date, most recent first)
- Footer: user profile row (avatar + name), tappable to push ProfileView

**Top navigation bar:**
- Left: hamburger icon (opens sidebar)
- Center: context-dependent title (piece name during session, empty on home)
- Right: context-dependent

All secondary screens (Profile, Session Review, Focus Mode) push via NavigationStack.

## Section 2: Login Screen

- `Image5.jpg` added to iOS asset catalog, full-bleed background (`contentMode: .fill`)
- Radial gradient overlay: `rgba(45,41,38,0.4)` center to `rgba(45,41,38,0.85)` edges (matches web)
- Centered frosted card (`.ultraThinMaterial`, 16pt corner radius):
  - "CrescendAI" (Lora displayXL, cream)
  - "A teacher for every pianist." (SF Pro bodyLG, secondary)
  - Apple Sign In button (native, white, 50pt)
  - Loading spinner + error text
- Card animates in with opacity + scale on appear

## Section 3: Main Screen Empty State

- Centered greeting: "How's practice going?" (Lora headingXL)
- Time-aware sub-greeting: "Good evening, Jai" (SF Pro bodyMD, secondary)
- Quick-action chips below:
  - "Start practicing" (primary style -- cream bg, espresso text)
  - "Review last session" (secondary -- bordered)
  - "Ask a question" (secondary -- bordered)
- Bottom bar always visible below

## Section 4: Dual-Mode Bottom Bar

**Idle mode:**
- Text input field: rounded rect, surface bg, placeholder "Ask your teacher..."
- Send button appears when text entered
- Above text field: "Start Practicing" button -- full-width, cream bg, mic icon, 50pt height

**Recording mode (after tapping Start Practicing):**
- Crossfade animation (~0.3s ease-in-out)
- WaveformView slides in
- Controls below: Pause (left), Stop (center, cream circle), "How was that?" (right)
- Piece name / elapsed time above waveform
- Semi-transparent material background

## Section 5: Typography

Update `CrescendFont` token definitions:
- **Lora:** displayXL, displayLG, displayMD, headingXL, headingLG, headingMD
- **SF Pro (system font):** bodyLG, bodyMD, bodySM, labelLG, labelMD, labelSM

Change propagates automatically through all views that reference tokens.

## Section 6: Reuse Plan

**Reused as-is** (automatic typography update via tokens):
- ChatView, MessageBubble, ObservationCard, DimensionPill
- WaveformView (moved into bottom bar but component unchanged)
- ProfileView (pushed instead of sheet, content unchanged)
- SparklineView, SessionReviewView, FocusModeView
- ComponentCards (ExerciseSetCard, KeyboardGuideCard, ScoreHighlightCard)
- ChatViewModel, all services, all SwiftData models
- Color and Spacing tokens

**Rewritten:**
- `MainView` -- new NavigationStack shell + sidebar drawer
- `SidebarView` -- full drawer with session list (replaces 56pt icon strip)
- `SignInView` -- real photo background (replaces gradient)
- `ContentView` -- simplified auth routing
- `ChatInputBar` -- replaced with dual-mode bottom bar
- `PracticeControlBar` -- folded into bottom bar's recording state

## Key Files

- `apps/ios/CrescendAI/App/MainView.swift` -- rewrite
- `apps/ios/CrescendAI/App/SidebarView.swift` -- rewrite
- `apps/ios/CrescendAI/App/ContentView.swift` -- simplify
- `apps/ios/CrescendAI/Features/Auth/SignInView.swift` -- rewrite
- `apps/ios/CrescendAI/Features/Chat/ChatInputBar.swift` -- replace with DualModeBottomBar
- `apps/ios/CrescendAI/Features/Practice/PracticeControlBar.swift` -- fold into bottom bar
- `apps/ios/CrescendAI/DesignSystem/Tokens/Typography.swift` -- serif/sans split
- `apps/web/public/Image5.jpg` -- copy to iOS asset catalog

## Design References

- **Claude iOS app:** sidebar drawer pattern, hamburger button, clean minimal layout
- **Suno:** full-bleed photo splash, bold wordmark
- **Breathwork app:** dark theme, elegant typography, immersive recording modes
