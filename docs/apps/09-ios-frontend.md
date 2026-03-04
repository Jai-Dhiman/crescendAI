# Slice 9: iOS Practice Companion Frontend

**Status:** IN PROGRESS
**Last verified:** 2026-03-03
**What's done:** PracticeView with basic session UI. DesignSystem with tokens (Colors, Typography, Spacing) and components (CrescendButton, CrescendCard). SignInView.
**What remains:** Observation view, session review, focus mode UI, profile/settings. Old feature directories (Chat, Analysis, MIDI, Recording, Listening, Home) deleted in cleanup.

See `docs/architecture.md` for the full system architecture.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the iOS frontend for the practice companion -- a premium, restrained interface designed for adult pianists. Minimal during practice, insightful after.

**Architecture:** Native SwiftUI app. Three primary screens: Practice (active session), Observation (teaching moment response), and Session Review (post-practice). Focus mode is a guided overlay. Design language is serious and adult -- think Oura, Headspace, or a fine instrument tuner.

**Tech Stack:** SwiftUI, AVAudioEngine (from Slice 2), URLSession, Combine/async-await

---

## Context

The current web app (Leptos/WASM) serves a different product -- the upload-and-analyze report card. The practice companion is a fundamentally different interaction model and benefits from native iOS for: background audio, reliable performance, platform-native feel, and the trust that comes with a dedicated app.

The web app (crescend.ai) continues to exist as the landing page and demo gallery. The iOS app is the practice companion.

## Design Principles

- **Invisible during practice:** The screen should not compete for attention while the student is playing
- **Glanceable when needed:** Observations appear large and readable from arm's length (phone on music stand)
- **Serious, not playful:** No gradients, no animations for animation's sake, no rounded-everything. Typography-driven, muted palette.
- **Session-aware:** The app understands that practice is a continuous activity, not a series of discrete uploads

## Screens

### 1. Practice Screen (Active Session)

**During recording (most of the time):**

- Large, subtle audio level indicator (waveform or amplitude bar) -- confirms the mic is listening
- Session timer (elapsed time, small, top corner)
- Dim background
- Single prominent element: **"How was that?"** button (large, centered-bottom, always reachable with one hand)
- Optional: voice activation indicator ("Listening for your voice..." when speech detection is enabled)

**While processing (after "how was that?"):**

- Brief processing indicator (1-3 seconds)
- Transitions to Observation screen

### 2. Observation Screen

**One observation displayed prominently:**

- Large, readable text (system font, 20pt+, high contrast)
- Readable from 2-3 feet away (phone on music stand)
- Below the observation: two actions:
  - "Tell me more" -- expands to elaboration
  - "Back to practice" -- returns to Practice Screen
- Observations accumulate in a scrollable list if multiple questions are asked in one session

**The observation itself:**

- Natural language, 1-3 sentences
- No scores, no charts, no data visualization
- Example: "The crescendo in the second phrase peaked too early -- the sforzando didn't land. Try holding back the build longer."

### 3. Session Review (Post-Practice)

**Shown when session ends:**

- Session summary: duration, number of questions asked
- Timeline of observations (chronological)
- Each observation tagged with which dimension and where in the session
- Overall session note from the teacher (LLM-generated 1-2 sentence summary)
- Check-in question (if triggered by Slice 5 logic)

**Session history:**

- Scrollable list of past sessions
- Tap into any session to review observations
- Simple, chronological -- not gamified

### 4. Focus Mode Overlay

**When focus mode is active:**

- Header: "Focusing on: Pedaling"
- Current exercise displayed:
  - Title
  - Instructions (readable text)
  - Notation snippet (if available -- rendered MusicXML via OSMD or a static image)
- "I'm ready" button (to record an attempt)
- After attempt: focused feedback appears
- Progress indicator: Exercise 1 of 3, Exercise 2 of 3, etc.

### 5. Profile / Settings

- Apple ID (from Sign in with Apple)
- Explicit goals (editable: "Recital June 15", "Focusing on Chopin Ballade No. 1")
- Session history overview
- Dimension trend (simple line chart per dimension over sessions -- NOT a radar chart)
- Mic sensitivity / recording quality settings
- Sign out

## Navigation

```
Tab Bar (if used) or simple flow:

[Practice] ---- tap "How was that?" ---- [Observation]
    |                                          |
    |                                     "Back to practice"
    |                                          |
    +------ End session --------> [Session Review]
    |                                          |
    |                                     [Session History]
    |
    +------ Focus mode ---------> [Focus Mode Overlay]
    |
    +------ Profile/Settings ---> [Profile]
```

The primary flow is: Practice -> Observation -> Practice (loop). Everything else is secondary.

## Design Language

**Typography:** San Francisco (system), large sizes. The observation text is the hero element. Note: the design system may evolve from Lora serif to system San Francisco, based on readability testing at arm's length.

**Color palette:** Muted, warm neutrals. One accent color for interactive elements. No bright colors competing with the observation text.

**Inspiration (feel, not copy):**

- Oura Ring: data-driven insight, premium feel, restrained color
- Headspace: session-based, calm, purposeful
- iA Writer: typography-first, no distractions
- A high-end metronome app: functional, serious, beautiful in its simplicity

**What it does NOT look like:**

- Yousician (gamified, colorful, youth-oriented)
- A music notation app (score-centric)
- A social media app (feeds, likes, profiles)

## What This Slice Does NOT Include

- Notation rendering engine (use static images or OSMD WebView for V1)
- Voice input processing (use button for V1, add speech-to-text later)
- MIDI playback
- iPad layout
- Android

## Tasks

**Task 1: Project setup and navigation skeleton**

- Clean up existing apps/ios/ stub
- Set up SwiftUI navigation (NavigationStack or tab-based)
- Implement screen shells: Practice, Observation, Session Review, Profile
- Wire up navigation between screens
- Integrate Sign in with Apple flow on first launch

**Task 2: Practice screen**

- Audio level indicator (AVAudioEngine tap for amplitude)
- Session timer
- "How was that?" button with processing state
- Dim/minimal visual design

**Task 3: Observation screen**

- Display observation text (large, readable)
- "Tell me more" expansion
- "Back to practice" navigation
- Observation history list for multi-question sessions

**Task 4: Session review screen**

- Session summary layout
- Observation timeline
- Check-in question display + answer input
- Session history list

**Task 5: Focus mode overlay**

- Exercise display (title, instructions, notation placeholder)
- "I'm ready" recording trigger
- Focused feedback display
- Exercise progress indicator

**Task 6: Profile and settings**

- Auth state display
- Goals editor
- Dimension trend chart (simple line chart, one per dimension)
- Settings (mic, sign out)

**Task 7: Visual design pass**

- Typography scale
- Color palette
- Spacing and layout refinement
- Dark mode support (important for practice environments)

### Open Questions

1. Should the web app (crescend.ai) link to the iOS app? "Download for practice companion" CTA?
2. Dark mode by default? Many pianists practice in dimly lit rooms.
3. Notation rendering: WebView with OSMD, native rendering library, or static images for V1?
4. Voice activation for "how was that?" -- feasible with on-device speech recognition (iOS Speech framework)?
