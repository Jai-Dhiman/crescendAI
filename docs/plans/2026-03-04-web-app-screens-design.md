# Web App Screens Design

**Date:** 2026-03-04
**Scope:** Sign-in page, main signed-in screen, navigation/auth flow, remove `/analyze`

## Overview

Build two new screens for the web app: a sign-in page and the main signed-in experience. The main screen is a functional web companion to the iOS practice app, with a chat-first interface for receiving teacher observations. Design-only for now -- audio capture and real inference will be wired later.

Auth: Sign in with Apple only, consistent with iOS.

## Sign-In Page (`/signin`)

- Full-viewport, full-bleed `Image4.jpg` background (hands on keys, dark/moody)
- Subtle dark gradient overlay for readability
- Centered floating card:
  - `bg-surface/80` with `backdrop-blur-xl`, `border border-border`
  - No rounded corners (matches landing page square aesthetic)
  - "crescend" wordmark in Lora display-sm
  - Tagline: "A teacher for every pianist." in body-md, secondary text
  - "Sign in with Apple" button (Apple HIG standard black button)
  - Small footer: "By signing in, you agree to our Terms of Service"
- Card enters with `--animate-fade-in-up`
- Mobile: card takes full width with horizontal padding, image still fills viewport

## Main Signed-In Screen (`/app`)

Inspired by Gemini's layout: thin icon sidebar, centered content area, minimal chrome.

### Thin Icon Sidebar (~48px, left edge)

Icon-only, cream color, tooltip on hover. Top-aligned cluster:

1. `+` New Chat -- starts a new practice session
2. Chat bubble icon -- Chats (opens panel/drawer with session history)
3. Metronome icon

No logo in sidebar. No profile in sidebar.

### Top Right

Profile initial circle (floating). Click opens dropdown with "Sign Out".

### Main Center Area (Empty/Home State)

- Time-aware greeting: "Good evening, Jai." in display-md Lora
- Input box below (`bg-surface border border-border`):
  - Placeholder: "What are you practicing today?"
  - Microphone/record icon button on the right (Start Recording action)
- No quick-action pills below

### Active Session State (Stubbed)

- Center becomes scrolling chat
- Teacher observations: left-aligned, surface cards
- Student messages: right-aligned
- Bottom input bar: text input + mic button, placeholder "Ask about your playing..."
- Session header with piece name + "End Session"

### Mobile (<768px)

- Sidebar collapses (hamburger or hidden)
- Chat takes full width
- Input sticks to bottom

## Navigation & Auth Flow

### Routes

| Route | Access | Purpose |
|-------|--------|---------|
| `/` | Public | Landing page (existing) |
| `/signin` | Public | Sign-in page |
| `/app` | Protected | Main signed-in screen |

Remove `/analyze` route entirely.

### Auth Guard (Stubbed)

- `beforeLoad` on `/app` checks for auth state
- No token: redirect to `/signin`
- Stub: localStorage flag to toggle signed-in/signed-out
- Real Sign in with Apple wired later

### Header Behavior

- `/` (landing): existing header with "Sign In" link
- `/signin`: no header (self-contained full-bleed)
- `/app`: no header (sidebar + profile replaces it)

### CTA Navigation

- Landing page "Start Practicing" button: checks auth, routes to `/app` or `/signin`
- Successful sign-in: redirect to `/app`
- Sign out: clear state, redirect to `/`

## Design System

All existing tokens apply. No new colors or fonts needed:

- Palette: espresso/cream/surface/border
- Typography: Lora throughout (display + body scales)
- Buttons: square (no border-radius), cream on espresso for primary
- Cards: `bg-surface border border-border`, no rounded corners
- Animation: `--animate-fade-in-up` for entry
