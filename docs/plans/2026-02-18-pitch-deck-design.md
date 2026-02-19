# Pitch Deck Design: CrescendAI Pre-Seed

**Date:** 2026-02-18

## Context

Pre-seed angel investor pitch deck. Dual-purpose: live presentation (backup visual aid) and leave-behind PDF. 10 slides following the structure from `docs/investor-meeting-prep.md`. Built in Figma Slides.

## Tool

**Figma Slides** (dedicated presentation tool). Fixed 1920x1080 slides, built-in transitions, presenter notes, PDF export.

Chosen over Figma Design prototypes (no presenter notes) and hybrid approaches (overkill for a single deck).

## Brand Direction

Matches the new landing page direction (`apps/docs/landing-page-redesign.md`), not the current live site. Warm, music-forward, modern -- not the old scholarly/stuffy aesthetic.

## Color Palette

### Dominant (60%) -- Warm Amber Gradient Backgrounds

Slides use warm amber gradients that shift across the deck:

| Token | Hex | Usage |
|-------|-----|-------|
| cream-start | `#fdf8f0` | Lightest gradient start |
| amber-light | `#f6e9d5` | Mid gradient |
| amber-mid | `#f0dfc4` | Deeper gradient |
| amber-warm | `#eedcca` | Warm sections |
| amber-deep | `#e8d5b8` | Rich warm sections |
| amber-rich | `#dfc9a8` | Deepest warm |

Two slides (Problem and Ask) use an inverted dark background for contrast:

| Token | Hex | Usage |
|-------|-----|-------|
| ink-dark | `#2d2926` | Dark gradient start |
| ink-darkest | `#1a1816` | Dark gradient end |

### Secondary (30%) -- Sepia & Ink Tones

| Token | Hex | Usage |
|-------|-----|-------|
| ink-900 | `#2d2926` | Headings on light |
| ink-700 | `#433d38` | Strong body text |
| ink-600 | `#5a524a` | Body text |
| ink-500 | `#746a5c` | Secondary text |
| sepia-500 | `#a69276` | Decorative elements |
| sepia-600 | `#8b7355` | Borders, icons |
| sepia-400 | `#c9b89a` | Subtle accents |
| sepia-300 | `#e3d5c0` | Light borders |
| paper-50 | `#fefdfb` | Text on dark slides |

### Accent (10%) -- Scholarly Gold

| Token | Hex | Usage |
|-------|-----|-------|
| gold | `#c9a227` | Key metrics, highlights, CTAs |
| gold-dark | `#9a7b1a` | Accent lines, markers |

### CTA Gradient

`linear-gradient(135deg, #8b7355 0%, #6e5a43 100%)`

## Typography

| Role | Font | Weight | Size Range |
|------|------|--------|------------|
| Slide titles | Plus Jakarta Sans | Bold (700) | 42-72pt |
| Hero numbers | Plus Jakarta Sans | Bold (700) | 96-120pt |
| Subtitles | Plus Jakarta Sans | SemiBold (600) | 28-36pt |
| Body text | Inter | Regular (400) / Medium (500) | 24-28pt |
| Labels/captions | Inter | Medium (500) | 18-20pt |
| Tagline/quotes | Source Serif 4 | Italic (400) | 28-36pt |

Maximum 2 fonts per slide. Use weight variations for hierarchy within a family.

## Slide-by-Slide Design

### Slide 1: Title

- **Layout:** Centered, minimal
- **Background:** Warm amber gradient (`#fdf8f0` -> `#f6e9d5`)
- **Elements:**
  - "CrescendAI" -- Plus Jakarta Sans Bold 72pt, `#2d2926`
  - "A teacher for every pianist." -- Source Serif 4 Italic 36pt, `#5a524a`
  - Horizontal gold accent line (`#c9a227`, 2px) between name and tagline
  - Name, crescend.ai, contact -- Inter 18pt, `#746a5c`, bottom-aligned
- **Optional texture:** Faint SVG waveform curve at ~5% opacity in gradient

### Slide 2: The Problem

- **Layout:** Left-aligned bullets with gold accent bars
- **Background:** Dark gradient (`#2d2926` -> `#1a1816`) -- contrast slide
- **Elements:**
  - Three bullet points -- Inter 28pt, `#fefdfb`:
    - "Students practice alone most of the time -- no one to help shape their sound"
    - "Quality feedback costs $50-200/hr, is infrequent, and hard to access"
    - "Existing apps check note accuracy. None help you sound better."
  - Gold accent bar (3px, `#c9a227`) left of each bullet
  - Bottom: "30+ educator interviews confirmed this gap" -- Inter 20pt, `#c9b89a`

### Slide 3: The Insight

- **Layout:** Split-screen (text left 55%, visual right 45%)
- **Background:** Warm gradient (`#f6e9d5` -> `#eedcca`)
- **Left:**
  - "AI trained on millions of hours of music can hear what matters" -- Plus Jakarta Sans Bold 42pt, `#2d2926`
  - "55% more accurate than existing approaches. Published, first-author paper." -- Inter 24pt, `#5a524a`
  - "These models only became capable in the last 1-2 years" -- Inter 20pt, `#746a5c`
- **Right:** Placeholder frame for waveform/audio visualization

### Slide 4: The Product

- **Layout:** Full-width product showcase
- **Background:** Light gradient (`#fefdfb` -> `#f5f1e8`)
- **Elements:**
  - "Record yourself playing. Get the feedback a great teacher would give you, in seconds." -- Plus Jakarta Sans Bold 36pt, `#2d2926`, centered
  - Large placeholder frame (60% of slide width) -- product screenshot area with subtle card shadow
  - Bottom strip: Three points separated by gold pipes -- Inter 20pt, `#5a524a`:
    - "Specific, actionable feedback" | "Pedaling, dynamics, tone, phrasing" | "Runs on Cloudflare edge, <$20/month"

### Slide 5: The Results

- **Layout:** Hero number center, supporting data below
- **Background:** Warm gradient (`#f6e9d5` -> `#f0dfc4`)
- **Elements:**
  - "55%" -- Plus Jakarta Sans Bold 120pt, `#c9a227` (gold)
  - "improvement over symbolic approaches" -- Inter 24pt, `#433d38`
  - Two-column data: "R^2 = 0.537 vs 0.347" and "p < 10^-25" -- Inter 28pt, `#2d2926`
  - "Validated across soundfonts, difficulty levels, multiple performers" -- Inter 20pt, `#5a524a`
  - "arXiv paper | ISMIR 2026 submission" -- Inter 18pt, `#a69276`

### Slide 6: Market

- **Layout:** Big number + breakdown + horizontal flow
- **Background:** Deep warm gradient (`#eedcca` -> `#e8d5b8`)
- **Elements:**
  - "~40M" -- Plus Jakarta Sans Bold 96pt, `#c9a227`
  - "piano students globally" -- Inter 28pt, `#433d38`
  - "Online music education growing post-COVID" -- Inter 24pt, `#5a524a`
  - Horizontal revenue path: "B2C subscription ($10-30/mo)" -> "Institutional licenses" -> "API licensing" -- connected with gold arrow accents, Inter 20pt

### Slide 7: Traction & Validation

- **Layout:** Grid of proof points (2x3 or 3x2)
- **Background:** Light gradient (`#fefdfb` -> `#f5f1e8`)
- **Grid cells:** Each has gold number/icon (Plus Jakarta Sans Bold 48pt, `#c9a227`) + description (Inter 20pt, `#5a524a`):
  - "30+" / "Educator interviews"
  - "Published" / "arXiv + ISMIR 2026"
  - "Feedback from" / "MIR researchers, ML engineers at OpenAI & Google"
  - "3x" / "Hackathon winner"
  - "890K+" / "Lines shipped"
  - "Founding engineer" / "Capture: 0->50 users"

### Slide 8: Founder

- **Layout:** Split-screen (photo left 40%, bio right 60%)
- **Background:** Warm gradient (`#f6e9d5` -> `#eedcca`)
- **Left:** Placeholder frame for headshot (rounded corners, subtle sepia shadow)
- **Right:**
  - Name -- Plus Jakarta Sans Bold 42pt, `#2d2926`
  - Bio points -- Inter 24pt, `#5a524a`, with gold bullet markers:
    - "Berklee College of Music (percussion, 5x Dean's List)"
    - "Pianist since age 8, active orchestral musician"
    - "Self-taught ML engineer -> founding engineer -> founder"
    - "Deep domain expertise + builds the whole stack alone"

### Slide 9: Roadmap

- **Layout:** Horizontal timeline
- **Background:** Light gradient (`#fefdfb` -> `#f5f1e8`)
- **Elements:**
  - "Roadmap" -- Plus Jakarta Sans Bold 42pt, `#2d2926`
  - Horizontal gold line connecting four nodes
  - Each node: gold circle marker (`#c9a227`), label (Plus Jakarta Sans SemiBold 20pt), description (Inter 18pt, `#5a524a`):
    - **Now:** "Curated gallery with AI feedback"
    - **3 months:** "User uploads, accounts, progress tracking"
    - **6 months:** "Mobile app, real-time analysis, instrument expansion"
    - **Research:** "Dual-encoder (audio + score), large-scale data"

### Slide 10: The Ask

- **Layout:** Centered, clean -- bookends with Slide 2
- **Background:** Dark gradient (`#2d2926` -> `#1a1816`)
- **Elements:**
  - "Let's talk." -- Plus Jakarta Sans Bold 72pt, `#fefdfb`
  - "Pre-seed to accelerate: first hire, GPU credits, user research." -- Inter 24pt, `#e3d5c0`
  - Gold accent line (`#c9a227`, 2px) separating message from contact
  - Contact info + crescend.ai -- Inter 20pt, `#c9b89a`

## Transitions

- **Between slides:** Dissolve (0.3s) -- subtle, professional, doesn't distract
- **Object animations:** Fade-in for text elements on each slide (stagger headlines before body)
- **Dark slide transitions (slides 2, 10):** Slightly slower dissolve (0.5s) to let the mood shift register

## Presenter Notes

Each slide includes markdown presenter notes drawn from `docs/investor-meeting-prep.md`:
- Slide 1: 30-second pitch
- Slide 2: Problem narrative from founder story
- Slide 3: Insight section from 2-minute pitch
- Slide 4: Demo runbook cues
- Slides 5-7: Key talking points and FAQ answers for follow-up questions
- Slide 8: Team section from 2-minute pitch
- Slide 9: Brief expansion on each milestone
- Slide 10: The ask language + natural close questions

## Figma Slides Constraints

- Fixed 1920x1080 slide size
- One animation per object (no chaining)
- PDF export does not include presenter notes
- No built-in charting -- charts/timelines designed as manual vector layouts
- No embedded video/audio

## Visual Assets Needed

All assets use placeholder frames in the initial build. Replace later:

1. **Slide 3:** Waveform/audio visualization graphic
2. **Slide 4:** Product screenshot (analysis page with radar chart + feedback)
3. **Slide 8:** Founder headshot
4. **Slide 1 (optional):** CrescendAI logo mark

## Implementation Approach

Build in Figma Slides directly. Since the Figma MCP tools are read-only and the Figma Slides API requires a plugin context, the deck will be built manually in Figma following this spec. An implementation plan will detail the step-by-step build order.
