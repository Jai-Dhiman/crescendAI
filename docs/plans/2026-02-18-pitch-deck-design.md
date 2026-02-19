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

## Visual Techniques

### Spacing System (8px grid)

All spacing uses multiples of 8px:

| Element | Value |
|---------|-------|
| Slide outer margin | 120px left/right, 80px top/bottom |
| Content max width | 1680px (1920 - 2x120) |
| Content max height | 920px (1080 - 2x80) |
| Section gap (vertical) | 48px |
| Element gap (within section) | 24px |
| Card internal padding | 40px |
| Card gap (horizontal) | 32px |
| Title to content gap | 40px |

### Layout Grid (apply in Design Mode)

12-column grid: Margin 120px, Gutter 32px. Toggle Design Mode with Shift+D.

### Background Depth Layering

For dark slides (2 and 10), add visual depth:

1. Base: slide background set to linear gradient (`#2d2926` -> `#1a1816`, 180deg)
2. Layer 1: large ellipse (600x400px) in gold `#c9a227` at 4% opacity, Layer Blur 200px, placed off-center for ambient warmth
3. Layer 2 (optional): Noise texture rectangle (1920x1080), Overlay blend mode, 3% opacity (via Noise & Texture plugin)

For light slides, keep backgrounds clean -- the gradient alone is enough.

### Card Style

- Rectangle: corner radius 16px
- Fill: `#fefdfb` at 80% opacity
- Border: 1px stroke, `#e3d5c0` at 40% opacity
- Shadow: X:0 Y:4 Blur:16 Spread:-2, `rgba(37,31,26,0.06)`
- Internal padding: 40px (auto layout)

### Accent Line Style

- Width: 60-120px
- Height: 3px
- Fill: gold `#c9a227`
- Corner radius: 2px

### Gradient Text (for hero numbers)

Select text layer -> Fill section -> switch from Solid to Linear Gradient -> set stops: `#c9a227` (gold) to `#9a7b1a` (gold-dark), 135deg angle.

## Implementation Approach

Build in Figma Slides directly using Design Mode (Shift+D) for full design control. Follow the build guide below slide by slide.

---

## Build Guide

### Prerequisites

1. Open Figma and create a new **Figma Slides** file (File > New Figma Slides file)
2. Name it "CrescendAI Pitch Deck"
3. Press **Shift+D** to enter Design Mode
4. Set up a 12-column layout grid: click the empty slide, right sidebar > Layout guides > add Columns, Count: 12, Margin: 120, Gutter: 32

### Global Setup

**Fonts:** All are Google Fonts (available by default in Figma):
- Plus Jakarta Sans: Bold (700), SemiBold (600)
- Inter: Regular (400), Medium (500)
- Source Serif 4: Italic (400)

**Create Color Styles** (right sidebar > Local styles > + Color):
- `bg/cream`: `#fdf8f0`
- `bg/amber-light`: `#f6e9d5`
- `bg/amber-mid`: `#f0dfc4`
- `bg/amber-warm`: `#eedcca`
- `bg/amber-deep`: `#e8d5b8`
- `bg/ink-dark`: `#2d2926`
- `bg/ink-darkest`: `#1a1816`
- `text/heading`: `#2d2926`
- `text/body`: `#5a524a`
- `text/secondary`: `#746a5c`
- `text/light`: `#fefdfb`
- `text/muted-light`: `#c9b89a`
- `accent/gold`: `#c9a227`
- `accent/gold-dark`: `#9a7b1a`
- `accent/sepia`: `#a69276`
- `border/light`: `#e3d5c0`

**Create Text Styles** (right sidebar > Local styles > + Text):
- `display/hero`: Plus Jakarta Sans Bold, 120px, line-height 100%, letter-spacing -3%
- `display/title`: Plus Jakarta Sans Bold, 72px, line-height 110%, letter-spacing -2%
- `display/h1`: Plus Jakarta Sans Bold, 48px, line-height 120%, letter-spacing -1%
- `display/h2`: Plus Jakarta Sans Bold, 42px, line-height 120%, letter-spacing -1%
- `display/h3`: Plus Jakarta Sans SemiBold, 36px, line-height 130%
- `body/large`: Inter Regular, 28px, line-height 150%
- `body/medium`: Inter Regular, 24px, line-height 150%
- `body/small`: Inter Medium, 20px, line-height 140%
- `caption`: Inter Medium, 18px, line-height 140%, letter-spacing 2%
- `tagline`: Source Serif 4 Italic, 36px, line-height 140%

---

### Slide 1: Title

**Background:** Click the slide (deselect all), right sidebar > Background > Gradient (Linear). Set angle to 180deg. Stop 1: `#fdf8f0` at 0%. Stop 2: `#f6e9d5` at 100%.

**Build (all centered on slide):**

1. Text: "CrescendAI"
   - Style: `display/title` (Plus Jakarta Sans Bold, 72px)
   - Fill: `#2d2926`
   - Position: center horizontally, Y ~360px

2. Accent line below:
   - Rectangle: 80px wide, 3px tall
   - Fill: `#c9a227`
   - Corner radius: 2px
   - Position: centered, 24px below the title text

3. Text: "A teacher for every pianist."
   - Style: `tagline` (Source Serif 4 Italic, 36px)
   - Fill: `#5a524a`
   - Position: centered, 24px below the accent line

4. Bottom info block:
   - Text: "Your Name | crescend.ai | email@example.com"
   - Style: `caption` (Inter Medium, 18px)
   - Fill: `#746a5c`
   - Position: centered, Y ~960px (80px from bottom edge)

**Transition:** Dissolve, 400ms, ease-out

**Presenter note:** Paste your 30-second pitch from the investor prep doc.

---

### Slide 2: The Problem

**Background:** Gradient (Linear), 180deg. Stop 1: `#2d2926` at 0%. Stop 2: `#1a1816` at 100%.

**Depth layer (optional):**
- Ellipse: 600x400px, fill `#c9a227` at 4% opacity
- Apply Layer Blur: 200px
- Position: X 1200, Y 600 (lower-right, off-center)

**Build:**

1. Create a vertical auto layout frame (the content container):
   - Padding: 0
   - Gap between items: 48px
   - Position: X 120, Y ~240 (left-aligned, vertically centered)
   - Width: constrain to ~1200px

2. Three bullet rows (each is a horizontal auto layout frame):
   - Left element: Rectangle 3px wide, 100% height of text, fill `#c9a227`, corner radius 2px
   - Gap: 24px
   - Right element: Text
   - Row 1 text: "Students practice alone most of the time -- no one to help shape their sound"
   - Row 2 text: "Quality feedback costs $50-200/hr, is infrequent, and hard to access"
   - Row 3 text: "Existing apps check note accuracy. None help you sound better."
   - All text: `body/large` (Inter Regular 28px), fill `#fefdfb`

3. Bottom proof point:
   - Text: "30+ educator interviews confirmed this gap"
   - Style: `body/small` (Inter Medium, 20px)
   - Fill: `#c9b89a`
   - Position: X 120, Y ~920 (bottom-left)

**Transition:** Dissolve, 500ms, ease-out (slightly slower for the dark mood shift)

**Object animation:** Fade in on each bullet row, staggered (row 1 on click, rows 2-3 sequential after)

**Presenter note:** Problem narrative from founder story arc: "When you're practicing alone..."

---

### Slide 3: The Insight

**Background:** Gradient (Linear), 180deg. Stop 1: `#f6e9d5` at 0%. Stop 2: `#eedcca` at 100%.

**Build:**

1. Create a horizontal auto layout frame spanning the content area:
   - Padding: 0
   - Gap: 64px

2. Left column (55% width, ~880px):
   - Vertical auto layout, gap 24px
   - Text: "AI trained on millions of hours of music can hear what matters"
     - Style: `display/h2` (Plus Jakarta Sans Bold 42px)
     - Fill: `#2d2926`
     - Max width: 880px
   - Text: "55% more accurate than existing approaches. Published, first-author paper."
     - Style: `body/medium` (Inter Regular 24px)
     - Fill: `#5a524a`
   - Text: "These models only became capable in the last 1-2 years"
     - Style: `body/small` (Inter Medium 20px)
     - Fill: `#746a5c`

3. Right column (45% width, ~680px):
   - Placeholder frame: 680x560px rectangle
   - Fill: `#fefdfb` at 60% opacity
   - Corner radius: 16px
   - Border: 1px `#e3d5c0` at 40% opacity
   - Shadow: X:0 Y:4 Blur:16 Spread:-2, `rgba(37,31,26,0.06)`
   - Center text inside: "[ Waveform visualization ]" in Inter Medium 20px, `#a69276`

**Transition:** Dissolve, 400ms, ease-out

**Presenter note:** Insight section from 2-minute pitch: "Any app can tell you if you're playing the right notes..."

---

### Slide 4: The Product

**Background:** Gradient (Linear), 180deg. Stop 1: `#fefdfb` at 0%. Stop 2: `#f5f1e8` at 100%.

**Build:**

1. Title (centered):
   - Text: "Record yourself playing. Get the feedback a great teacher would give you, in seconds."
   - Style: `display/h3` (Plus Jakarta Sans SemiBold 36px)
   - Fill: `#2d2926`
   - Max width: 1200px, center-aligned
   - Position: centered, Y ~120px

2. Product screenshot placeholder (centered):
   - Rectangle: 1100x560px
   - Fill: `#fefdfb` at 80% opacity
   - Corner radius: 16px
   - Border: 1px `#e3d5c0`
   - Shadow: X:0 Y:8 Blur:24 Spread:-4, `rgba(37,31,26,0.08)`
   - Center text: "[ Product screenshot: radar chart + feedback cards ]" in Inter Medium 20px, `#a69276`
   - Position: centered, Y ~260px

3. Bottom feature strip (centered, horizontal auto layout):
   - Gap: 0px (use pipe separators)
   - Three text segments with gold pipe characters between:
     - "Specific, actionable feedback" | "Pedaling, dynamics, tone, phrasing" | "Runs on Cloudflare edge, <$20/month"
   - Text style: `body/small` (Inter Medium 20px), fill `#5a524a`
   - Pipe "|" characters: fill `#c9a227`
   - Gaps: 24px between text and pipe on each side
   - Position: centered, Y ~880px

**Transition:** Dissolve, 400ms, ease-out

**Presenter note:** Demo runbook cues: "Here are performances from some of the greatest pianists..."

---

### Slide 5: The Results

**Background:** Gradient (Linear), 180deg. Stop 1: `#f6e9d5` at 0%. Stop 2: `#f0dfc4` at 100%.

**Build (vertical auto layout, centered, gap 32px):**

1. Hero number:
   - Text: "55%"
   - Style: `display/hero` (Plus Jakarta Sans Bold 120px)
   - Fill: **Gradient** -- linear 135deg, stop 1 `#c9a227`, stop 2 `#9a7b1a`
   - Center-aligned

2. Label:
   - Text: "improvement over symbolic approaches"
   - Style: `body/medium` (Inter Regular 24px)
   - Fill: `#433d38`
   - Center-aligned

3. Data row (horizontal auto layout, gap 64px, centered):
   - Left: "R^2 = 0.537 vs 0.347" -- Inter Regular 28px, `#2d2926`
   - Right: "p < 10^-25" -- Inter Regular 28px, `#2d2926`

4. Validation text:
   - Text: "Validated across soundfonts, difficulty levels, multiple performers"
   - Style: `body/small` (Inter Medium 20px)
   - Fill: `#5a524a`
   - Center-aligned

5. Source text:
   - Text: "arXiv paper  |  ISMIR 2026 submission"
   - Style: `caption` (Inter Medium 18px)
   - Fill: `#a69276`
   - Center-aligned

**Transition:** Dissolve, 400ms, ease-out

**Object animation:** Fade in on the "55%" number (first, on click), then remaining text (sequential)

**Presenter note:** "Our approach is 55% more accurate than symbolic-only methods. This was validated rigorously..."

---

### Slide 6: Market

**Background:** Gradient (Linear), 180deg. Stop 1: `#eedcca` at 0%. Stop 2: `#e8d5b8` at 100%.

**Build:**

1. Top section (centered, vertical auto layout, gap 16px):
   - Text: "~40M"
     - Style: `display/hero` but at 96px (Plus Jakarta Sans Bold)
     - Fill: gradient `#c9a227` -> `#9a7b1a`
   - Text: "piano students globally"
     - Style: `body/large` (Inter Regular 28px)
     - Fill: `#433d38`
   - Position: centered, Y ~160px

2. Supporting text:
   - Text: "Online music education growing post-COVID"
   - Style: `body/medium` (Inter Regular 24px)
   - Fill: `#5a524a`
   - Position: centered, Y ~360px

3. Revenue path (horizontal auto layout, centered, Y ~560px):
   - Three cards in a row with arrows between:
   - Each card: auto layout frame, 380x160px, padding 32px
     - Fill: `#fefdfb` at 70% opacity, corner radius 16px, border 1px `#e3d5c0` at 30% opacity
     - Card 1 title: "B2C Subscription" (Plus Jakarta Sans SemiBold 20px, `#2d2926`)
     - Card 1 detail: "$10-30/month" (Inter Regular 18px, `#5a524a`)
     - Card 2: "Institutional Licenses" / "Schools + Conservatories"
     - Card 3: "API Licensing" / "Manufacturers + Apps"
   - Between cards: arrow shape or ">" text in `#c9a227`, vertically centered
   - Gap between card and arrow: 16px

**Transition:** Dissolve, 400ms, ease-out

**Presenter note:** "There are about 40 million piano students globally. Online music education has been growing steadily post-COVID..."

---

### Slide 7: Traction & Validation

**Background:** Gradient (Linear), 180deg. Stop 1: `#fefdfb` at 0%. Stop 2: `#f5f1e8` at 100%.

**Build:**

1. Title:
   - Text: "Traction & Validation"
   - Style: `display/h2` (Plus Jakarta Sans Bold 42px)
   - Fill: `#2d2926`
   - Position: centered, Y ~100px

2. Grid of 6 proof points (3 columns x 2 rows):
   - Use auto layout: wrap in a grid-like structure
   - Outer frame: horizontal auto layout, gap 32px, centered
   - Each row: 3 cards side by side
   - Each card: vertical auto layout frame, ~480x200px, padding 32px
     - Fill: `#fefdfb` at 70% opacity, corner radius 16px
     - Border: 1px `#e3d5c0` at 30% opacity
     - Number/keyword: Plus Jakarta Sans Bold 48px, `#c9a227`
     - Description: Inter Regular 20px, `#5a524a`, 8px gap below number

   Card contents:
   - "30+" / "Educator interviews"
   - "Published" / "arXiv + ISMIR 2026"
   - "3x" / "Hackathon winner"
   - "890K+" / "Lines shipped"
   - "0 -> 50" / "Users at Capture (founding engineer)"
   - "Cited by" / "Researchers at OpenAI & Google"

**Transition:** Dissolve, 400ms, ease-out

**Object animation:** Fade in on each row of cards (row 1 on click, row 2 sequential)

**Presenter note:** Key talking points for each proof point, plus FAQ answers if questioned.

---

### Slide 8: Founder

**Background:** Gradient (Linear), 180deg. Stop 1: `#f6e9d5` at 0%. Stop 2: `#eedcca` at 100%.

**Build (horizontal auto layout, gap 80px):**

1. Left: Photo placeholder (40% width, ~640px):
   - Rectangle: 520x600px
   - Fill: `#eedcca` at 80%
   - Corner radius: 24px
   - Shadow: X:0 Y:8 Blur:24 Spread:-4, `rgba(166,146,118,0.15)` (sepia shadow)
   - Center text: "[ Founder headshot ]" in Inter Medium 20px, `#a69276`
   - Position: left-aligned at X 120, vertically centered

2. Right: Bio (60% width):
   - Vertical auto layout, gap 32px
   - Name: Plus Jakarta Sans Bold 42px, `#2d2926`
   - Role: Inter Medium 20px, `#746a5c` -- "Founder & CEO"
   - Accent line: 60px x 3px, `#c9a227`, corner radius 2px
   - Bio points (vertical auto layout, gap 20px):
     - Each point is a horizontal auto layout: gold circle (8px diameter, `#c9a227`) + 16px gap + text
     - "Berklee College of Music (percussion, 5x Dean's List)"
     - "Pianist since age 8, active orchestral musician"
     - "Self-taught ML engineer -> founding engineer -> founder"
     - "Deep domain expertise + builds the whole stack alone"
     - Text: Inter Regular 24px, `#5a524a`

**Transition:** Dissolve, 400ms, ease-out

**Presenter note:** Team section from 2-minute pitch: "I'm a solo technical founder..."

---

### Slide 9: Roadmap

**Background:** Gradient (Linear), 180deg. Stop 1: `#fefdfb` at 0%. Stop 2: `#f5f1e8` at 100%.

**Build:**

1. Title:
   - Text: "Roadmap"
   - Style: `display/h2` (Plus Jakarta Sans Bold 42px)
   - Fill: `#2d2926`
   - Position: centered, Y ~100px

2. Timeline (centered, Y ~400px):
   - Horizontal line: 1200px wide, 2px tall, fill `#e3d5c0`
   - Position: centered

   - Four node groups evenly spaced along the line (each ~300px apart):
     - Circle marker: 16px diameter, fill `#c9a227` for "Now", stroke-only `#c9a227` 2px for future nodes
     - Position: centered on the line

   - Below each circle (vertical auto layout, gap 8px, center-aligned):
     - Label: Plus Jakarta Sans SemiBold 20px
       - "Now" in `#c9a227`
       - "3 Months", "6 Months", "Research" in `#2d2926`
     - Description: Inter Regular 18px, `#5a524a`, max width 240px, center-aligned
       - "Curated gallery with AI feedback"
       - "User uploads, accounts, progress tracking"
       - "Mobile app, real-time analysis, instrument expansion"
       - "Dual-encoder (audio + score), large-scale data"

**Transition:** Dissolve, 400ms, ease-out

**Presenter note:** Brief expansion on each milestone and what "done" looks like.

---

### Slide 10: The Ask

**Background:** Gradient (Linear), 180deg. Stop 1: `#2d2926` at 0%. Stop 2: `#1a1816` at 100%.

**Depth layer (same as Slide 2):**
- Ellipse: 600x400px, fill `#c9a227` at 4% opacity, Layer Blur 200px
- Position: X 200, Y 300 (upper-left this time, mirroring Slide 2)

**Build (centered vertically and horizontally):**

1. Text: "Let's talk."
   - Style: `display/title` (Plus Jakarta Sans Bold, 72px)
   - Fill: `#fefdfb`
   - Center-aligned

2. Text (24px gap below):
   - "Pre-seed to accelerate: first hire, GPU credits, user research."
   - Style: `body/medium` (Inter Regular 24px)
   - Fill: `#e3d5c0`
   - Center-aligned

3. Accent line (48px gap below):
   - Rectangle: 80px x 2px, fill `#c9a227`, corner radius 1px
   - Centered

4. Contact info (24px gap below accent):
   - Text: "Your Name | crescend.ai | email@example.com"
   - Style: `body/small` (Inter Medium 20px)
   - Fill: `#c9b89a`
   - Center-aligned

**Transition:** Dissolve, 500ms, ease-out

**Presenter note:** The ask language + natural close questions: "I'd love to keep you in the loop. What's the best way to do that?"

---

### Final Steps

1. **Review all slides** in Present mode (click Present button) -- check flow, transitions, readability
2. **Add presenter notes** to each slide from `docs/investor-meeting-prep.md`
3. **Replace placeholders** with actual assets (product screenshot, founder photo, waveform graphic)
4. **Export PDF** for the leave-behind version (File > Export as PDF)
5. **Share link** for remote viewing (Share > Anyone with link > Can view)
