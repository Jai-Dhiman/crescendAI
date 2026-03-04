# Landing Page Redesign

Date: 2026-03-04
Status: APPROVED

Redesign the crescend.ai landing page and update the design system. Inspired by pillowtalk (full-bleed photography hero), Suno/[untitled] (dark surface feature cards), and a food brand (cascading editorial photography with serif type).

---

## Design System Changes

### Color Palette

Two colors. Everything derives from their interplay.

Primary dark: **#2D2926** (Espresso)
Primary light: **#FDF8F0** (Warm Cream)

Dark palette (landing page):
```
--dark-bg:             #2D2926
--dark-surface:        #3A3633
--dark-surface-2:      #454140
--dark-border:         #504B48
--dark-text-primary:   #FDF8F0
--dark-text-secondary: #A8A29E
--dark-text-tertiary:  #78716C
```

Light palette (web app, iOS -- for reference):
```
--light-bg:            #FDF8F0
--light-surface:       #F5F0E8
--light-surface-2:     #EDE8E0
--light-border:        #DDD8D0
--light-text-primary:  #2D2926
--light-text-secondary: #605B56
--light-text-tertiary:  #A8A29E
```

No accent color. Richness comes from photography, typography weight, and the espresso/cream interplay.

### Typography

Unchanged families:
- Display: **Lora** (serif) -- headlines, pull quotes, section titles
- Body/UI: **DM Sans** (sans-serif) -- nav, body, buttons, labels

Type scale unchanged from existing design system.

### Buttons

Primary CTA (dark mode): pill shape with subtle rounding.
```
bg: #FDF8F0 (cream)
text: #2D2926 (espresso)
padding: 14px 32px
border-radius: 100px (full pill)
font: DM Sans 15px, weight 500
hover: slight brightness increase
```

---

## Page Structure

Six sections. Short page. Narrative arc: captivate, inform, move, convert.

### 1. Navigation

Fixed top bar. Backdrop blur on scroll.

```
[crescend]                                    [Start Practicing]
```

- Logo: "crescend" in Lora, 18-20px, weight 500, warm cream
- CTA: "Start Practicing" pill button (cream bg, espresso text)
- No other links. No hamburger menu. Maximum restraint.
- On scroll: backdrop-filter blur(20px), espresso bg at 80% opacity

### 2. Hero

Full viewport height. Full-bleed piano photograph (Image1.jpg -- grand piano from above). Gradient overlay for text legibility.

Content (centered):
```
A teacher for every pianist.

[Start Practicing]
```

- Headline: Lora, clamp(3rem, 8vw, 7rem), warm cream, text-wrap balance, centered
- CTA: pill button, same as nav
- No subhead. No explanatory text. No "Free on iPhone." The photograph and six words do the work.
- Gradient overlay: espresso gradient, ~60% at bottom fading to ~10% at top
- Image: object-fit cover, full viewport

### 3. Feature Cards

Three dark surface cards in a responsive grid.

Layout: 3 columns desktop, 1 column mobile. Gap: 24px. Section padding: 96-128px vertical.

Card 1: "Your teacher is listening"
- Visual: placeholder (eventually waveform/listening indicator)
- Copy: "Your phone listens while you play. When you pause and ask, your teacher is ready with the one thing that matters most."

Card 2: "Exercises built for you"
- Visual: placeholder (eventually exercise card UI)
- Copy: "Not generic drills. Targeted practice for the specific passage and skill your teacher identified."

Card 3: "See what you hear"
- Visual: placeholder (eventually keyboard guide with lit keys)
- Copy: "The score lights up on a piano keyboard. See the notes, the fingering, the dynamics -- then play along."

Card styling:
- Background: #3A3633 (dark-surface)
- Border: 1px solid #504B48 (dark-border)
- Border-radius: 12px
- Visual area: top ~60% of card (placeholder gradient or muted tone)
- Title: Lora, ~1.5rem
- Description: DM Sans, body-md, secondary text color
- Internal padding: 24-32px

### 4. Cascading Photography + Pull Quote

Two-column layout. Emotional turning point of the page.

Left column (~45%): 2-3 overlapping/cascading piano photographs, staggered vertically and offset horizontally. Uses Image2.jpg, Image3.jpg, Image4.jpg. Warm light, high contrast.

Right column (~55%): Pull quote.
- Quote: "What's the one thing that sounds off that I can't hear myself?" -- Lora italic, ~2.25rem
- Attribution: "The question every pianist asks." -- DM Sans, secondary color

Mobile: photos stack above the quote, single column.
Background: espresso (#2D2926). The photos provide all visual energy.

### 5. Final CTA

Centered. Generous whitespace (128px+ vertical padding).

- Headline: "Start practicing with a teacher who's always listening." -- Lora, display-xl, warm cream
- CTA: pill button
- Note: "Free on iPhone." -- DM Sans, tertiary color

### 6. Footer

Single row, three elements.

```
[crescend]     Built on published research. Read the paper.     2026
```

- Logo: Lora, warm cream
- Research: DM Sans, tertiary color. "Read the paper" link in secondary color, subtle underline on hover
- Copyright: DM Sans, tertiary color
- No social links, no extra navigation

---

## Responsive Behavior

Mobile (< 768px):
- Hero headline scales down via clamp() to ~3rem minimum
- Feature cards collapse to single column
- Cascading photos stack vertically above the quote
- Section padding reduces to 48-80px
- Footer stacks vertically, center-aligned

---

## Assets

Existing images in /public/:
- Image1.jpg -- grand piano from above (hero)
- Image2.jpg -- sheet music on piano keys (cascading section)
- Image3.jpg -- piano score with markings (cascading section)
- Image4.jpg -- hands on piano (cascading section)

---

## What Changes From Current Implementation

- Remove: SocialProofBar, ProblemSection, HowItWorksSection, FeedbackSection, ResearchSection (separate section)
- Remove: nav links (How It Works, Analyze, Paper)
- Add: Feature cards grid (new)
- Add: Cascading photography + pull quote section (new)
- Replace: hero with full-bleed photography hero
- Replace: color palette (#1C1A18 -> #2D2926, token names change)
- Replace: CTA button style (square -> pill)
- Simplify: footer (3-column -> single row)
- Keep: typography families (Lora + DM Sans), type scale, spacing system

## What Does NOT Change

- Framework: TanStack Start
- Deployment: Cloudflare Workers
- Copy voice: literary, restrained
- Font families: Lora + DM Sans
- Photos: same image assets
