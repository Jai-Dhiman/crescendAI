# Landing Page Design

**crescend.ai**
Version 2.0 -- March 2026
Dark mode. Chat-based product showcase. Warm cream + charcoal palette.

---

## Overview

The Crescend landing page is the first touchpoint for prospective users. It communicates what the product is, how it feels to use, and why it exists -- without relying on feature lists or statistics.

The page uses a dark charcoal palette with warm cream typography to create a cinematic, focused atmosphere that mirrors the feeling of a late-night practice session. The product is shown through a conversational interface mockup, not screenshots or feature grids.

---

## Design Principles

### 1. Typography carries the message

The headline is the hero. Large serif type at 60-120px desktop fills the viewport and communicates confidence. No illustrations, no abstract shapes -- just the words and the space around them.

### 2. Show, don't tell

Instead of describing features, the page shows a live chat interaction: a student plays, asks "how was that?" and the teacher responds with one specific, grounded observation. The product sells itself through the quality of the feedback.

### 3. Photography as architecture

Full-bleed piano photography breaks up the page between sections. Images are compositional elements -- not decoration. Think Steinway print ads: warm light, tight crops, real texture.

### 4. Restraint signals quality

No stats bar. No social proof carousel. No feature comparison grid. Research credibility appears as a quiet footnote in the footer. Generous negative space communicates that we have nothing to prove.

---

## Color Palette

The landing page uses the dark palette from the design system. The accent color (Warm Cream) doubles as the primary text color on dark backgrounds.

| Swatch | Token | Hex | Use |
|--------|-------|-----|-----|
| | Charcoal | #1C1A18 | Page background |
| | Surface | #262422 | Elevated cards and panels |
| | Surface 2 | #302E2B | Secondary elevation, chat bubbles |
| | Border | #3A3836 | Subtle dividers, 1px lines |
| | Warm Cream | #F0E6D3 | Primary text + accent, CTA background |
| | Muted | #9A9590 | Secondary text, captions |
| | Tertiary | #6B6560 | Disabled text, hints, timestamps |

---

## Typography

### Display (Lora, serif)

Used for all headlines, the pull quote, and the chat mockup piece title. Set large, tracked tight (-0.03em on hero), with text-wrap: balance.

### Body (DM Sans, sans-serif)

Used for all body text, navigation, buttons, labels, and the chat interface text. Comfortable line-height (1.6-1.75) at 14-18px.

| Element | Font | Size | Weight |
|---------|------|------|--------|
| Hero headline | Lora | clamp(3rem, 8vw, 7rem) | 400 |
| Section headline | Lora | clamp(2rem, 4vw, 3.5rem) | 400 |
| Pull quote | Lora | clamp(1.5rem, 3vw, 2.25rem) | 400 italic |
| Body text | DM Sans | 1.0625rem (17px) | 400 |
| Navigation / labels | DM Sans | 0.875rem (14px) | 400-500 |
| Overline / caption | DM Sans | 0.6875rem (11px) | 500, uppercase |
| CTA button | DM Sans | 0.9375rem (15px) | 500 |

---

## Page Structure

The page flows through seven sections. Each serves a distinct purpose in the narrative arc: intrigue, demonstrate, explain, inspire, convert.

| Section | Description |
|---------|-------------|
| **1. Navigation** | Fixed top bar. Logo (Lora, left) + "How It Works" link + "Start Practicing" CTA (right). Backdrop blur on scroll. Minimal, no center content. |
| **2. Hero** | Full viewport height. Headline: "A teacher for every pianist." at display-3xl. Subhead: one sentence describing the core loop. CTA button + "Free on iPhone" note. Subtle piano photography behind, fading through gradients. |
| **3. Image Break** | Full-bleed photography placeholder. ~50vh height. Warm-lit piano detail. Edge gradients fade into charcoal. Caption in bottom-right corner. |
| **4. Product Showcase** | Overline: "How It Feels." Headline: "Like a private lesson, every time you practice." Body paragraph. Below: chat interface mockup showing a full conversation -- listening state, student question, teacher observation, follow-up exercise with rendered notation. |
| **5. How It Works** | Two-column grid. Left: headline + subtitle. Right: three numbered steps (Open and play / Ask "how was that?" / Practice with focus). Large ghosted step numbers. |
| **6. Pull Quote** | Centered serif italic: "What's the one thing that sounds off that I can't hear myself?" Attribution: "The question every pianist asks." Breathing room above and below. |
| **7. Final CTA** | Large serif headline: "Start practicing with a teacher who's always listening." CTA button + note. Generous padding. |

---

## Chat Mockup Detail

The product showcase is the most important section on the page. It replaces the traditional feature grid or screenshot with a live conversation that demonstrates the product's value.

### Structure

The mockup is styled as a dark card (#262422 surface) with a 1px border. It contains:

1. **Header bar** -- green dot (listening), status text, piece name (Lora, right-aligned)
2. **Listening indicator** -- animated waveform bars + italic caption
3. **Student message** -- right-aligned, cream-tinted background: "How was that?"
4. **Teacher response** -- left-aligned, surface-2 background, 2px cream left border. One specific observation referencing bar numbers.
5. **Follow-up** -- student asks for a practice exercise
6. **Exercise card** -- inline card with notation placeholder, practice instructions, and "Start Focus Mode" link
7. **Input bar** -- text field + send button at the bottom

### Example conversation content

*Piece: Ballade No. 1 in G minor, Chopin*

**Student:** "How was that?"

**Teacher:** "The crescendo in the second phrase peaked too early -- the sforzando at bar 28 didn't land because you were already at forte by bar 24. Try holding back the build longer. Start the crescendo softer than you think you need to."

**Student:** "Can you give me something to practice for that?"

**Teacher:** [Exercise card] Dynamic Control -- Bars 20-28. Isolate the left hand. Play at three dynamic levels: pp, mf, ff. Then play as written, feeling the crescendo as one continuous gesture.

---

## Photography Requirements

Two image break sections punctuate the page. These are not decorative -- they are architectural elements that create rhythm and breathing room.

### Image Break 1 (after hero)

- Subject: hands on keys, or close-up of grand piano interior
- Height: ~50vh (minimum 360px)
- Treatment: warm directional light, high contrast, slightly desaturated
- Edge treatment: gradient fade to charcoal on left and right edges

### Image Break 2 (after How It Works)

- Subject: sheet music detail, or practice room atmosphere
- Height: ~40vh (minimum 280px)
- Same treatment as Image Break 1

### Hero background

- Subtle piano photography visible behind the headline text
- Heavy gradient overlay: 30% opacity at top, fading to 95% at bottom, 100% at page background
- The text must remain fully legible -- the image is atmosphere, not content

---

## Component Specifications

### Primary CTA button

- Background: #F0E6D3 (Warm Cream)
- Text: #1C1A18 (Charcoal), DM Sans 15px, weight 500
- Padding: 14px 32px. No border-radius.
- Hover: background lightens to #FFF5E6, subtle translateY(-1px)

### Navigation bar

- Fixed position. Backdrop blur (20px). Background: charcoal at 80% opacity.
- Logo: Lora 20px, weight 500, left-aligned
- Links: DM Sans 14px, secondary color. CTA button on the right.

### Chat mockup card

- Background: #262422 (Surface). Border: 1px solid #3A3836.
- Max-width: 640px. No border-radius.
- Teacher messages: left-aligned, Surface 2 background, 2px cream left border

---

## Footer & Research

The footer is the only place research credentials appear. The approach is understated -- a single line, not a dedicated section.

### Footer layout

Three-column flex row: logo (left), navigation links + research line (center), copyright (right).

### Research line

Text: "Built on published research." followed by a link: "Read the paper" pointing to the arXiv URL.

Font: DM Sans 12px, tertiary color. The link uses secondary color with a 1px bottom border at 30% opacity, darkening on hover.

No statistics (55%, 30+, 15s). No social proof bar. No logos. No testimonials. The research stands on its own.

---

## Responsive Behavior

### Mobile (< 768px)

- Hero headline scales down via clamp() to ~3rem minimum
- How It Works collapses to single column
- Chat mockup goes edge-to-edge (no side borders)
- Image breaks reduce to 35vh / 240px minimum
- Section padding reduces to 48-80px vertical
- Footer stacks vertically, center-aligned

---

## Copy Reference

All page copy for implementation reference.

| Location | Copy |
|----------|------|
| Nav CTA | Start Practicing |
| Hero headline | A teacher for every pianist. |
| Hero subhead | Play. Ask how it sounded. Get the one thing a great teacher would tell you -- specific, grounded, yours. |
| Hero note | Free on iPhone. No account required. |
| Showcase overline | How It Feels |
| Showcase headline | Like a private lesson, every time you practice. |
| Showcase body | Not a report card. A conversation. Your phone listens while you play. When you pause and ask, your teacher responds with the one thing that matters most right now. |
| How It Works headline | Your phone on the piano. That's it. |
| How It Works subtitle | No MIDI keyboard. No sheet music to upload. Your phone listens while you play -- and when you ask, your teacher is ready. |
| Step 1 | Open the app and play |
| Step 2 | Ask "how was that?" |
| Step 3 | Practice with focus |
| Pull quote | What's the one thing that sounds off that I can't hear myself? |
| Final CTA headline | Start practicing with a teacher who's always listening. |
| Footer research | Built on published research. Read the paper |
