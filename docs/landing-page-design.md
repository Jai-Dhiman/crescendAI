# Landing Page Design

**crescend.ai**
Version 3.0 -- March 2026
Dark mode. Full-bleed photography. Espresso + cream palette.

---

## Overview

The Crescend landing page is the first touchpoint for prospective users. It communicates what the product is, how it feels to use, and why it exists -- without relying on feature lists or statistics.

The page uses a dark espresso palette (#2D2926) with warm cream typography (#FDF8F0) to create a cinematic, focused atmosphere that mirrors the feeling of a late-night practice session. The product is shown through evocative photography, a feature card grid, and a cascading editorial photo section -- not screenshots or feature grids.

---

## Design Principles

### 1. Photography as hero

The landing hero uses a full-viewport piano photograph with the headline overlaid. The image does the emotional work; the typography does the conceptual work. No competing elements above the fold.

### 2. Show the feeling, not the features

Instead of listing features, three dark surface cards hint at what the product does. The emphasis is on how it feels ("Your teacher is listening") rather than technical capability.

### 3. Photography as architecture

Cascading editorial photography breaks up the page and creates visual rhythm. Images overlap and layer, creating depth. Think Steinway print ads: warm light, tight crops, real texture.

### 4. Restraint signals quality

No stats bar. No social proof carousel. No feature comparison grid. Research credibility appears as a quiet line in the footer. Generous negative space communicates that we have nothing to prove.

---

## Color Palette

The landing page uses the dark palette from the design system. Cream doubles as the primary text color on the espresso background.

| Token | Hex | Use |
|-------|-----|-----|
| Espresso | #2D2926 | Page background |
| Cream | #FDF8F0 | Primary text, CTA background |
| Surface | #3A3633 | Elevated cards and panels |
| Surface 2 | #454140 | Secondary elevation, placeholder visuals |
| Border | #504B48 | Subtle dividers, card borders |
| Text Secondary | #A8A29E | Muted body text, descriptions |
| Text Tertiary | #78716C | Disabled text, hints, timestamps |

---

## Typography

### Display (Lora, serif)

Used for all headlines, the pull quote, and card titles. Set large, tracked tight (-0.03em on hero), with text-wrap: balance.

### Body (DM Sans, sans-serif)

Used for all body text, navigation, buttons, labels. Comfortable line-height (1.6-1.75) at 14-18px.

| Element | Font | Size | Weight |
|---------|------|------|--------|
| Hero headline | Lora | clamp(3rem, 8vw, 7rem) | 400 |
| Section headline | Lora | display-md to display-xl responsive | 500 |
| Pull quote | Lora | display-md to display-lg responsive | 400 italic |
| Card title | Lora | display-sm (1.875rem) | 500 |
| Body text | DM Sans | body-md (1rem) | 400 |
| Navigation / CTA | DM Sans | body-sm (0.875rem) | 500 |
| Footer text | DM Sans | body-xs (0.75rem) | 400 |

---

## Page Structure

The page flows through six sections. Each serves a distinct purpose in the narrative arc: captivate, demonstrate, move, convert.

| Section | Description |
|---------|-------------|
| **1. Navigation** | Fixed top bar. Logo "crescend" (Lora, left) + "Start Practicing" pill CTA (right). Backdrop blur on scroll. Minimal, no nav links. |
| **2. Hero** | Full viewport height. Full-bleed piano photo (Image1.jpg) with gradient overlay (espresso, bottom-heavy). Headline: "A teacher for every pianist." centered at clamp(3rem, 8vw, 7rem). Pill CTA below. No subhead. |
| **3. Feature Cards** | Three dark surface cards in a responsive grid (3-col desktop, 1-col mobile). Each card: placeholder visual area (4:3 aspect, surface-2 bg) + title (display-sm) + description (body-md, text-secondary). Cards: "Your teacher is listening", "Exercises built for you", "See what you hear". |
| **4. Cascading Photos + Pull Quote** | Two-column layout (5fr/6fr). Left: three overlapping piano photographs (Image2, Image3, Image4) with absolute positioning and shadow. Right: serif italic pull quote: "What's the one thing that sounds off that I can't hear myself?" + attribution. |
| **5. Final CTA** | Large serif headline: "Start practicing with a teacher who's always listening." Pill CTA + "Free on iPhone." note. Generous vertical padding (128-160px). |
| **6. Footer** | Single flex row: logo (left), research line with paper link (center), copyright year (right). Stacks vertically on mobile. |

---

## Photography Requirements

### Hero background (Image1.jpg)

- Subject: grand piano seen from above
- Treatment: full-bleed, object-cover, fills viewport
- Overlay: gradient from espresso/80 at bottom through espresso/30 mid to espresso/10 top
- The text must remain fully legible -- the image is atmosphere, not content

### Cascading photos (Image2.jpg, Image3.jpg, Image4.jpg)

- Subject: sheet music on keys, piano score with markings, hands playing piano
- Treatment: rounded corners (rounded-lg), shadow-2xl, absolute positioned with overlap
- Layout: first image top-left (w-3/5, 4:5), second offset right and down (w-3/5, 4:5), third lower-left (w-1/2, 1:1)
- On mobile: the container collapses and images stack above the quote

---

## Component Specifications

### Primary CTA button (pill)

- Background: #FDF8F0 (Cream)
- Text: #2D2926 (Espresso), DM Sans body-sm, weight 500
- Padding: 14px 32px. Border-radius: 100px (full pill).
- Hover: brightness(1.1) filter
- Used in: nav header, hero section, final CTA section

### Navigation bar

- Fixed position. Backdrop blur (20px). Background: espresso at 80% opacity.
- Height: 64px.
- Logo: Lora 18px (text-lg), cream, tracking-tight, left-aligned
- CTA: pill button, right-aligned.

### Feature cards

- Background: surface (#3A3633). Border: 1px solid border (#504B48). Rounded-xl.
- Visual area: 4:3 aspect ratio, surface-2 background (placeholder for product screenshots).
- Text area: p-6 to p-8. Title in display-sm cream, description in body-md text-secondary.

---

## Footer & Research

The footer is the only place research credentials appear. A single line, not a dedicated section.

### Footer layout

Single flex row: logo (left), research line (center), year (right). On mobile, stacks vertically with center alignment.

### Research line

Text: "Built on published research." followed by a link: "Read the paper" pointing to the arXiv URL. Link uses text-secondary color with underline, hovering to cream.

No statistics. No social proof bar. No logos. No testimonials.

---

## Responsive Behavior

### Mobile (< 768px)

- Hero headline scales down via clamp() to ~3rem minimum
- Feature cards collapse to single column
- Cascading photos stack above pull quote (single column)
- Footer stacks vertically, center-aligned
- Section padding reduces to 96px vertical

---

## Copy Reference

All page copy for implementation reference.

| Location | Copy |
|----------|------|
| Nav logo | crescend |
| Nav CTA | Start Practicing |
| Hero headline | A teacher for every pianist. |
| Hero CTA | Start Practicing |
| Card 1 title | Your teacher is listening |
| Card 1 body | Your phone listens while you play. When you pause and ask, your teacher is ready with the one thing that matters most. |
| Card 2 title | Exercises built for you |
| Card 2 body | Not generic drills. Targeted practice for the specific passage and skill your teacher identified. |
| Card 3 title | See what you hear |
| Card 3 body | The score lights up on a piano keyboard. See the notes, the fingering, the dynamics -- then play along. |
| Pull quote | What's the one thing that sounds off that I can't hear myself? |
| Pull quote attribution | The question every pianist asks. |
| Final CTA headline | Start practicing with a teacher who's always listening. |
| Final CTA button | Start Practicing |
| Final CTA note | Free on iPhone. |
| Footer logo | crescend |
| Footer research | Built on published research. Read the paper |
| Footer year | 2026 |
