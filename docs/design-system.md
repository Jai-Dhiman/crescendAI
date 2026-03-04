# CrescendAI Design System

**Mood:** "The practice room after midnight"
Dark, focused, still. The space where a serious pianist sits alone with their instrument and does the real work. No distractions. No decorations. Just the piano and the conversation.

---

## Surfaces & Modes

| Surface | Mode | Background | Use |
|---------|------|------------|-----|
| Landing page | Dark | #2D2926 (espresso) | Marketing, first impression |
| Web app | Light | #FDF8F0 (warm cream) | Daily-use product interface |
| iOS app | Light | systemBackground (warm-tinted) | Daily-use product interface |

The landing page is cinematic and emotional. The product is comfortable and functional.
They share typography, spacing rhythm, and signature patterns.

---

## Color Tokens

### Espresso/Cream palette

The system uses two anchors: Espresso (#2D2926) and Cream (#FDF8F0). On dark surfaces, cream is the primary text and CTA background. On light surfaces, espresso carries authority. No additional accent color -- richness comes from photography, typography, and light.

### Dark palette (landing page)

```
--color-espresso:       #2D2926    /* page background */
--color-cream:          #FDF8F0    /* primary text + CTA background */
--color-surface:        #3A3633    /* elevated cards/panels */
--color-surface-2:      #454140    /* secondary elevation */
--color-border:         #504B48    /* subtle dividers, card borders */
--color-text-primary:   #FDF8F0    /* cream -- primary text on dark */
--color-text-secondary: #A8A29E    /* muted body text, descriptions */
--color-text-tertiary:  #78716C    /* disabled text, hints, timestamps */
```

### Light palette (web app, iOS app)

```
--light-bg:             #FDF8F0    /* warm cream */
--light-surface:        #F7F2EB    /* cards, panels */
--light-surface-2:      #EDE9E2    /* secondary elevation */
--light-border:         #E0DCD5    /* subtle dividers */
--light-text-primary:   #2D2926    /* espresso */
--light-text-secondary: #605B56    /* muted body text */
--light-text-tertiary:  #A8A29E    /* disabled/hint text */
```

---

## Typography

### Font families

| Role | Family | Use |
|------|--------|-----|
| Display | Lora | Headlines, hero text, section titles, pull quotes |
| Body/UI | DM Sans | Navigation, body text, buttons, labels, app UI |

Body text uses DM Sans. Previously Lora was used for both display and body.

### Display scale (serif)

```
--display-3xl:  6rem / 1.0  / -0.03em   /* Landing hero, 96px */
--display-xl:   3.75rem / 1.05 / -0.02em /* Section headlines */
--display-lg:   3rem / 1.1  / -0.01em   /* Sub-headlines */
--display-md:   2.25rem / 1.15 / -0.01em /* Card titles */
--display-sm:   1.875rem / 1.2           /* Mobile headlines */
```

### Body scale (sans-serif)

```
--body-lg:      1.125rem / 1.75    /* Lead paragraphs */
--body-md:      1rem / 1.75        /* Default body */
--body-sm:      0.875rem / 1.6     /* Secondary text, nav */
--body-xs:      0.75rem / 1.5      /* Captions, timestamps */
--label-sm:     0.6875rem / 1.4 / 500 / 0.075em tracking  /* Overlines, tags */
```

### Typography rules

- Headlines: serif, used LARGE (60-120px desktop), text-wrap balance
- Body: sans-serif, comfortable reading widths (max 65ch)
- In the product app: serif only for page/screen titles; everything else sans
- In the landing page: serif for all headlines and pull quotes; sans for body and nav

---

## Spacing

Base unit: 4px

```
--space-1:   4px
--space-2:   8px
--space-3:   12px
--space-4:   16px
--space-6:   24px
--space-8:   32px
--space-10:  40px
--space-12:  48px
--space-16:  64px
--space-20:  80px
--space-24:  96px
--space-32:  128px
```

Section padding: 80-128px vertical on desktop, 48-64px on mobile.
Generous negative space. Confidence is communicated through what you leave out.

---

## Signature Design Patterns

### 1. Full-bleed photography hero

The landing page hero uses a full-viewport piano photograph with a gradient overlay (espresso, bottom-heavy) for text legibility. The headline "A teacher for every pianist." is centered over the image in massive Lora serif. A pill CTA sits below. No subhead, no feature text -- the image and headline do the work.

### 2. The vinyl shelf

Practice session history displayed as physical objects (vinyl record sleeves) on warm wooden shelves. Records can be "pulled out" to reveal the conversation from that session. Used in:
- Web app: practice library (each sleeve opens a chat thread)
- iOS app: practice history

### 3. Chat-based interface

The primary interaction model is conversational. Like Claude's interface: messages flow down, the teacher responds with one observation at a time. Inline rendered content includes:
- Practice exercises rendered as notation (MusicXML/Lilypond -> visual)
- Focus mode exercises with instructions
- Waveform snippets referencing specific passages
No scores on screen. No radar charts. No dimension breakdowns visible during practice.

### 4. Photography over illustration

Every image is a real photograph: pianos, hands, sheet music, practice rooms. Warm-lit, slightly moody, high contrast. No illustrations, no abstract graphics, no 3D renders. This separates CrescendAI from the cartoon aesthetic of competitors.

On the landing page: striking, full-bleed piano photography. Not decorative -- compositional. The imagery should feel like a Steinway ad.

### 5. Understated research

Research credibility communicated through restraint, not statistics. A quiet line in the footer: "Built on published research." with a link to the paper. No stats section, no percentage callouts, no social proof bar.

---

## Component Patterns

### Buttons

Primary CTA: pill shape, cream background on dark, espresso text.
```
bg: var(--color-cream)       /* #FDF8F0 */
text: var(--color-espresso)  /* #2D2926 */
hover: brightness(1.1)
padding: 14px 32px
border-radius: 100px         /* full pill */
font: body-sm, weight 500
```

Primary CTA (light mode): espresso background, cream text, pill shape.
```
bg: var(--color-espresso)    /* #2D2926 */
text: var(--color-cream)     /* #FDF8F0 */
hover: #3A3633
padding: 14px 32px
border-radius: 100px
font: body-sm, weight 500
```

Secondary: ghost style, border only.
```
border: 1px solid var(--color-border)
text: current text color
hover: slight background fill
border-radius: 100px
```

### Chat bubbles (product interface)

Teacher messages: left-aligned, light surface background, sans-serif body text.
Student messages: right-aligned, subtle espresso background, cream text.
Inline exercises: full-width card within the chat flow, rendered notation with instructions below.

### Cards

Dark surface cards on landing page: `bg-surface` with `border border-border rounded-xl`. Content-forward: the card disappears, the content speaks.
Light surface cards in product.

### Navigation

Landing page: fixed top bar. Logo (Lora, left) + pill CTA (right). Backdrop blur on scroll. No center content, no nav links.
Web app: sidebar or top bar with sans-serif labels, espresso active indicator.
iOS app: tab bar with simple icons, espresso tint for active state.

---

## Imagery Guidelines

- Subject matter: grand pianos, hands on keys, sheet music, practice rooms, recital halls
- Lighting: warm, directional, golden hour quality. One light source preferred.
- Treatment: high contrast, slightly desaturated (not heavily graded)
- Crop: tight, editorial. Show texture and detail. Full-bleed where possible.
- No stock photography "feel." Everything should look like it was shot for CrescendAI.
- On the landing page: use photography as compositional elements, not decoration. Think Steinway or Bechstein print ads.

---

## What This Is NOT

- Not a color-blast music creation app (Suno, Splice)
- Not a gamified learning app (Simply Piano, Flowkey)
- Not a cold productivity tool (Notion, Linear)
- Not a report card or grading system
- It IS: a warm, serious, beautiful practice companion for people who care about playing well.
