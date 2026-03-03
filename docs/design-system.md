# CrescendAI Design System

**Mood:** "The practice room after midnight"
Dark, focused, still. The space where a serious pianist sits alone with their instrument and does the real work. No distractions. No decorations. Just the piano and the conversation.

---

## Surfaces & Modes

| Surface | Mode | Background | Use |
|---------|------|------------|-----|
| Landing page | Dark | #1C1A18 (charcoal) | Marketing, first impression |
| Web app | Light | #FDF8F0 (warm cream paper) | Daily-use product interface |
| iOS app | Light | systemBackground (warm-tinted) | Daily-use product interface |

The landing page is cinematic and emotional. The product is comfortable and functional.
They share typography, accent color, spacing rhythm, and signature patterns.

---

## Color Tokens

### Accent color: Warm Cream

The accent is Warm Cream (#F0E6D3) -- the color of aged paper and ivory keys. On dark surfaces, it serves as the primary highlight. On light surfaces, it recedes and the charcoal text carries authority.

```
--accent:             #F0E6D3    /* warm cream, ivory keys */
--accent-hover:       #FFF5E6    /* lighter on hover (dark mode) */
--accent-muted:       rgba(240, 230, 211, 0.12) /* for backgrounds/highlights */
--accent-on-light:    #1C1A18    /* charcoal used as accent on light backgrounds */
```

### Dark palette (landing page)

```
--dark-bg:            #1C1A18    /* charcoal */
--dark-surface:       #262422    /* elevated cards/panels */
--dark-surface-2:     #302E2B    /* secondary elevation */
--dark-border:        #3A3836    /* subtle dividers */
--dark-text-primary:  #F0E6D3    /* warm cream -- accent IS the text */
--dark-text-secondary: #9A9590   /* muted caption text */
--dark-text-tertiary:  #6B6560   /* disabled/hint text */
```

### Light palette (web app, iOS app)

```
--light-bg:           #FDF8F0    /* warm cream paper */
--light-surface:      #F7F2EB    /* cards, panels */
--light-surface-2:    #EDE9E2    /* secondary elevation */
--light-border:       #E0DCD5    /* subtle dividers */
--light-text-primary: #1C1A18    /* charcoal */
--light-text-secondary: #605B56  /* muted body text */
--light-text-tertiary:  #9A9590  /* disabled/hint text */
```

### No second accent color

Richness comes from photography, typography, and light -- not from additional colors. The only "color" in the system is the interplay between warm cream and charcoal.

---

## Typography

### Font families

| Role | Family | Use |
|------|--------|-----|
| Display | Lora | Headlines, hero text, section titles |
| Body/UI | DM Sans | Navigation, body text, buttons, labels, app UI |

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

### 1. Typography-as-hero

The landing page hero uses the headline as the visual centerpiece. "A teacher for every pianist." at display-3xl fills the viewport width. Piano photography visible behind/through the text as a subtle overlay. No competing elements above the fold.

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

Research credibility communicated through restraint, not statistics. A quiet line near the footer: "Built on published research. Validated by 30+ educators." with a link to the paper. No stats section, no percentage callouts, no social proof bar.

---

## Component Patterns

### Buttons

Primary CTA (dark mode): warm cream background, charcoal text, no border-radius (or very subtle 2px).
```
bg: var(--accent)        /* #F0E6D3 */
text: var(--dark-bg)     /* #1C1A18 */
hover: var(--accent-hover) /* #FFF5E6 */
padding: 12px 24px
font: body-sm, weight 500
```

Primary CTA (light mode): charcoal background, cream text.
```
bg: var(--accent-on-light) /* #1C1A18 */
text: var(--light-bg)      /* #FDF8F0 */
hover: #2A2826
padding: 12px 24px
font: body-sm, weight 500
```

Secondary: ghost style, border only.
```
border: 1px solid var(--dark-border) or var(--light-border)
text: current text color
hover: slight background fill
```

### Chat bubbles (product interface)

Teacher messages: left-aligned, light surface background, sans-serif body text.
Student messages: right-aligned, subtle charcoal background, cream text.
Inline exercises: full-width card within the chat flow, rendered notation with instructions below.

### Cards

Dark surface cards on landing page. Light surface cards in product.
Subtle border (1px, low-opacity). No heavy shadows.
Content-forward: the card disappears, the content speaks.

### Navigation

Landing page: minimal. Logo left, 2-3 text links right. Fixed on scroll with subtle backdrop blur.
Web app: sidebar or top bar with sans-serif labels, charcoal active indicator.
iOS app: tab bar with simple icons, charcoal tint for active state.

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
