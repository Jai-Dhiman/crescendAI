# Landing Page Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the crescend.ai landing page with a dark espresso palette, full-bleed photography hero, feature card grid, and cascading editorial photography section.

**Architecture:** Complete rewrite of the landing page (`apps/web/src/routes/index.tsx`), root layout (`__root.tsx`), and stylesheet (`app.css`). The page flips from light mode (cream bg) to dark mode (espresso bg). CSS is rewritten from scratch using Tailwind v4 best practices -- design tokens in `@theme`, all layout via utility classes in JSX, minimal custom CSS.

**Tech Stack:** TanStack Start, React 19, Tailwind CSS v4, Cloudflare Workers

**Design doc:** `docs/plans/2026-03-04-landing-page-redesign-design.md`

---

### Task 1: Rewrite the stylesheet from scratch

**Files:**
- Rewrite: `apps/web/src/styles/app.css`
- Modify: `apps/web/src/routes/__root.tsx` (add DM Sans font loading)

**Context:** The current stylesheet has legacy CSS classes from the Leptos era (`.editorial-bleed`, `.editorial-bleed-text`, `.editorial-bleed-image`, `.container-editorial`, `.editorial-rule`, `.btn-primary`, `.btn-primary-inverted`). We're replacing all of this with a clean `@theme` block and minimal custom CSS. Layout will be handled by Tailwind utilities in JSX.

**Step 1: Rewrite `apps/web/src/styles/app.css`**

Replace the entire file with:

```css
@import 'tailwindcss' source('../');

@theme {
  /* Espresso/Cream palette */
  --color-espresso: #2D2926;
  --color-cream: #FDF8F0;

  /* Dark surface scale (landing page) */
  --color-surface: #3A3633;
  --color-surface-2: #454140;
  --color-border: #504B48;

  /* Text on dark */
  --color-text-primary: #FDF8F0;
  --color-text-secondary: #A8A29E;
  --color-text-tertiary: #78716C;

  /* Typography */
  --font-display: 'Lora', Georgia, serif;
  --font-sans: 'DM Sans', system-ui, sans-serif;

  /* Display type scale */
  --text-display-3xl: 6rem;
  --text-display-3xl--line-height: 1;
  --text-display-3xl--letter-spacing: -0.03em;
  --text-display-3xl--font-weight: 400;

  --text-display-xl: 3.75rem;
  --text-display-xl--line-height: 1.05;
  --text-display-xl--letter-spacing: -0.02em;
  --text-display-xl--font-weight: 500;

  --text-display-lg: 3rem;
  --text-display-lg--line-height: 1.1;
  --text-display-lg--letter-spacing: -0.01em;
  --text-display-lg--font-weight: 500;

  --text-display-md: 2.25rem;
  --text-display-md--line-height: 1.15;
  --text-display-md--letter-spacing: -0.01em;
  --text-display-md--font-weight: 500;

  --text-display-sm: 1.875rem;
  --text-display-sm--line-height: 1.2;
  --text-display-sm--font-weight: 500;

  /* Body type scale */
  --text-body-lg: 1.125rem;
  --text-body-lg--line-height: 1.75;

  --text-body-md: 1rem;
  --text-body-md--line-height: 1.75;

  --text-body-sm: 0.875rem;
  --text-body-sm--line-height: 1.6;

  --text-body-xs: 0.75rem;
  --text-body-xs--line-height: 1.5;

  --text-label-sm: 0.6875rem;
  --text-label-sm--line-height: 1.4;
  --text-label-sm--font-weight: 500;
  --text-label-sm--letter-spacing: 0.075em;

  /* Animations */
  --animate-fade-in-up: fade-in-up 600ms cubic-bezier(0.16, 1, 0.3, 1) both;

  @keyframes fade-in-up {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
  }
}

@layer base {
  html {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
    scroll-behavior: smooth;
  }

  body {
    background-color: var(--color-espresso);
    color: var(--color-text-primary);
    font-family: var(--font-sans);
    line-height: 1.75;
  }

  h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-display);
    text-wrap: balance;
  }

  p {
    text-wrap: pretty;
  }

  ::selection {
    background-color: var(--color-surface-2);
    color: var(--color-cream);
  }
}
```

**What was removed and why:**
- `.container-editorial` -- replaced by Tailwind `max-w-6xl mx-auto px-6` (or similar) in JSX
- `.editorial-bleed`, `.editorial-bleed-text`, `.editorial-bleed-image` -- replaced by grid utilities in JSX
- `.editorial-rule` -- removed entirely (no horizontal rules in new design)
- `.btn-primary`, `.btn-primary-inverted` -- replaced by utility classes in JSX (pill button is just `bg-cream text-espresso rounded-full px-8 py-3.5 font-medium`)
- `.texture-grain` -- removed (the grain overlay doesn't fit the new design)
- `.stagger` -- kept as `@keyframes` in theme, can be applied via utilities
- All `clay-*`, `paper-*`, `ink-*` color tokens -- replaced by the new espresso/cream/surface palette

**Step 2: Add DM Sans font loading in `apps/web/src/routes/__root.tsx`**

Find the `links` array in the `head()` function. Add DM Sans alongside Lora:

```tsx
links: [
  { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
  {
    rel: 'preconnect',
    href: 'https://fonts.gstatic.com',
    crossOrigin: 'anonymous',
  },
  {
    rel: 'stylesheet',
    href: 'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap',
  },
  { rel: 'stylesheet', href: appCss },
  { rel: 'icon', type: 'image/png', href: '/crescendai.png' },
],
```

**Step 3: Verify the build compiles**

Run: `cd apps/web && bun run build`
Expected: Build succeeds. The page will look broken in the browser because JSX still references old class names -- that's expected and gets fixed in subsequent tasks.

**Step 4: Commit**

```bash
git add apps/web/src/styles/app.css apps/web/src/routes/__root.tsx
git commit -m "rewrite stylesheet for espresso/cream dark palette

Replace legacy clay/paper/ink tokens with espresso/cream/surface palette.
Remove all custom CSS layout classes (editorial-bleed, container-editorial,
btn-primary). Add DM Sans font loading. All layout moves to Tailwind
utilities in JSX (subsequent commits)."
```

---

### Task 2: Rewrite the root layout (Header + Footer + body wrapper)

**Files:**
- Rewrite: `apps/web/src/routes/__root.tsx`

**Context:** The current root layout has a complex 3-column Header with nav links, a large centered logo, and a tagline. The Footer mirrors this. The body wrapper uses `bg-paper-50` (light). All of this changes: the body is dark, the Header becomes a minimal fixed nav, and the Footer becomes a single row.

**Step 1: Rewrite the `RootDocument` component**

The body wrapper changes from light to dark. Remove `texture-grain`. The Header/Footer are now embedded in the landing page itself (not the root layout) because the `/analyze` route needs different treatment.

Actually -- since the nav and footer are consistent across pages, keep them in root. But simplify dramatically.

```tsx
function RootDocument() {
  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body className="bg-espresso text-text-primary font-sans">
        <Header />
        <main>
          <Outlet />
        </main>
        <Footer />
        <Scripts />
      </body>
    </html>
  )
}
```

**Step 2: Rewrite the `Header` component**

Minimal fixed nav: logo left, CTA right. Backdrop blur on scroll.

```tsx
function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-espresso/80">
      <div className="max-w-7xl mx-auto px-6 lg:px-12 flex items-center justify-between h-16">
        <a href="/" className="font-display text-lg text-cream tracking-tight">
          crescend
        </a>
        <a
          href="/analyze"
          className="bg-cream text-espresso rounded-full px-6 py-2 text-body-sm font-medium hover:brightness-110 transition"
        >
          Start Practicing
        </a>
      </div>
    </header>
  )
}
```

**Step 3: Rewrite the `Footer` component**

Single row: logo, research footnote, copyright.

```tsx
function Footer() {
  return (
    <footer className="py-12 lg:py-16">
      <div className="max-w-7xl mx-auto px-6 lg:px-12">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6 text-body-xs text-text-tertiary">
          <a href="/" className="font-display text-sm text-cream tracking-tight">
            crescend
          </a>
          <p>
            Built on published research.{' '}
            <a
              href="https://arxiv.org/abs/2601.19029"
              target="_blank"
              rel="noopener"
              className="text-text-secondary underline underline-offset-2 hover:text-cream transition-colors"
            >
              Read the paper
            </a>
          </p>
          <p>2026</p>
        </div>
      </div>
    </footer>
  )
}
```

**Step 4: Verify dev server renders**

Run: `cd apps/web && bun run dev`
Expected: Dark background, minimal nav at top, minimal footer at bottom. The main content area will be broken (old class names) -- that's Task 3.

**Step 5: Commit**

```bash
git add apps/web/src/routes/__root.tsx
git commit -m "rewrite root layout with minimal dark nav and footer

Fixed header with backdrop blur, logo + CTA only. Single-row footer
with research footnote. Dark espresso body background."
```

---

### Task 3: Rewrite the hero section

**Files:**
- Modify: `apps/web/src/routes/index.tsx`

**Context:** Replace the entire `LandingPage` component and `HeroSection`. Remove all old section components (SocialProofBar, ProblemSection, HowItWorksSection, FeedbackSection, ResearchSection, FinalCtaSection). Build the hero first, add other sections in subsequent tasks.

**Step 1: Replace `index.tsx` with the hero section only**

Replace the entire file:

```tsx
import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/')({ component: LandingPage })

function LandingPage() {
  return (
    <div>
      <HeroSection />
    </div>
  )
}

function HeroSection() {
  return (
    <section className="relative h-screen flex items-center justify-center overflow-hidden">
      {/* Full-bleed background image */}
      <img
        src="/Image1.jpg"
        alt="Grand piano seen from above"
        className="absolute inset-0 w-full h-full object-cover"
      />

      {/* Gradient overlay for text legibility */}
      <div className="absolute inset-0 bg-gradient-to-t from-espresso/80 via-espresso/30 to-espresso/10" />

      {/* Content */}
      <div className="relative z-10 text-center px-6">
        <h1
          className="font-display text-cream text-balance"
          style={{ fontSize: 'clamp(3rem, 8vw, 7rem)', lineHeight: 1.05, letterSpacing: '-0.03em' }}
        >
          A teacher for every pianist.
        </h1>

        <div className="mt-10">
          <a
            href="/analyze"
            className="bg-cream text-espresso rounded-full px-8 py-3.5 text-body-sm font-medium hover:brightness-110 transition inline-block"
          >
            Start Practicing
          </a>
        </div>
      </div>
    </section>
  )
}
```

**Step 2: Verify in dev server**

Run: `cd apps/web && bun run dev`
Expected: Full-viewport hero with Image1.jpg as background, "A teacher for every pianist." overlaid in large Lora serif, pill CTA button. Dark gradient at bottom fading to the espresso background. Nav floats above.

**Step 3: Commit**

```bash
git add apps/web/src/routes/index.tsx
git commit -m "rewrite hero section with full-bleed photography

Full-viewport piano photo with gradient overlay. Massive serif headline
centered. Pill CTA. All old sections removed (rebuilt in subsequent commits)."
```

---

### Task 4: Build the feature cards section

**Files:**
- Modify: `apps/web/src/routes/index.tsx`

**Context:** Add three dark surface cards in a responsive grid below the hero. Placeholder visuals for now (gradient backgrounds). Cards showcase future features: listening bar, custom exercises, keyboard guide.

**Step 1: Add `FeatureCardsSection` component to `index.tsx`**

Add after `HeroSection` in the `LandingPage` component and define:

```tsx
function FeatureCardsSection() {
  const cards = [
    {
      title: 'Your teacher is listening',
      description:
        'Your phone listens while you play. When you pause and ask, your teacher is ready with the one thing that matters most.',
    },
    {
      title: 'Exercises built for you',
      description:
        'Not generic drills. Targeted practice for the specific passage and skill your teacher identified.',
    },
    {
      title: 'See what you hear',
      description:
        'The score lights up on a piano keyboard. See the notes, the fingering, the dynamics -- then play along.',
    },
  ]

  return (
    <section className="py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-6 lg:px-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {cards.map((card) => (
            <div
              key={card.title}
              className="bg-surface border border-border rounded-xl overflow-hidden"
            >
              {/* Placeholder visual area */}
              <div className="aspect-[4/3] bg-surface-2" />

              {/* Text content */}
              <div className="p-6 lg:p-8">
                <h3 className="font-display text-display-sm text-cream mb-3">
                  {card.title}
                </h3>
                <p className="text-body-md text-text-secondary">
                  {card.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
```

Update `LandingPage`:
```tsx
function LandingPage() {
  return (
    <div>
      <HeroSection />
      <FeatureCardsSection />
    </div>
  )
}
```

**Step 2: Verify in dev server**

Expected: Three dark cards in a row on desktop, stacked on mobile. Each has a dark placeholder visual area at top and title + description below.

**Step 3: Commit**

```bash
git add apps/web/src/routes/index.tsx
git commit -m "add feature cards section with placeholder visuals

Three dark surface cards: listening bar, custom exercises, keyboard guide.
Placeholder gradient visuals, to be replaced with product screenshots."
```

---

### Task 5: Build the cascading photography + pull quote section

**Files:**
- Modify: `apps/web/src/routes/index.tsx`

**Context:** Two-column layout. Left: 2-3 overlapping/cascading photos (staggered with CSS transforms). Right: pull quote in Lora italic. This is the editorial signature of the page.

**Step 1: Add `CascadingQuoteSection` component**

```tsx
function CascadingQuoteSection() {
  return (
    <section className="py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-6 lg:px-12">
        <div className="grid grid-cols-1 lg:grid-cols-[5fr_6fr] gap-12 lg:gap-16 items-center">
          {/* Cascading photos */}
          <div className="relative h-[500px] lg:h-[600px]">
            <img
              src="/Image2.jpg"
              alt="Sheet music resting on piano keys"
              className="absolute top-0 left-0 w-3/5 rounded-lg object-cover shadow-2xl"
              style={{ aspectRatio: '4/5' }}
            />
            <img
              src="/Image3.jpg"
              alt="Piano score with dynamic markings"
              className="absolute top-[20%] left-[25%] w-3/5 rounded-lg object-cover shadow-2xl"
              style={{ aspectRatio: '4/5' }}
            />
            <img
              src="/Image4.jpg"
              alt="Hands playing piano in warm light"
              className="absolute top-[40%] left-[10%] w-1/2 rounded-lg object-cover shadow-2xl"
              style={{ aspectRatio: '1/1' }}
            />
          </div>

          {/* Pull quote */}
          <div>
            <blockquote className="font-display italic text-display-md lg:text-display-lg text-cream leading-snug">
              "What's the one thing that sounds off that I can't hear myself?"
            </blockquote>
            <p className="mt-6 text-body-md text-text-secondary">
              The question every pianist asks.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
```

Update `LandingPage`:
```tsx
function LandingPage() {
  return (
    <div>
      <HeroSection />
      <FeatureCardsSection />
      <CascadingQuoteSection />
    </div>
  )
}
```

**Step 2: Verify in dev server**

Expected: On desktop, overlapping photos on left with the pull quote on right. On mobile, photos stack above the quote. The photos should overlap naturally via absolute positioning.

**Step 3: Commit**

```bash
git add apps/web/src/routes/index.tsx
git commit -m "add cascading photography + pull quote section

Editorial photo layout with 3 overlapping piano images and serif italic
pull quote. Emotional turning point between feature cards and final CTA."
```

---

### Task 6: Build the final CTA section

**Files:**
- Modify: `apps/web/src/routes/index.tsx`

**Context:** Large centered headline + CTA + small note. Generous whitespace.

**Step 1: Add `FinalCtaSection` component**

```tsx
function FinalCtaSection() {
  return (
    <section className="py-32 lg:py-40">
      <div className="max-w-4xl mx-auto px-6 lg:px-12 text-center">
        <h2 className="font-display text-display-md lg:text-display-xl text-cream">
          Start practicing with a teacher who's always listening.
        </h2>

        <div className="mt-10">
          <a
            href="/analyze"
            className="bg-cream text-espresso rounded-full px-8 py-3.5 text-body-sm font-medium hover:brightness-110 transition inline-block"
          >
            Start Practicing
          </a>
          <p className="mt-4 text-body-xs text-text-tertiary">
            Free on iPhone.
          </p>
        </div>
      </div>
    </section>
  )
}
```

Update `LandingPage`:
```tsx
function LandingPage() {
  return (
    <div>
      <HeroSection />
      <FeatureCardsSection />
      <CascadingQuoteSection />
      <FinalCtaSection />
    </div>
  )
}
```

**Step 2: Verify in dev server**

Expected: Large serif headline, pill CTA, "Free on iPhone." note. Generous vertical padding.

**Step 3: Commit**

```bash
git add apps/web/src/routes/index.tsx
git commit -m "add final CTA section

Large serif headline with pill CTA and 'Free on iPhone' note.
Generous whitespace."
```

---

### Task 7: Update the /analyze route for dark theme

**Files:**
- Modify: `apps/web/src/routes/analyze.tsx`

**Context:** The /analyze page still references old `clay-*`, `ink-*`, `paper-*` class names. Update to use the new palette tokens.

**Step 1: Update `analyze.tsx`**

```tsx
import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/analyze')({ component: AnalyzePage })

function AnalyzePage() {
  return (
    <section className="pt-32 pb-24 md:pt-40 md:pb-32">
      <div className="max-w-2xl mx-auto px-6 text-center">
        <h1 className="font-display text-display-md md:text-display-lg text-cream mb-6">
          Analyze
        </h1>
        <p className="text-body-lg text-text-secondary mb-8">
          The analysis experience is moving to our iOS app -- record, listen,
          and get feedback right from your phone. Coming soon.
        </p>
        <a
          href="/"
          className="text-body-sm text-text-secondary underline underline-offset-2 hover:text-cream transition-colors"
        >
          Back to home
        </a>
      </div>
    </section>
  )
}
```

**Step 2: Verify both routes work**

Run: `cd apps/web && bun run dev`
Navigate to `/` and `/analyze`. Both should render on the dark espresso background with cream text.

**Step 3: Commit**

```bash
git add apps/web/src/routes/analyze.tsx
git commit -m "update analyze page for dark espresso palette"
```

---

### Task 8: Update design system and landing page docs

**Files:**
- Modify: `docs/design-system.md`
- Modify: `docs/landing-page-design.md`

**Context:** Both docs reference the old palette (#1C1A18 charcoal) and old page structure. Update to reflect the new espresso/cream palette and the new page sections.

**Step 1: Update `docs/design-system.md`**

Key changes:
- Landing page surface mode: `#2D2926` (espresso) replaces `#1C1A18` (charcoal)
- Accent color section: remove -- no accent color, just espresso/cream interplay
- Dark palette tokens: update all hex values to match new palette
- Light palette tokens: update to derive from `#2D2926` and `#FDF8F0`
- Button section: update to pill shape with `border-radius: 100px`
- Remove `.editorial-bleed` and other custom CSS class references
- Add note: "Body text uses DM Sans (previously Lora was used for both display and body)"

**Step 2: Update `docs/landing-page-design.md`**

Key changes:
- Color palette table: update all hex values
- Page structure table: replace 7 sections with new 6 (Nav, Hero, Feature Cards, Cascading Photos + Quote, Final CTA, Footer)
- Hero section: full-bleed photo, ultra-minimal copy
- Remove: chat mockup detail, social proof bar, problem section, how it works
- Add: feature cards specification, cascading photo layout specification
- Copy reference table: update to reflect new copy

**Step 3: Commit**

```bash
git add docs/design-system.md docs/landing-page-design.md
git commit -m "update design system and landing page docs for redesign

New espresso/cream palette, pill CTAs, feature cards grid, cascading
editorial photography. Docs now match the implemented landing page."
```

---

### Task 9: Final build verification and cleanup

**Files:**
- All modified files (verification only)

**Step 1: Run full build**

```bash
cd apps/web && bun run build
```

Expected: Build succeeds with no errors.

**Step 2: Run type check**

```bash
cd apps/web && npx tsc --noEmit
```

Expected: No type errors.

**Step 3: Visual verification in dev server**

Run: `cd apps/web && bun run dev`

Verify all six sections:
1. Nav: fixed, logo left, CTA right, backdrop blur on scroll
2. Hero: full-viewport photo, "A teacher for every pianist.", CTA
3. Feature cards: 3 cards, responsive grid
4. Cascading photos + pull quote: overlapping images, serif italic quote
5. Final CTA: large headline, CTA, "Free on iPhone."
6. Footer: logo, research link, copyright

Verify responsive behavior:
- Resize to mobile width (375px)
- Feature cards stack to 1 column
- Cascading photos stack above quote
- Footer stacks vertically
- Hero headline scales down

**Step 4: Commit any final adjustments**

If visual tweaks are needed (spacing, sizing, photo positions), make them and commit:

```bash
git add -A apps/web/
git commit -m "polish landing page spacing and responsive behavior"
```
