# Landing Page Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the landing page from a SaaS-template feel to a warm, conversational, music-forward experience matching the investor pitch voice ("A teacher for every pianist").

**Architecture:** Pure frontend changes. Rewrite landing page copy and structure, swap display typography from Cormorant Garamond to Plus Jakarta Sans, replace flat white section backgrounds with warm amber gradients, remove paper texture overlay. No backend, route, or `/analyze` page changes.

**Tech Stack:** Rust/Leptos (WASM), Tailwind CSS, Google Fonts

**Design doc:** `apps/docs/landing-page-redesign.md`

---

### Task 1: Swap display typography

Replace Cormorant Garamond with Plus Jakarta Sans for a friendlier, warmer feel. Keep Source Serif 4 for body, Inter for UI, JetBrains Mono for code.

**Files:**
- Modify: `apps/web/src/app.rs:20-22` (Google Fonts `<Link>`)
- Modify: `apps/web/tailwind.config.js:93` (fontFamily.display)
- Modify: `apps/web/tailwind.css:43` (--font-display CSS variable)

**Step 1: Update Google Fonts link in app.rs**

Replace the Google Fonts `<Link>` href (line 21) to swap `Cormorant+Garamond` for `Plus+Jakarta+Sans`:

```rust
        <Link
            href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&display=swap"
            rel="stylesheet"
        />
```

**Step 2: Update Tailwind fontFamily config**

In `apps/web/tailwind.config.js` line 93, change:

```js
fontFamily: {
    display: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
    serif: ['"Source Serif 4"', 'Georgia', 'serif'],
    sans: ['"Inter"', 'system-ui', 'sans-serif'],
    mono: ['"JetBrains Mono"', 'Consolas', 'monospace'],
},
```

**Step 3: Update CSS variable**

In `apps/web/tailwind.css` line 43, change:

```css
--font-display: "Plus Jakarta Sans", system-ui, sans-serif;
```

**Step 4: Verify build compiles**

Run: `cd apps/web && cargo check`
Expected: compiles without errors (font is purely CSS, Rust code uses `font-display` class unchanged)

**Step 5: Commit**

```bash
git add apps/web/src/app.rs apps/web/tailwind.config.js apps/web/tailwind.css
git commit -m "swap display font from Cormorant Garamond to Plus Jakarta Sans"
```

---

### Task 2: Add warm gradient backgrounds and remove paper texture

Replace the flat white backgrounds and paper texture with warm amber gradients. Add new gradient utilities to the design system.

**Files:**
- Modify: `apps/web/tailwind.config.js:274-281` (backgroundImage)
- Modify: `apps/web/tailwind.css:306-318` (remove texture-paper, add gradients)
- Modify: `apps/web/src/app.rs:29` (remove `texture-paper` class from root div)

**Step 1: Add gradient background utilities to tailwind.config.js**

In the `backgroundImage` section (line 274), add these new gradients alongside existing ones:

```js
backgroundImage: {
    'gradient-sepia': 'linear-gradient(135deg, #a69276 0%, #8b7355 100%)',
    'gradient-sepia-subtle': 'linear-gradient(135deg, #f0e9dc 0%, #e3d5c0 100%)',
    'gradient-paper': 'linear-gradient(180deg, #fefdfb 0%, #f5f1e8 100%)',
    'gradient-warm': 'linear-gradient(180deg, #fbf9f5 0%, #ede6d9 100%)',
    'gradient-page': 'linear-gradient(180deg, #fefdfb 0%, #f5f1e8 100%)',
    'gradient-cta': 'linear-gradient(135deg, #8b7355 0%, #6e5a43 100%)',
    // New warm landing page gradients
    'gradient-hero': 'linear-gradient(180deg, #fdf8f0 0%, #f6e9d5 50%, #f0dfc4 100%)',
    'gradient-warm-mid': 'linear-gradient(180deg, #f6e9d5 0%, #eedcca 100%)',
    'gradient-warm-deep': 'linear-gradient(180deg, #eedcca 0%, #e8d5b8 100%)',
    'gradient-warm-rich': 'linear-gradient(180deg, #e8d5b8 0%, #dfc9a8 100%)',
    'noise': "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E\")",
},
```

**Step 2: Remove texture-paper from CSS**

In `apps/web/tailwind.css`, remove lines 306-318 (the `.texture-paper` and `.texture-paper::before` rules). Keep everything else.

**Step 3: Remove texture-paper from app.rs root div**

In `apps/web/src/app.rs` line 29, change:

```rust
<div class="min-h-screen bg-gradient-page flex flex-col texture-paper">
```

to:

```rust
<div class="min-h-screen bg-gradient-hero flex flex-col">
```

**Step 4: Verify build compiles**

Run: `cd apps/web && cargo check`

**Step 5: Commit**

```bash
git add apps/web/tailwind.config.js apps/web/tailwind.css apps/web/src/app.rs
git commit -m "add warm gradient backgrounds, remove paper texture overlay"
```

---

### Task 3: Rewrite landing page -- hero section

Replace the current hero with the new conversational copy from the investor pitch. Drop the social proof strip.

**Files:**
- Modify: `apps/web/src/pages/landing.rs:1-56` (LandingPage component + HeroSection)

**Step 1: Rewrite the top-level LandingPage and HeroSection**

Replace lines 1-56 of `apps/web/src/pages/landing.rs` with:

```rust
use leptos::prelude::*;

/// Conversational landing page for Crescend.
#[component]
pub fn LandingPage() -> impl IntoView {
    view! {
        <div>
            <HeroSection />
            <ProblemSection />
            <FeedbackShowcase />
            <HowItWorksStrip />
            <MissionSection />
            <FinalCtaSection />
        </div>
    }
}

// -- Hero -------------------------------------------------------------------

#[component]
fn HeroSection() -> impl IntoView {
    view! {
        <section class="relative overflow-hidden bg-gradient-hero">
            <div class="container-wide py-16 md:py-24 lg:py-32">
                <div class="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
                    // Left: Copy
                    <div class="max-w-xl animate-fade-in">
                        <h1 class="font-display text-display-xl md:text-display-2xl text-ink-900 tracking-tight mb-6">
                            "A teacher for every pianist."
                        </h1>
                        <p class="text-body-lg text-ink-600 mb-8 max-w-md">
                            "Record yourself playing. Get the feedback a great teacher would give you -- on your tone, your dynamics, your phrasing."
                        </p>
                        <a href="/analyze" class="btn-primary text-lg px-8 py-4">
                            "Try It Free"
                        </a>
                    </div>

                    // Right: Animated product preview
                    <div class="relative animate-fade-in" style="animation-delay: 200ms; animation-fill-mode: both">
                        <HeroProductPreview />
                    </div>
                </div>
            </div>
        </section>
    }
}
```

Note: `HeroProductPreview` and `HeroMockCard` components (lines 58-154) stay unchanged -- they still show the waveform and feedback card animation.

**Step 2: Verify build compiles**

Run: `cd apps/web && cargo check`
Expected: compiles (the component names referenced in LandingPage changed, so ensure all sub-components exist)

**Step 3: Commit**

```bash
git add apps/web/src/pages/landing.rs
git commit -m "rewrite hero with investor pitch copy, drop social proof strip"
```

---

### Task 4: Rewrite landing page -- problem section

Replace the 3-paragraph PAS framework section with 2 direct lines from the investor pitch.

**Files:**
- Modify: `apps/web/src/pages/landing.rs` (ProblemSection component, ~lines 156-175)

**Step 1: Rewrite ProblemSection**

Replace the existing ProblemSection with:

```rust
// -- Problem ----------------------------------------------------------------

#[component]
fn ProblemSection() -> impl IntoView {
    view! {
        <section class="bg-gradient-warm-mid">
            <div class="container-narrow text-center py-16 md:py-24">
                <p class="font-display text-heading-xl md:text-display-sm text-ink-800 leading-relaxed">
                    "Any app can tell you if you played the right notes. But that's not what separates good playing from great playing."
                </p>
                <p class="text-body-lg text-ink-700 font-medium mt-6 max-w-2xl mx-auto">
                    "Your tone. Your dynamics. Your phrasing. That's always needed a teacher."
                </p>
            </div>
        </section>
    }
}
```

**Step 2: Verify build compiles**

Run: `cd apps/web && cargo check`

**Step 3: Commit**

```bash
git add apps/web/src/pages/landing.rs
git commit -m "simplify problem section to two direct lines from pitch"
```

---

### Task 5: Replace How It Works + What You'll Learn with Feedback Showcase

Remove the old `HowItWorksSection`, `StepCard`, `WhatYouLearnSection`, and `CategoryPreviewCard` components. Replace with a single `FeedbackShowcase` that shows one real feedback example inline.

**Files:**
- Modify: `apps/web/src/pages/landing.rs` (replace ~lines 177-307)

**Step 1: Delete old components and write FeedbackShowcase**

Replace the `HowItWorksSection`, `StepCard`, `WhatYouLearnSection`, and `CategoryPreviewCard` components with:

```rust
// -- Feedback Showcase ------------------------------------------------------

#[component]
fn FeedbackShowcase() -> impl IntoView {
    view! {
        <section class="bg-gradient-warm-deep">
            <div class="container-wide py-16 md:py-24">
                <h2 class="font-display text-display-sm md:text-display-md text-ink-900 text-center mb-12">
                    "Here's what Crescend hears"
                </h2>

                <div class="max-w-2xl mx-auto">
                    <div class="card p-6 md:p-8 bg-paper-50/80 backdrop-blur-sm">
                        <div class="mb-4">
                            <p class="text-label-sm uppercase tracking-wider text-sepia-600 mb-1">
                                "Chopin -- Ballade No. 1 in G minor"
                            </p>
                            <p class="text-body-sm text-ink-500">"Performed by Krystian Zimerman"</p>
                        </div>

                        <div class="space-y-4">
                            <FeedbackPoint
                                label="Sound Quality"
                                text="Your dynamic range in measures 24-31 stays mostly at mezzo-forte where Chopin's marking calls for a gradual crescendo to fortissimo. Try exaggerating the build -- start softer, arrive louder."
                            />
                            <FeedbackPoint
                                label="Technical Control"
                                text="Pedal changes in the lyrical section are clean, but running passages in bars 56-64 accumulate harmonic blur. Try half-pedaling through the chromatic descent."
                            />
                        </div>
                    </div>
                </div>
            </div>
        </section>
    }
}

#[component]
fn FeedbackPoint(label: &'static str, text: &'static str) -> impl IntoView {
    view! {
        <div class="border-l-2 border-sepia-400 pl-4">
            <p class="text-label-sm font-medium text-sepia-700 mb-1">{label}</p>
            <p class="text-body-sm text-ink-600 leading-relaxed">{text}</p>
        </div>
    }
}
```

**Step 2: Write the compact HowItWorksStrip**

Add below the FeedbackShowcase:

```rust
// -- How It Works (compact) -------------------------------------------------

#[component]
fn HowItWorksStrip() -> impl IntoView {
    view! {
        <section id="how-it-works" class="bg-gradient-warm-mid scroll-mt-20">
            <div class="container-wide py-10 md:py-14">
                <div class="flex flex-col sm:flex-row items-center justify-center gap-6 sm:gap-12 text-center">
                    <StripStep
                        icon_path="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4M12 15a3 3 0 003-3V6a3 3 0 00-6 0v6a3 3 0 003 3z"
                        text="Record yourself"
                    />
                    <svg class="hidden sm:block w-5 h-5 text-sepia-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
                    </svg>
                    <StripStep
                        icon_path="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        text="Upload"
                    />
                    <svg class="hidden sm:block w-5 h-5 text-sepia-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
                    </svg>
                    <StripStep
                        icon_path="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                        text="Get feedback"
                    />
                </div>
            </div>
        </section>
    }
}

#[component]
fn StripStep(icon_path: &'static str, text: &'static str) -> impl IntoView {
    view! {
        <div class="flex items-center gap-2">
            <div class="w-8 h-8 rounded-lg bg-sepia-100 flex items-center justify-center flex-shrink-0">
                <svg class="w-4 h-4 text-sepia-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d=icon_path />
                </svg>
            </div>
            <span class="text-body-md font-medium text-ink-700">{text}</span>
        </div>
    }
}
```

**Step 3: Verify build compiles**

Run: `cd apps/web && cargo check`

**Step 4: Commit**

```bash
git add apps/web/src/pages/landing.rs
git commit -m "replace how-it-works and category cards with feedback showcase"
```

---

### Task 6: Rewrite credibility section as mission + proof points

Replace the standalone "Credibility" section (with section label and founder bio) with a combined mission + proof points section.

**Files:**
- Modify: `apps/web/src/pages/landing.rs` (replace CredibilitySection, ~lines 309-348)

**Step 1: Replace CredibilitySection with MissionSection**

```rust
// -- Mission + Credibility --------------------------------------------------

#[component]
fn MissionSection() -> impl IntoView {
    view! {
        <section class="bg-gradient-warm-rich">
            <div class="container-narrow text-center py-16 md:py-24">
                <p class="font-display text-display-sm text-ink-900 mb-8">
                    "Quality feedback shouldn't cost $200 an hour."
                </p>
                <p class="text-body-lg text-ink-600 max-w-2xl mx-auto mb-12">
                    "We built Crescend on published research so that every pianist can practice smarter -- not just those who can afford weekly lessons."
                </p>

                <div class="flex flex-wrap justify-center gap-x-8 gap-y-3 text-body-sm text-ink-500">
                    <span>"55% more accurate than note-based approaches"</span>
                    <span class="hidden sm:inline text-paper-400" aria-hidden="true">"|"</span>
                    <span>"Informed by 30+ educator interviews"</span>
                    <span class="hidden sm:inline text-paper-400" aria-hidden="true">"|"</span>
                    <span>
                        "Published on "
                        <a href="https://arxiv.org/abs/2601.19029" target="_blank" rel="noopener" class="underline underline-offset-2">"arXiv"</a>
                    </span>
                </div>
            </div>
        </section>
    }
}
```

**Step 2: Verify build compiles**

Run: `cd apps/web && cargo check`

**Step 3: Commit**

```bash
git add apps/web/src/pages/landing.rs
git commit -m "replace credibility section with mission + proof points"
```

---

### Task 7: Update final CTA and meta tags

Update the final CTA copy and meta tags to match the new voice.

**Files:**
- Modify: `apps/web/src/pages/landing.rs` (FinalCtaSection, ~lines 350-366)
- Modify: `apps/web/src/app.rs:25-26` (Title and Meta)

**Step 1: Update FinalCtaSection**

```rust
// -- Final CTA --------------------------------------------------------------

#[component]
fn FinalCtaSection() -> impl IntoView {
    view! {
        <section class="bg-gradient-hero">
            <div class="container-narrow text-center py-16 md:py-24">
                <h2 class="font-display text-display-md text-ink-900 mb-6">
                    "Ready to hear what your playing really sounds like?"
                </h2>
                <a href="/analyze" class="btn-primary text-lg px-8 py-4">
                    "Try It Free"
                </a>
            </div>
        </section>
    }
}
```

**Step 2: Update meta tags in app.rs**

In `apps/web/src/app.rs`, change lines 25-26:

```rust
<Title text="Crescend -- A Teacher for Every Pianist"/>
<Meta name="description" content="Record yourself playing piano. Get the feedback a great teacher would give you -- on your tone, your dynamics, your phrasing."/>
```

**Step 3: Verify build compiles**

Run: `cd apps/web && cargo check`

**Step 4: Commit**

```bash
git add apps/web/src/pages/landing.rs apps/web/src/app.rs
git commit -m "update final CTA copy and meta tags to match pitch voice"
```

---

### Task 8: Clean up old website-overhaul-design.md

Delete the old design doc that has been superseded by the new one.

**Files:**
- Delete: `apps/docs/website-overhaul-design.md`

**Step 1: Check if file still exists**

Run: `ls apps/docs/website-overhaul-design.md`

If it exists, delete it. If already deleted, skip this task.

**Step 2: Delete and commit**

```bash
git rm apps/docs/website-overhaul-design.md
git commit -m "remove superseded website overhaul design doc"
```

---

### Task 9: Full build and visual verification

Build the full project and verify everything works.

**Step 1: Full cargo check**

Run: `cd apps/web && cargo check`
Expected: no errors

**Step 2: Tailwind build**

Run the tailwind build to generate CSS (check wrangler.toml or package.json for the exact command -- likely `npx tailwindcss -i tailwind.css -o dist/output.css` or similar).

**Step 3: Visual checklist**

Manually verify:
- [ ] Landing page loads with "A teacher for every pianist" headline
- [ ] Plus Jakarta Sans renders for display text (not Cormorant Garamond)
- [ ] Warm gradient backgrounds visible across all sections (no flat white)
- [ ] No paper texture overlay
- [ ] Social proof strip removed from hero
- [ ] Feedback showcase shows Chopin example with 2 feedback points
- [ ] How-it-works is a compact horizontal strip
- [ ] Mission section shows "$200 an hour" copy with proof points
- [ ] Final CTA says "Try It Free"
- [ ] Meta title says "A Teacher for Every Pianist"
- [ ] No "Credibility" section label anywhere
- [ ] No founder bio
- [ ] `/analyze` page unchanged
- [ ] Header navigation still works
- [ ] Footer unchanged

**Step 4: Commit any fixes**

If any visual issues found, fix and commit.
