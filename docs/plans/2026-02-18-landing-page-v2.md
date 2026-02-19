# Landing Page V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restyle the landing page from generic warm-gradient SaaS to a handcrafted editorial feel: Lora serif headlines, flat cream background (#FDF8F0 + #2D2926), clay/terracotta accents, subtle musical SVG decorations.

**Architecture:** Pure frontend changes. Swap font, replace sepia color palette with clay, flatten all gradient backgrounds to solid cream, add decorative SVG elements (waveform dividers, staff lines, grain texture). No backend, route, or /analyze page logic changes. The sepia->clay rename touches 14 source files.

**Tech Stack:** Rust/Leptos (WASM), Tailwind CSS, Google Fonts

---

### Task 1: Swap display font from Plus Jakarta Sans to Lora

**Files:**
- Modify: `apps/web/src/app.rs:20-22` (Google Fonts `<Link>`)
- Modify: `apps/web/tailwind.config.js:93` (fontFamily.display)
- Modify: `apps/web/tailwind.css:43` (--font-display CSS variable)

**Step 1: Update Google Fonts link in app.rs**

Replace the Google Fonts `<Link>` href (line 21) to swap `Plus+Jakarta+Sans` for `Lora`:

```rust
        <Link
            href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"
            rel="stylesheet"
        />
```

Note: Source Serif 4 is dropped entirely -- Lora covers the serif need.

**Step 2: Update Tailwind fontFamily config**

In `apps/web/tailwind.config.js` line 93, change:

```js
fontFamily: {
    display: ['"Lora"', 'Georgia', 'serif'],
    serif: ['"Lora"', 'Georgia', 'serif'],
    sans: ['"Inter"', 'system-ui', 'sans-serif'],
    mono: ['"JetBrains Mono"', 'Consolas', 'monospace'],
},
```

**Step 3: Update CSS variable**

In `apps/web/tailwind.css` line 43-44, change:

```css
--font-display: "Lora", Georgia, serif;
--font-serif: "Lora", Georgia, serif;
```

**Step 4: Verify build compiles**

Run: `cd apps/web && cargo check`
Expected: compiles without errors (font is purely CSS, Rust code uses `font-display` class unchanged)

**Step 5: Commit**

```bash
git add apps/web/src/app.rs apps/web/tailwind.config.js apps/web/tailwind.css
git commit -m "swap display font from Plus Jakarta Sans to Lora serif"
```

---

### Task 2: Replace sepia palette with clay palette

Rename the `sepia` color scale to `clay` across the entire design system. Update hex values from amber/gold to brown-rose (no yellow). This is a mechanical find-replace across 14 source files plus config.

**Files:**
- Modify: `apps/web/tailwind.config.js` (sepia color scale, backgroundImage, boxShadow)
- Modify: `apps/web/tailwind.css` (all sepia CSS vars and class references)
- Modify: 14 `.rs` files in `apps/web/src/` (class name references)

**Step 1: Update tailwind.config.js color scale**

Replace the `sepia` key and all its values with `clay`:

```js
clay: {
  50:  '#faf7f5',
  100: '#f0ebe7',
  200: '#e0d6cf',
  300: '#c9bab0',
  400: '#a89386',
  500: '#8b7668',  // Primary accent
  600: '#72604f',  // Buttons, links
  700: '#5a4b3e',
  800: '#433830',
  900: '#2d2824',
  950: '#1e1a17',
},
```

Also in the same file, update all `sepia` references:

In `backgroundImage`:
- Change `'gradient-sepia'` key and its value: `'linear-gradient(135deg, #8b7668 0%, #72604f 100%)'`
- Change `'gradient-sepia-subtle'` key and its value: `'linear-gradient(135deg, #e0d6cf 0%, #c9bab0 100%)'`
- Change `'gradient-cta'`: `'linear-gradient(135deg, #72604f 0%, #5a4b3e 100%)'`

In `boxShadow`:
- Change `'sepia'`: `'0 4px 14px -2px rgba(139, 118, 104, 0.20)'`
- Change `'sepia-lg'`: `'0 8px 24px -4px rgba(139, 118, 104, 0.25)'`
- Change `'glow'`: `'0 0 20px rgba(139, 118, 104, 0.15)'`

**Step 2: Update tailwind.css CSS variables and class references**

Replace all `--color-sepia-*` CSS variables with `--color-clay-*` and new hex values:

```css
--color-clay-50: #faf7f5;
--color-clay-100: #f0ebe7;
--color-clay-200: #e0d6cf;
--color-clay-300: #c9bab0;
--color-clay-400: #a89386;
--color-clay-500: #8b7668;
--color-clay-600: #72604f;
--color-clay-700: #5a4b3e;
--color-clay-800: #433830;
--color-clay-900: #2d2824;
```

Also update `--shadow-sepia` to `--shadow-clay` with new rgba values.

Then do a global find-replace in the same file: `sepia-` -> `clay-` for all Tailwind class references (in `@apply` directives, component classes, etc.).

**Step 3: Update all .rs source files**

Global find-replace across all 14 files: `sepia-` -> `clay-` in CSS class strings. The files are:

- `src/app.rs`
- `src/pages/landing.rs`
- `src/pages/demo.rs`
- `src/components/category_card.rs`
- `src/components/header.rs`
- `src/components/loading_spinner.rs`
- `src/components/audio_player.rs`
- `src/components/chat_panel.rs`
- `src/components/chat_message.rs`
- `src/components/chat_input.rs`
- `src/components/expandable_citation.rs`
- `src/components/teacher_feedback.rs`
- `src/components/practice_tips.rs`
- `src/components/radar_chart.rs`

**Step 4: Verify build compiles**

Run: `cd apps/web && cargo check`
Expected: compiles (class names are strings, Rust doesn't validate them)

**Step 5: Verify Tailwind builds**

Run: `cd apps/web && npx tailwindcss -i ./tailwind.css -o ./dist/client/pkg/output.css --minify`
Expected: builds without errors

**Step 6: Commit**

```bash
git add apps/web/tailwind.config.js apps/web/tailwind.css apps/web/src/
git commit -m "rename sepia palette to clay with brown-rose hex values"
```

---

### Task 3: Flatten backgrounds to solid cream

Remove all gradient backgrounds. Use flat `#FDF8F0` everywhere.

**Files:**
- Modify: `apps/web/tailwind.config.js:274-285` (backgroundImage -- remove landing page gradients)
- Modify: `apps/web/tailwind.css:91` (body background)
- Modify: `apps/web/src/app.rs:29` (root div background)
- Modify: `apps/web/src/pages/landing.rs` (remove bg-gradient-* from all sections)

**Step 1: Update paper-50 color to #FDF8F0**

In `apps/web/tailwind.config.js`, change `paper` scale:

```js
paper: {
  50:  '#FDF8F0',  // Main background
  100: '#f7f3ec',  // Slightly deeper
  200: '#ede8e0',  // Light warm
  300: '#e2dbd3',  // Borders
  400: '#d4cbc0',  // Aged paper
  500: '#c4b9ab',  // Darker
},
```

**Step 2: Remove landing page gradient utilities from backgroundImage**

In `apps/web/tailwind.config.js`, remove these keys from `backgroundImage`:
- `'gradient-hero'`
- `'gradient-warm-mid'`
- `'gradient-warm-deep'`
- `'gradient-warm-rich'`
- `'gradient-page'`
- `'gradient-paper'`
- `'gradient-warm'`

Keep: `'gradient-sepia'` (renamed to `'gradient-clay'` in task 2), `'gradient-sepia-subtle'` (renamed), `'gradient-cta'` (renamed), `'noise'`.

**Step 3: Update root div in app.rs**

Change line 29 from:

```rust
<div class="min-h-screen bg-gradient-hero flex flex-col">
```

to:

```rust
<div class="min-h-screen bg-paper-50 flex flex-col">
```

**Step 4: Remove gradient classes from landing.rs sections**

Replace all `bg-gradient-*` classes with no background (sections inherit from the page background):

- Line 23: `"relative overflow-hidden bg-gradient-hero"` -> `"relative overflow-hidden"`
- Line 152: `"bg-gradient-warm-mid"` -> remove the class, keep the tag
- Line 170: `"bg-gradient-warm-deep"` -> remove the class
- Line 217: `"bg-gradient-warm-mid scroll-mt-20"` -> `"scroll-mt-20"`
- Line 263: `"bg-gradient-warm-rich"` -> remove the class
- Line 292: `"bg-gradient-hero"` -> remove the class

The sections become transparent, showing the page's `bg-paper-50` (#FDF8F0) through.

**Step 5: Update header background**

In `apps/web/src/components/header.rs` line 6, the header uses `bg-paper-50/95`. This now resolves to `#FDF8F0` at 95% opacity, which is correct. No change needed.

**Step 6: Verify build compiles and Tailwind builds**

Run: `cd apps/web && cargo check && npx tailwindcss -i ./tailwind.css -o ./dist/client/pkg/output.css --minify`

**Step 7: Commit**

```bash
git add apps/web/tailwind.config.js apps/web/tailwind.css apps/web/src/app.rs apps/web/src/pages/landing.rs
git commit -m "flatten all backgrounds to solid cream #FDF8F0, remove gradients"
```

---

### Task 4: Update button style from gradient to solid clay

The `.btn-primary` currently uses `bg-gradient-cta`. Change to solid clay background.

**Files:**
- Modify: `apps/web/tailwind.css:152-162` (btn-primary class)

**Step 1: Update btn-primary**

In `apps/web/tailwind.css`, change `.btn-primary`:

```css
.btn-primary {
    @apply inline-flex items-center justify-center gap-2
           px-6 py-3
           bg-clay-700 text-paper-50 font-medium
           rounded-md shadow-button
           transition-all duration-300
           hover:bg-clay-800 hover:shadow-button-hover
           active:bg-clay-900 active:shadow-button-active active:scale-[0.98]
           focus-visible:ring-2 focus-visible:ring-clay-500 focus-visible:ring-offset-2;
    transition-timing-function: cubic-bezier(0.16, 1, 0.3, 1);
}
```

Note: solid `bg-clay-700` (#5a4b3e) replaces `bg-gradient-cta`. Hover goes to `clay-800`, active to `clay-900`.

**Step 2: Verify Tailwind builds**

Run: `cd apps/web && npx tailwindcss -i ./tailwind.css -o ./dist/client/pkg/output.css --minify`

**Step 3: Commit**

```bash
git add apps/web/tailwind.css
git commit -m "change CTA button from gradient to solid clay background"
```

---

### Task 5: Add decorative SVG elements

Add three subtle musical decorations: waveform section dividers, faint staff lines in hero, and a light grain texture overlay.

**Files:**
- Modify: `apps/web/tailwind.css` (add grain texture class, waveform divider styles)
- Modify: `apps/web/src/pages/landing.rs` (add SVG dividers between sections, staff lines in hero)
- Modify: `apps/web/src/app.rs:29` (add grain texture class to root div)

**Step 1: Add grain texture and waveform divider CSS to tailwind.css**

Add inside `@layer components`, before the closing `}`:

```css
/* Grain texture overlay - subtle analog warmth */
.texture-grain {
    position: relative;
}

.texture-grain::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
    opacity: 0.02;
    pointer-events: none;
    mix-blend-mode: multiply;
    z-index: 1;
}

/* Waveform section divider */
.wave-divider {
    width: 100%;
    height: 24px;
    overflow: hidden;
}

.wave-divider svg {
    width: 100%;
    height: 100%;
}
```

**Step 2: Add grain texture to root div in app.rs**

Change line 29 (already updated in Task 3) to:

```rust
<div class="min-h-screen bg-paper-50 flex flex-col texture-grain">
```

**Step 3: Add WaveDivider component and staff lines to landing.rs**

Add a `WaveDivider` component after the imports:

```rust
#[component]
fn WaveDivider() -> impl IntoView {
    view! {
        <div class="wave-divider" aria-hidden="true">
            <svg viewBox="0 0 1440 24" preserveAspectRatio="none" fill="none">
                <path
                    d="M0 12 C240 4, 480 20, 720 12 S1200 4, 1440 12"
                    stroke="#c9bab0"
                    stroke-width="1"
                    fill="none"
                />
            </svg>
        </div>
    }
}
```

Add a `StaffLines` component for the hero:

```rust
#[component]
fn StaffLines() -> impl IntoView {
    view! {
        <div class="absolute inset-0 flex flex-col justify-center pointer-events-none" aria-hidden="true"
             style="opacity: 0.035">
            <div class="space-y-3 max-w-4xl mx-auto w-full px-8">
                <div class="h-px bg-clay-400"></div>
                <div class="h-px bg-clay-400"></div>
                <div class="h-px bg-clay-400"></div>
                <div class="h-px bg-clay-400"></div>
                <div class="h-px bg-clay-400"></div>
            </div>
        </div>
    }
}
```

**Step 4: Wire decorative components into LandingPage**

Update `LandingPage` to include dividers:

```rust
#[component]
pub fn LandingPage() -> impl IntoView {
    view! {
        <div>
            <HeroSection />
            <WaveDivider />
            <ProblemSection />
            <WaveDivider />
            <FeedbackShowcase />
            <WaveDivider />
            <HowItWorksStrip />
            <WaveDivider />
            <MissionSection />
            <WaveDivider />
            <FinalCtaSection />
        </div>
    }
}
```

Update `HeroSection` to include staff lines:

```rust
#[component]
fn HeroSection() -> impl IntoView {
    view! {
        <section class="relative overflow-hidden">
            <StaffLines />
            <div class="container-wide py-16 md:py-24 lg:py-32 relative z-10">
```

(Add `relative z-10` to the container div so content sits above the staff lines.)

**Step 5: Verify build compiles and Tailwind builds**

Run: `cd apps/web && cargo check && npx tailwindcss -i ./tailwind.css -o ./dist/client/pkg/output.css --minify`

**Step 6: Commit**

```bash
git add apps/web/tailwind.css apps/web/src/pages/landing.rs apps/web/src/app.rs
git commit -m "add decorative elements: grain texture, waveform dividers, staff lines"
```

---

### Task 6: Clean up unused config

Remove leftover gradient definitions and unused color scales.

**Files:**
- Modify: `apps/web/tailwind.config.js` (remove highlight colors if unused, clean backgroundImage)
- Modify: `apps/web/tailwind.css` (remove --color-highlight vars, --shadow-sepia now --shadow-clay)

**Step 1: Remove highlight color scale**

In `apps/web/tailwind.config.js`, check if `highlight` is used outside the design system. If only used in `.metric-highlight` and `badge-highlight` (which are for the /analyze page), keep it. Otherwise remove.

**Step 2: Remove unused backgroundImage entries**

Confirm all gradient-hero, gradient-warm-*, gradient-page, gradient-paper, gradient-warm entries are removed (done in Task 3). Keep only:
- `gradient-clay` (renamed from gradient-sepia)
- `gradient-clay-subtle` (renamed from gradient-sepia-subtle)
- `gradient-cta`
- `noise`

**Step 3: Verify build compiles and Tailwind builds**

Run: `cd apps/web && cargo check && npx tailwindcss -i ./tailwind.css -o ./dist/client/pkg/output.css --minify`

**Step 4: Commit**

```bash
git add apps/web/tailwind.config.js apps/web/tailwind.css
git commit -m "clean up unused gradient and color definitions"
```

---

### Task 7: Full build and visual verification

**Step 1: Full cargo check**

Run: `cd apps/web && cargo check`
Expected: no errors

**Step 2: Tailwind build**

Run: `cd apps/web && npx tailwindcss -i ./tailwind.css -o ./dist/client/pkg/output.css --minify`
Expected: builds without errors

**Step 3: Visual checklist**

- [ ] Landing page has flat cream (#FDF8F0) background everywhere, no gradients
- [ ] Lora serif renders for display text / headlines
- [ ] Clay/terracotta accents on buttons, links, icons (no gold/amber)
- [ ] Waveform SVG dividers visible between sections
- [ ] Faint staff lines behind hero text (barely visible, ~3.5% opacity)
- [ ] Subtle grain texture across the page
- [ ] CTA buttons are solid clay, not gradient
- [ ] "Try It Free" buttons work and link to /analyze
- [ ] Header renders correctly with clay accent colors
- [ ] Footer renders correctly with clay accent colors
- [ ] /analyze page unchanged in functionality
- [ ] No "sepia" class names remaining anywhere in source
