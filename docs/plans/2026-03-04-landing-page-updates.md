# Landing Page Updates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update the landing page with a staircase image layout, laptop+phone mockup section, and animation-ready feature cards.

**Architecture:** All changes are in `apps/web/`. The landing page is a single-file route (`src/routes/index.tsx`) with four section components. We add one new section (DeviceMockupSection), rework two existing sections (CascadingQuoteSection, FeatureCardsSection), and create two standalone mockup HTML files for screenshotting.

**Tech Stack:** React, Tailwind CSS v4, HTML/CSS (mockups)

---

### Task 1: Rework CascadingQuoteSection to Staircase Layout

**Files:**
- Modify: `apps/web/src/routes/index.tsx:104-141`

**Step 1: Replace the overlapping absolute-positioned images with a staircase grid**

The current layout uses absolute positioning with percentage offsets creating overlap. Replace with a flex/grid layout where each image is offset but touches at the corner -- no overlap.

Replace the `CascadingQuoteSection` function with:

```tsx
function CascadingQuoteSection() {
  return (
    <section className="py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-6 lg:px-12">
        <div className="grid grid-cols-1 lg:grid-cols-[5fr_6fr] gap-12 lg:gap-16 items-center">
          {/* Staircase photos */}
          <div className="flex flex-col">
            <div className="w-[55%] self-start">
              <img
                src="/Image2.jpg"
                alt="Practicing alone -- the struggle of hearing your own mistakes"
                className="w-full object-cover"
                style={{ aspectRatio: '4/5' }}
              />
            </div>
            <div className="w-[55%] self-center">
              <img
                src="/Image3.jpg"
                alt="A moment of guidance -- focused attention on the score"
                className="w-full object-cover"
                style={{ aspectRatio: '4/5' }}
              />
            </div>
            <div className="w-[55%] self-end">
              <img
                src="/Image4.jpg"
                alt="The breakthrough -- playing with confidence"
                className="w-full object-cover"
                style={{ aspectRatio: '4/5' }}
              />
            </div>
          </div>

          {/* Pull quote */}
          <div>
            <blockquote className="font-display italic text-display-md lg:text-display-lg text-cream leading-snug">
              "What's the one thing that sounds off that I can't hear myself?"
            </blockquote>
          </div>
        </div>
      </div>
    </section>
  )
}
```

Key change: Images stack vertically in a flex column. Each image is 55% width. `self-start`, `self-center`, `self-end` create the staircase offset. No absolute positioning, no overlap -- they touch at the edge where one ends and the next begins.

**Step 2: Verify in browser**

Run: `cd apps/web && bun dev`

Open `http://localhost:5173` and scroll to the cascading section. Confirm:
- Three images form a staircase stepping down-right
- Images touch at borders, no overlap
- Pull quote is positioned to the right on desktop
- Mobile stacks vertically

**Step 3: Commit**

```bash
git add apps/web/src/routes/index.tsx
git commit -m "feat(web): rework cascading images to staircase layout"
```

---

### Task 2: Build Desktop Chat Mockup HTML

**Files:**
- Create: `apps/web/mockups/desktop-chat.html`

**Step 1: Create the mockups directory**

```bash
mkdir -p apps/web/mockups
```

**Step 2: Write the desktop chat mockup**

Create `apps/web/mockups/desktop-chat.html` -- a self-contained HTML file that renders a desktop-width chat UI mockup. This is for screenshotting, not for embedding in the landing page.

Requirements:
- Width: ~1200px viewport appearance
- Shows the CrescendAI web app layout: collapsed sidebar (icons only) on the left, chat area in the center
- Chat conversation about a Bach Invention (e.g., Invention No. 8 in F major):
  - User message: "I just played through Invention No. 8 -- the F major one. Something feels off in the middle section but I can't tell what."
  - Teacher response: A warm, specific observation about the left hand voice in bars 15-20 losing independence, with a score highlight on-demand component shown inline
- Score highlight component: A card showing annotated bars 15-20 with colored highlights on the left hand passages where voices converge
- Use the espresso/cream color palette: background #2D2926, surfaces #3A3633/#454140, text #FDF8F0/#A8A29E, border #504B48
- Fonts: Lora for display, system sans-serif for body (no Google Fonts import needed for mockup -- use fallbacks)
- Sidebar icons: simple SVG circles/rectangles as placeholders (no icon library)
- Score highlight card: stylized representation using CSS (staff lines, colored rectangles for highlighted bars, annotation text). Does not need to be real notation.

**Step 3: Open in browser and verify**

```bash
open apps/web/mockups/desktop-chat.html
```

Confirm it looks like a realistic desktop chat app with an inline score highlight card.

**Step 4: Commit**

```bash
git add apps/web/mockups/desktop-chat.html
git commit -m "feat(web): add desktop chat mockup with score highlight for landing page"
```

---

### Task 3: Build Mobile Chat Mockup HTML

**Files:**
- Create: `apps/web/mockups/mobile-chat.html`

**Step 1: Write the mobile chat mockup**

Create `apps/web/mockups/mobile-chat.html` -- same concept as desktop but at mobile width (~390px).

Requirements:
- Width: ~390px viewport appearance (iPhone frame)
- No sidebar visible (mobile layout)
- Same conversation thread as desktop, but the inline on-demand component is an exercise set (not score highlight)
- Teacher response ends with: "Here are some exercises to help the left hand find its voice again:"
- Exercise set component: 2-3 practice variation cards inline in the chat:
  1. "Slow LH alone" -- Play the left hand part of bars 15-20 at half tempo, focusing on even tone
  2. "Alternating voices" -- Play bars 15-20 hands together, but accent the left hand while keeping the right hand pianissimo
  3. "Rhythmic variation" -- Play the left hand part with dotted rhythms to strengthen finger independence
- Each exercise card: surface background, title in cream, description in secondary text, a subtle play/start icon
- Same color palette and font approach as desktop mockup

**Step 2: Open in browser and verify**

```bash
open apps/web/mockups/mobile-chat.html
```

Confirm it looks like a realistic mobile chat with exercise cards inline.

**Step 3: Commit**

```bash
git add apps/web/mockups/mobile-chat.html
git commit -m "feat(web): add mobile chat mockup with exercise set for landing page"
```

---

### Task 4: Add DeviceMockupSection to Landing Page

**Files:**
- Modify: `apps/web/src/routes/index.tsx:5-13` (add section to page layout)
- Modify: `apps/web/src/routes/index.tsx` (add new component at end of file)

**Step 1: Add the DeviceMockupSection component**

Add this function at the end of `index.tsx`, before the closing of the file:

```tsx
function DeviceMockupSection() {
  return (
    <section className="py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-6 lg:px-12">
        {/* Device frames side by side */}
        <div className="flex items-end justify-center gap-8 lg:gap-12">
          {/* Laptop frame */}
          <div className="w-full max-w-3xl">
            <div className="bg-surface-2 rounded-t-xl p-2">
              {/* Browser chrome dots */}
              <div className="flex gap-1.5 px-2 py-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-border" />
                <div className="w-2.5 h-2.5 rounded-full bg-border" />
                <div className="w-2.5 h-2.5 rounded-full bg-border" />
              </div>
              {/* Screenshot placeholder */}
              <div className="aspect-[16/10] bg-surface rounded-sm overflow-hidden">
                {/* Replace with: <img src="/mockup-desktop.png" alt="CrescendAI desktop chat" className="w-full h-full object-cover" /> */}
                <div className="w-full h-full flex items-center justify-center text-text-tertiary text-body-sm">
                  Desktop mockup screenshot
                </div>
              </div>
            </div>
            {/* Laptop base */}
            <div className="h-3 bg-surface-2 rounded-b-sm mx-[-2%]" />
          </div>

          {/* Phone frame */}
          <div className="w-[140px] lg:w-[180px] shrink-0">
            <div className="bg-surface-2 rounded-2xl p-1.5">
              {/* Notch */}
              <div className="w-16 h-4 bg-surface-2 rounded-full mx-auto mb-1" />
              {/* Screenshot placeholder */}
              <div className="aspect-[9/19.5] bg-surface rounded-xl overflow-hidden">
                {/* Replace with: <img src="/mockup-mobile.png" alt="CrescendAI mobile chat" className="w-full h-full object-cover" /> */}
                <div className="w-full h-full flex items-center justify-center text-text-tertiary text-body-xs text-center px-2">
                  Mobile mockup screenshot
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
```

**Step 2: Add the section to the page layout**

In the `LandingPage` function (lines 5-13), insert `<DeviceMockupSection />` between `<CascadingQuoteSection />` and `<FinalCtaSection />`:

```tsx
function LandingPage() {
  return (
    <div>
      <HeroSection />
      <FeatureCardsSection />
      <CascadingQuoteSection />
      <DeviceMockupSection />
      <FinalCtaSection />
    </div>
  )
}
```

**Step 3: Verify in browser**

Run: `cd apps/web && bun dev`

Confirm:
- New section appears between cascading images and final CTA
- Laptop and phone frames are visible with placeholder text
- Responsive: laptop frame scales, phone stays proportional
- Laptop and phone are bottom-aligned (items-end)

**Step 4: Commit**

```bash
git add apps/web/src/routes/index.tsx
git commit -m "feat(web): add device mockup section with laptop and phone frames"
```

---

### Task 5: Update Feature Cards for Animation Embeds

**Files:**
- Modify: `apps/web/src/routes/index.tsx:56-101`

**Step 1: Update the cards data and placeholder to support video/animation**

Replace the `FeatureCardsSection` function. Changes:
- Add an `id` field to each card for future animation targeting
- Change the placeholder from a plain div to a container ready for video/Lottie embed
- Keep current titles/descriptions for now (will be updated after Jitter animations are designed)

```tsx
function FeatureCardsSection() {
  const cards = [
    {
      id: 'listen',
      title: 'Your teacher is listening',
      description:
        'Your phone listens while you play. When you pause and ask, your teacher is ready with the one thing that matters most.',
    },
    {
      id: 'annotate',
      title: 'Exercises built for you',
      description:
        'Not generic drills. Targeted practice for the specific passage and skill your teacher identified.',
    },
    {
      id: 'exercises',
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
              key={card.id}
              className="bg-surface border border-border rounded-xl overflow-hidden"
            >
              {/* Animation area -- replace with <video> or Lottie player */}
              <div className="aspect-[4/3] bg-surface-2 flex items-center justify-center">
                <span className="text-text-tertiary text-body-xs">
                  Animation placeholder
                </span>
              </div>

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

**Step 2: Verify in browser**

Confirm cards still render correctly with "Animation placeholder" text in the visual area.

**Step 3: Commit**

```bash
git add apps/web/src/routes/index.tsx
git commit -m "feat(web): prepare feature cards for animation embeds"
```

---

### Task 6: Stock Photo Research and Notes

This is a manual/research task, not a code task.

**Step 1: Document image requirements**

The three stock photos need to tell: Struggle, Guidance, Breakthrough.

Suggested search terms and visual direction:

1. **Struggle:** "pianist frustrated" / "hands resting on piano keys" / "person pausing at piano." Look for: someone stopped mid-practice, head down or hands lifted off keys, sense of solitude. Dark/moody lighting preferred.

2. **Guidance:** "studying sheet music closely" / "annotated piano score" / "musician analyzing music." Look for: close-up of someone examining a score with intention, pencil marks on sheet music, or a teacher's hand pointing at a passage. Warm focused light.

3. **Breakthrough:** "pianist playing confidently" / "joyful piano performance" / "hands flowing over piano keys." Look for: motion, energy, confidence. Warm light, sense of flow. Avoid concert/performance shots -- this should feel like a private moment of mastery.

Image specs: High resolution (at least 1200px wide), portrait orientation preferred (4:5 aspect ratio to match layout), warm tones that work with espresso palette.

Sources to check: Unsplash, Pexels, Stocksy (paid, higher quality).

**Step 2: Once photos are selected, replace in `apps/web/public/`**

Replace Image2.jpg, Image3.jpg, Image4.jpg with the new stock photos. Keep the same filenames so no code changes are needed.

---

## Summary

| Task | What | Dependencies |
|------|------|-------------|
| 1 | Staircase layout for cascading images | None |
| 2 | Desktop chat mockup HTML (score highlight) | None |
| 3 | Mobile chat mockup HTML (exercise set) | None |
| 4 | DeviceMockupSection on landing page | None |
| 5 | Feature cards ready for animation embeds | None |
| 6 | Stock photo research (manual) | None |

Tasks 1-5 are all independent and can be done in parallel. Task 6 is manual research.

After this plan: Jitter animation design happens externally. Feature card copy will be updated once animations exist. Mockup screenshots will be taken from the HTML files and placed in `apps/web/public/`.
