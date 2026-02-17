# Website Overhaul Design: Product-First CrescendAI

**Date:** 2026-02-17

## Context

The current crescend.ai landing page is an academic research paper walkthrough (motivation, approach, key findings, validation, per-dimension results). The demo page shows 19 PercePiano dimensions that are being replaced by a teacher-grounded taxonomy of 5-8 dimensions. The investor meeting prep materials (deck, one-pager, demo runbook) cover the near-term meeting, but the website needs to become a coherent product -- not a research showcase.

This design doc redefines crescend.ai as a product-first consumer website targeting serious piano students (Persona 1: dedicated self-learner, age 25-45, 1-3 hours daily practice, willing to pay $15-30/month).

---

## Design Principles (Research-Informed)

### 1. Never lead with "AI"

Multiple studies confirm that "AI" labeling lowers consumer purchase intent (WSU 2024, Lippincott n=11,000+, Dell CES 2026 walkback). Every successful music ed product (Yousician, Flowkey, SmartMusic, Tonestro, Fender Play) describes outcomes, not mechanisms.

**Rule:** The word "AI" should not appear in headlines, CTAs, or primary navigation. It can appear in secondary credibility contexts ("backed by published research") and in technical documentation. The brand name "CrescendAI" stays in the logo but consumer-facing copy uses "Crescend" where possible.

**Language to use:** "Listens to your playing," "detailed feedback," "practice companion," "built by musicians," "the feedback between lessons"

**Language to avoid:** "AI-powered," "neural network," "machine learning," "automated feedback," "algorithm"

### 2. Show the product, not descriptions of it

Abstract illustrations and research diagrams are out. The 2026 standard is showing the actual product interface. Animated product demos in the hero section increase conversions 20-30% and visitor time-on-page 5x.

### 3. One CTA, value-driven

Single-CTA pages convert 12% better than multi-CTA. CTA copy should emphasize value ("Get your feedback") not action ("Sign up"). CTAs above the fold outperform below-fold by 304%.

### 4. Warm precision, not saccharine praise

Feedback tone: specific, constructive, treats the musician as an intelligent adult. Uses musical vocabulary (piano, forte, crescendo, legato). Not cold metrics ("Score: 4.2/10"), not empty praise ("Great job!"), but precise observation ("Your dynamic range in measures 24-31 stays mostly at mezzo-forte where Chopin's marking calls for a gradual crescendo to fortissimo").

### 5. Premium aesthetic through restraint

Keep the serif-forward typography (Cormorant Garamond display, Source Serif 4 body), warm sepia palette, and generous whitespace. This differentiates from every competitor using geometric sans-serifs and signals "serious, cultivated, for adults." Think: "well-appointed music conservatory," not "app store game."

---

## Site Structure

| Route | Purpose | Status |
|-------|---------|--------|
| `/` | Product landing page | **New** (replaces research paper) |
| `/analyze` | Upload + analysis experience | **Rewrite** (replaces `/demo`) |
| `/analyze/:id` | Deep link to analysis result | **Keep** (update route) |
| `/demo` | Redirect to `/analyze` | **New** (backwards compat) |

Research page: **Removed.** Paper link goes directly to arXiv.

### Navigation (Header)

- Logo: "Crescend" (left)
- "How It Works" (anchor scroll on landing page, or link to `/#how-it-works`)
- "Analyze" (goes to `/analyze`)
- "Paper" (external link to arXiv)

### Footer

- Logo, "Paper" link, email contact, copyright
- Privacy note: "Your recordings are yours. We don't store or train on your data."

---

## Landing Page (`/`)

### Section 1: Hero

**Headline:** Short, outcome-focused, passes the caveman test (<8 words).

Candidates (to be tested/chosen during implementation):

- "Hear what your playing really sounds like"
- "Know exactly what to practice next"
- "Concert-level feedback in seconds"
- "The feedback between lessons"

**Subline:** One sentence explaining how. Example: "Upload a recording. Get detailed, personalized feedback on your sound, musical shaping, technique, and interpretation."

**Primary CTA:** "Analyze Your Playing" (links to `/analyze`)

**Visual:** Animated product preview showing the analysis experience -- a waveform being analyzed, category feedback cards appearing, feedback text arriving. This can be a looping CSS/JS animation or a short auto-playing muted video. Not a static screenshot, not an abstract illustration.

**Social proof strip** directly in the hero (not below): "Backed by published research" | "55% more accurate than note-based approaches" | "Built by a Berklee-trained musician"

### Section 2: Problem (PAS Framework)

**Problem:** "You practice for hours. But without a teacher's ear, you don't know what to fix."

**Agitate:** "Is it your pedaling? Your dynamics? Your phrasing? Existing apps check note accuracy -- but that's not what separates good playing from great playing."

**Solution:** "Crescend listens to the things that matter. Not just the right notes -- but how they sound."

Keep this section short. 3 sentences max. The copy does the work, no complex visuals needed.

### Section 3: How It Works

3-step visual flow:

1. **Record** -- "Play your piece and record with any device"
2. **Upload** -- "Upload your recording in seconds"
3. **Get Feedback** -- "Receive detailed feedback across four dimensions of your playing"

Each step gets a simple icon and one-line description. Clean, scannable.

### Section 4: What You'll Learn

Preview the 4 provisional feedback categories as cards:

- **Sound Quality** -- "How does your playing sound? Dynamics, tone, projection."
- **Musical Shaping** -- "How do you shape the music? Phrasing, timing, flow."
- **Technical Control** -- "How clean is your technique? Pedaling, articulation, clarity."
- **Interpretive Choices** -- "What story are you telling? Musical decisions, character, expression."

Each card includes a brief sample feedback snippet showing the tone and specificity of the actual product output. This shows the product rather than describing it.

### Section 5: Credibility

"Built on published research"

3 proof points in a compact layout:

- "55% more accurate than note-based approaches"
- "Informed by 30+ educator interviews"
- "Published on arXiv, submitted to ISMIR 2026"

Link to paper for those who want to dig in.

Founder credibility signal: "Built by Jai Dhiman -- Berklee-trained musician, pianist since age 8, and the ML engineer behind the research."

### Section 6: Final CTA

"Ready to hear what your playing really sounds like?"

"Analyze Your Playing" button. Clean, high-contrast, generous whitespace.

---

## Analyze Page (`/analyze`)

### Design Philosophy

Designed from Persona 1's perspective: a dedicated self-learner who records themselves practicing, uploads, and wants specific feedback on what to improve. Upload is the primary action. Example analyses are secondary (for visitors without a recording handy).

### Layout

**Top: Upload Zone (Primary)**

- Prominent drag-and-drop area
- "Upload your recording"
- Supported formats: MP3, WAV, M4A, WebM (max 50MB)
- After upload: file name, duration, waveform preview

**Below: Example Analyses (Secondary)**

- "Don't have a recording? Try one of these:"
- 2-3 compact cards (performer, piece, duration)
- Visually de-emphasized compared to upload
- Clicking loads the analysis directly

**Analysis Flow**

When a recording is selected/uploaded:

1. **Audio Player** -- waveform visualization, play/pause, current time
2. **"Get Feedback" button** -- single, prominent CTA
3. **Processing State** -- multi-step progress with meaningful messages:
   - "Listening to your recording..." (with waveform visualization)
   - "Evaluating your sound quality..."
   - "Analyzing musical shaping..."
   - "Preparing your feedback..."
   This transforms wait time into trust-building.
4. **Results Display** -- the core product experience (see below)

### Results Display: 4-Category Feedback

Replace the 19-dimension radar chart with a category-based feedback display.

**Structure:**

4 category cards, stacked vertically, each containing:

- **Category name + icon** (Sound Quality, Musical Shaping, Technical Control, Interpretive Choices)
- **Summary indicator** -- not a numeric score, but a qualitative signal. Consider: a subtle gradient bar, or simply a one-sentence summary ("Your dynamics are a strength; pedal clarity could use attention")
- **Detailed feedback** -- 2-4 sentences of specific, constructive feedback using musical vocabulary. References specific moments in the recording where possible.
- **Practice suggestion** -- one actionable thing to try in the next practice session

**Why not a radar chart?**

- 4 provisional categories don't make a compelling radar
- The 19-dimension radar was tied to PercePiano dimensions being abandoned
- Category cards with natural language feedback better match the "teacher" metaphor
- Cards are more mobile-friendly and scannable
- When the final taxonomy lands (5-8 empirical dimensions), cards adapt easily

**Technical mapping (interim):**

Each provisional category aggregates PercePiano dimensions using composite scores weighted by MLP probing R-squared (from the taxonomy design doc):

```
Sound Quality     = weighted_avg(dynamic_range, timbre_depth, timbre_variety, timbre_loudness, timbre_brightness)
Musical Shaping   = weighted_avg(timing, tempo, space, drama)
Technical Control = weighted_avg(pedal_amount, pedal_clarity, articulation_length, articulation_touch)
Interpretive      = weighted_avg(mood_valence, mood_energy, mood_imagination, sophistication, interpretation)
```

Weights proportional to max(0, MLP_R2) from the data audit, normalized per category.

### Chat Panel

Keep the existing chat Q&A capability but frame it differently:

- "Have a question about your feedback?"
- Text input, not a chat history view
- Responses appear inline, conversational
- Maintains the "companion" framing, not a chatbot

---

## Technical Implementation

### Files to Modify

**Complete rewrites:**

- `apps/web/src/pages/landing.rs` -- currently 3,600+ lines of research paper. Replace entirely with product landing page.
- `apps/web/src/pages/demo.rs` -- rewrite as analyze page with upload-first flow and 4-category results.

**Significant changes:**

- `apps/web/src/app.rs` -- update routes (`/demo` -> `/analyze`, add redirect)
- `apps/web/src/components/header.rs` -- update navigation links
- `apps/web/src/components/radar_chart.rs` -- either replace with category cards component or adapt to 4-point display
- `apps/web/src/components/teacher_feedback.rs` -- restructure around 4 categories
- `apps/web/src/components/practice_tips.rs` -- organize by category
- `apps/web/src/components/loading_spinner.rs` -- enhance to multi-step progress indicator

**New components:**

- `apps/web/src/components/category_card.rs` -- feedback card for each of the 4 categories
- `apps/web/src/components/hero_animation.rs` -- animated product preview for landing page (if CSS/JS animation approach)

**New logic:**

- Category aggregation: map 19 PercePiano dimensions -> 4 composite scores (can live in `apps/web/src/models/analysis.rs` or a new `apps/web/src/services/categories.rs`)

**Existing components to keep as-is:**

- `apps/web/src/components/audio_upload.rs` -- upload functionality works
- `apps/web/src/components/audio_player.rs` -- playback works
- `apps/web/src/components/chat_panel.rs` -- keep chat capability
- `apps/web/src/components/chat_input.rs` -- keep
- `apps/web/src/components/chat_message.rs` -- keep
- `apps/web/src/components/expandable_citation.rs` -- keep for feedback citations

**Styling:**

- `apps/web/tailwind.css` -- keep the design system. Minor additions for new components (category card styles). No palette or typography changes.
- `apps/web/tailwind.config.js` -- no changes needed.
- Remove `/public/figures/` directory (10 research diagram PNGs no longer needed on the site)

### Existing Utilities to Reuse

- `apps/web/src/api/handlers.rs` -- performance list/get endpoints stay
- `apps/web/src/services/huggingface.rs` -- inference pipeline stays
- `apps/web/src/services/rag.rs` -- RAG feedback pipeline stays
- `apps/web/src/services/feedback.rs` -- feedback generation stays (may need category-aware prompting)
- `apps/web/src/models/analysis.rs` -- PerformanceDimensions struct stays, add category composite calculation
- `apps/web/src/models/performance.rs` -- Performance types stay
- `apps/web/src/shell.rs` -- HTML shell stays (update meta tags/description)

---

## Content Changes

### Meta/SEO

**Title:** "Crescend -- Detailed Piano Feedback in Seconds"
**Description:** "Upload a piano recording and get detailed, personalized feedback on your sound quality, musical shaping, technique, and interpretation."

### Copy Tone

- Precise, confident, understated (premium tool language)
- Musical vocabulary where natural (dynamics, phrasing, pedaling)
- No exclamation marks
- No "AI-powered", "revolutionary", "groundbreaking"
- First person from founder where it adds credibility: "Built by a Berklee-trained musician"

### Privacy Statement

Footer or subtle note on analyze page: "Your recordings are yours. We don't store or train on your data." (This directly addresses the #1 privacy fear identified in the research.)

---

## What This Design Does NOT Cover

- User accounts / authentication (future)
- Progress tracking across sessions (future)
- Pricing page (future -- will follow prestige pricing: round numbers, $20 or $25/month)
- Mobile app (future)
- New model taxonomy integration (the 4 provisional categories will be swapped for empirical dimensions once the taxonomy work from `docs/plans/2026-02-17-teacher-grounded-taxonomy-design.md` is complete)
- Investor materials (handled separately in `investor/` directory)

---

## Verification

After implementation:

1. Landing page loads at crescend.ai with product messaging, not research paper
2. No "AI" in headlines or CTAs
3. Single primary CTA on landing page points to `/analyze`
4. `/analyze` page allows upload and shows 4-category feedback
5. `/demo` redirects to `/analyze`
6. Research figures removed from public assets
7. Header navigation updated (How It Works, Analyze, Paper)
8. Meta tags updated for SEO
9. Mobile responsive
10. Build succeeds
11. Deploy succeeds
