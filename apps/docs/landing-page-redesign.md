# Landing Page Redesign: Conversational Journey

**Date:** 2026-02-18

## Context

The current landing page was the first product-first pass, replacing the original research paper walkthrough. It works structurally but reads like a SaaS template -- corporate section labels, clinical typography, too much whitespace, stiff copy that doesn't sound like a musician wrote it. The investor meeting prep (2026-02-16) established a stronger voice: direct, vivid, mission-driven. The website needs to match that voice and feel alive.

**Goal:** Make the landing page feel like a music-forward, warm, inviting experience -- not a tech product page. Match the investor pitch framing ("A teacher for every pianist"). Make it visually rich (warm gradients, musical texture) instead of empty white sections.

## Design Decisions

### Voice and Copy

Match the investor pitch tone throughout. Direct, vivid, accessible. No marketing-speak.

- **Headline:** "A teacher for every pianist." (replaces "The feedback between lessons")
- **Subline:** "Record yourself playing. Get the feedback a great teacher would give you -- on your tone, your dynamics, your phrasing."
- **Problem copy:** "Any app can tell you if you played the right notes. But that's not what separates good playing from great playing. Your tone. Your dynamics. Your phrasing. That's always needed a teacher."
- **Mission:** "Quality feedback shouldn't cost $200 an hour." -- accessibility framing, not founder story
- **CTA:** "Try It Free" (replaces "Analyze Your Playing")

### Visual Direction

- **Gradient backgrounds:** Replace flat white/paper-50 with warm amber-to-honey gradients across sections. Each section shifts slightly in tone.
- **Waveform texture:** Subtle SVG waveform curves as low-opacity decorative elements in gradients.
- **Remove:** Paper texture overlay (`texture-paper`), flat white sections, clinical spacing.

### Typography

- **Drop:** Cormorant Garamond display font (too stuffy/conservatory)
- **Replace with:** DM Sans, Plus Jakarta Sans, or Fraunces (to be selected during implementation based on feel)
- **Body:** Keep Source Serif 4 or switch to paired sans-serif
- **UI elements:** Keep Inter

### Page Structure

**Section 1: Hero**

- "A teacher for every pianist."
- Subline from pitch
- "Try It Free" CTA
- Animated product preview (waveform + feedback cards) on gradient background
- Drop the social proof strip

**Section 2: Problem**

- No section label
- Two lines, centered, on deeper gradient
- "Any app can tell you if you played the right notes..."

**Section 3: Feedback Showcase**

- "Here's what Crescend hears"
- One real feedback example inline (piece name, performer, 1-2 feedback points)
- Replaces both "How It Works" and "What You'll Learn"

**Section 4: How It Works (compact)**

- "Record. Upload. Get feedback." -- horizontal strip, small icons, one line each
- Not a full section with cards and step numbers

**Section 5: Mission + Credibility (combined)**

- "Quality feedback shouldn't cost $200 an hour..."
- Proof points below: "55% more accurate. 30+ educator interviews. Published on arXiv."
- No "Credibility" label, no founder bio

**Section 6: Final CTA**

- "Ready to hear what your playing really sounds like?"
- "Try It Free"

**Footer:** Unchanged.

### What Gets Removed

- `apps/docs/website-overhaul-design.md` (delete file)
- Standalone "Credibility" section with section label
- Founder bio / personal story prominence
- Social proof strip in hero
- Cormorant Garamond font
- Flat white section backgrounds

### What Stays

- Warm color palette (shifted from flat to gradient)
- Product preview animation in hero (restyled)
- Header navigation, footer, all routes
- `/analyze` page (unchanged)
- All backend/services (unchanged)

## Files to Modify

- `apps/web/src/pages/landing.rs` -- rewrite all sections with new copy, structure, gradients
- `apps/web/tailwind.css` -- add gradient utilities, waveform texture classes, update typography tokens
- `apps/web/src/app.rs` -- update font imports (swap Cormorant Garamond for new display font)
- `apps/web/tailwind.config.js` -- update fontFamily config if display font changes

## Files to Delete

- `apps/docs/website-overhaul-design.md`

## Out of Scope

- `/analyze` page changes
- Backend/service changes
- New routes or navigation changes
- Mobile app
- User accounts
