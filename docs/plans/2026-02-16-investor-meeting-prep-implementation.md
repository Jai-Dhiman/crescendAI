# Investor Meeting Preparation - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build all materials for a casual pre-seed angel investor coffee meeting next week.

**Architecture:** HTML-based deliverables (slide deck via reveal.js, one-pager as print-friendly HTML) plus demo preparation and rehearsal materials. All investor materials live in `investor/` at the project root.

**Tech Stack:** HTML, CSS, reveal.js (CDN), browser print-to-PDF

**Design doc:** `docs/plans/2026-02-16-investor-meeting-prep-design.md`

---

### Task 1: Set Up Investor Materials Directory

**Files:**
- Create: `investor/deck/index.html`
- Create: `investor/one-pager/index.html`

**Step 1: Create directory structure**

```bash
mkdir -p investor/deck investor/one-pager
```

**Step 2: Scaffold the reveal.js deck**

Create `investor/deck/index.html` with reveal.js loaded from CDN. Use a minimal, clean theme (white background, dark text, no distracting transitions). Include:
- reveal.js CSS + JS from CDN (v5.x)
- PDF export plugin (built into reveal.js -- just append `?print-pdf` to URL)
- Custom CSS overrides for clean typography (Inter for body, serif for headings)
- Viewport configured for 16:9 aspect ratio

**Step 3: Scaffold the one-pager**

Create `investor/one-pager/index.html` as a single-page HTML document with print-friendly CSS:
- `@media print` styles for clean PDF output
- A4/Letter page size constraint
- No scrolling -- everything fits one page
- Clean typography matching the deck

**Step 4: Verify both files open in browser**

```bash
open investor/deck/index.html
open investor/one-pager/index.html
```

**Step 5: Commit**

```bash
git add investor/
git commit -m "scaffold investor materials directory with deck and one-pager"
```

---

### Task 2: Build Slide Deck -- Slides 1-5 (Story Half)

**Files:**
- Modify: `investor/deck/index.html`

Reference: Design doc Section 3 (Lightweight Deck)

**Step 1: Build Slide 1 (Title)**

Content:
- "CrescendAI" as large heading
- Tagline: "Expert-level piano feedback, powered by AI"
- Jai Dhiman | crescend.ai | jai.dhiman@outlook.com
- Clean, minimal -- no clutter

**Step 2: Build Slide 2 (The Problem)**

Content -- use short, punchy bullet points:
- "Piano students practice alone most of the time"
- "Expert feedback costs $50-200/hour and is hard to access"
- "Existing apps only check note accuracy -- they miss what actually matters"
- "30+ educator interviews confirmed this gap"

Design: Consider a simple visual split -- text on one side, a striking stat or quote on the other.

**Step 3: Build Slide 3 (The Insight)**

Content:
- "Audio foundation models can hear what MIDI can't"
- "Trained on millions of hours of music, they understand how music sounds"
- "These models only became capable enough in the last 1-2 years"

This slide should convey the "why now" -- the technological timing.

**Step 4: Build Slide 4 (The Product)**

Content:
- Screenshot of the CrescendAI demo page (radar chart + teacher feedback)
- "Upload a recording. Get 19-dimension analysis in under 15 seconds."
- "Radar chart + natural language feedback + practice tips"
- "Runs on Cloudflare edge -- costs <$20/month"

Note: Take a screenshot of crescend.ai demo results page and save to `investor/deck/img/product-screenshot.png`. If not possible, use a placeholder box with descriptive text.

**Step 5: Build Slide 5 (The Results)**

Content:
- Large stat: "+55% vs existing approaches"
- "R-squared = 0.537 vs 0.347 (symbolic baseline)"
- "Statistical significance: p < 10^-25"
- "Validated across piano timbres, difficulty levels, and performers"
- "Published: arXiv paper, submitted to ISMIR 2026"

Make the 55% number visually prominent -- this is the most impressive proof point.

**Step 6: Preview in browser and verify flow**

```bash
open investor/deck/index.html
```

Arrow keys should navigate between slides. Check that text is readable, not overcrowded, and the visual flow tells a story.

**Step 7: Commit**

```bash
git add investor/deck/
git commit -m "add slides 1-5: title, problem, insight, product, results"
```

---

### Task 3: Build Slide Deck -- Slides 6-10 (Business Half)

**Files:**
- Modify: `investor/deck/index.html`

Reference: Design doc Section 3 (Lightweight Deck)

**Step 1: Build Slide 6 (Market)**

Content:
- "~40M piano students globally"
- "Online music education growing post-COVID"
- Revenue path: "B2C subscription ($10-30/mo) -> Institutional licenses -> API for piano manufacturers"
- Keep it simple -- one clear market size number, one clear revenue model

**Step 2: Build Slide 7 (Traction & Validation)**

Content:
- "30+ structured educator interviews"
- "Published research: arXiv, ISMIR 2026 submission"
- "Iterated with MIR researchers, ML engineers at OpenAI and Google"
- "3x hackathon winner | Founding engineer at Capture (0->50 users)"
- "890K+ lines shipped across CrescendAI"

**Step 3: Build Slide 8 (Founder)**

This is the most personal slide. Content:
- "Jai Dhiman"
- "Berklee College of Music -- Percussion Performance (5x Dean's List)"
- "Pianist since age 8 | Active orchestral musician"
- "Self-taught ML engineer -> Founding engineer -> Founder"
- "The rare combination: deep musical expertise + builds the entire stack"

Consider: a simple layout with text, no photo needed for a casual meeting backup deck.

**Step 4: Build Slide 9 (Roadmap)**

Content as a simple timeline or progression:
- Now: "Curated gallery with AI feedback (live at crescend.ai)"
- 3 months: "User uploads, accounts, progress tracking"
- 6 months: "Mobile app, real-time analysis"
- Research: "Dual-encoder architecture (audio + score), large-scale data expansion"

**Step 5: Build Slide 10 (The Ask)**

Content -- keep deliberately vague per the design:
- "Pre-seed funding to accelerate"
- Three bullets: "First ML engineer hire | GPU training credits | User research and iteration"
- Contact info repeated
- Tone: invitation, not hard sell

**Step 6: Preview full deck end-to-end**

```bash
open investor/deck/index.html
```

Navigate all 10 slides. Check: readability, consistent styling, narrative flow from problem through ask. No slide should have more than 5-6 bullet points.

**Step 7: Commit**

```bash
git add investor/deck/
git commit -m "add slides 6-10: market, traction, founder, roadmap, ask"
```

---

### Task 4: Build One-Pager

**Files:**
- Modify: `investor/one-pager/index.html`

Reference: Design doc Section 4 (One-Pager)

**Step 1: Build the top third (headline)**

Content:
- "CrescendAI" as heading
- "AI-powered piano performance analysis"
- One sentence: "We use audio foundation models to deliver expert-level feedback on piano performances across 19 musical dimensions."

**Step 2: Build the middle third (three proof points)**

Three columns or three stacked sections:
- "It works" -- 55% better than existing approaches. Published, peer-reviewed.
- "It's real" -- Live demo at crescend.ai. Under 15-second analysis time.
- "There's demand" -- 30+ educator interviews. ~40M piano students globally.

Each proof point should be scannable in 3 seconds.

**Step 3: Build the bottom third (founder + contact)**

Content:
- "Founded by Jai Dhiman"
- "Berklee College of Music (Percussion Performance) | Pianist since age 8"
- "First-author arXiv paper | ISMIR 2026 submission | 3x hackathon winner"
- crescend.ai | jai.dhiman@outlook.com | linkedin.com/in/jai-d

**Step 4: Verify print layout**

```bash
open investor/one-pager/index.html
```

Use Cmd+P to check print preview. Everything should fit on one page with clean margins. No headers/footers from browser. Text should be readable when printed.

**Step 5: Commit**

```bash
git add investor/one-pager/
git commit -m "add investor one-pager leave-behind"
```

---

### Task 5: Demo Preparation and Fallback

**Files:**
- Create: `investor/demo-runbook.md`

**Step 1: Test the live demo**

Open crescend.ai in a browser. Walk through the full demo flow:
1. Landing page loads correctly
2. Navigate to demo page
3. All 3 performances display (Horowitz, Argerich, Gould)
4. Select Horowitz -- analysis runs successfully
5. Radar chart renders with 19 dimensions
6. Teacher feedback generates with citations
7. Audio player works

Document any issues found.

**Step 2: Pre-cache demo results**

Run the analysis for all 3 performances so results are cached. This ensures the demo loads instantly during the meeting (no cold-start GPU wait).

**Step 3: Write the demo runbook**

Create `investor/demo-runbook.md` with:
- Exact click-by-click flow (from design doc Section 2)
- Talking points for each screen
- Which 2-3 radar dimensions to highlight for Horowitz and why
- Specific sentences to read from teacher feedback
- The closer line: "This is what a $200/hour masterclass teacher gives you, available to anyone with a recording, in 15 seconds."

**Step 4: Create backup plan notes**

Add to the runbook:
- Instructions for recording a screen capture backup (QuickTime on Mac: Cmd+Shift+5)
- Note to have the recording saved on phone as well
- Offline fallback: screenshots of each demo step saved in `investor/deck/img/`

**Step 5: Commit**

```bash
git add investor/demo-runbook.md
git commit -m "add demo runbook with fallback plan"
```

---

### Task 6: Deck Visual Polish and Screenshots

**Files:**
- Modify: `investor/deck/index.html`
- Create: `investor/deck/img/` (screenshots)

**Step 1: Capture product screenshots**

Take screenshots of the CrescendAI demo:
1. Landing page hero section
2. Demo gallery with 3 performances
3. Analysis results: radar chart + teacher feedback for Horowitz
4. Save to `investor/deck/img/`

**Step 2: Add screenshots to Slide 4 (Product)**

Embed the results screenshot in the product slide. Size it to be clearly visible but not overwhelming -- the screenshot should demonstrate, not dominate.

**Step 3: Visual consistency pass**

Review all 10 slides for:
- Consistent font sizes (headings, body, stats)
- Consistent spacing and margins
- Color palette: stick to 2-3 colors max (dark text, one accent for key stats, white background)
- Key numbers (55%, 40M, <15s, $20/month) should be visually prominent wherever they appear

**Step 4: Test PDF export**

Open the deck with `?print-pdf` appended to the URL. Use Cmd+P to save as PDF. Verify all slides export cleanly.

```bash
open "investor/deck/index.html?print-pdf"
```

**Step 5: Commit**

```bash
git add investor/
git commit -m "add screenshots and visual polish to deck"
```

---

### Task 7: Final Review and Practice Materials

**Files:**
- Create: `investor/practice-checklist.md`

**Step 1: Create practice checklist**

Write `investor/practice-checklist.md` with:

**Before the meeting:**
- [ ] Test crescend.ai loads on your phone and laptop
- [ ] Run analysis on all 3 performances to warm cache
- [ ] Record screen capture backup of full demo flow
- [ ] Save backup recording on phone
- [ ] Review founder story arc (design doc Section 1)
- [ ] Review FAQ answers (design doc Section 6)
- [ ] Review investor questions to ask (design doc Section 5)
- [ ] Open deck on laptop, ready to pull up if needed
- [ ] Print one-pager or have PDF on phone to share

**During the meeting:**
- [ ] Lead with your story, not the deck
- [ ] Show the demo when conversation naturally goes there
- [ ] Ask at least 3 of the 7 prepared questions
- [ ] Save questions 6-7 (network + follow-up) for last 5 minutes
- [ ] Don't explain technical architecture unless asked

**After the meeting:**
- [ ] Send follow-up email within 24 hours
- [ ] Attach: one-pager PDF + deck PDF
- [ ] Reference something specific from the conversation
- [ ] Thank them for their time and advice

**Step 2: Write follow-up email template**

Add a template to the checklist file:

Subject: "Great meeting -- CrescendAI follow-up"
Body structure:
- Thanks for time + reference specific conversation point
- "As promised, attached: [one-pager] and [deck]"
- "I'd love to keep you posted on progress. [Next milestone]."
- Sign off

**Step 3: Commit**

```bash
git add investor/
git commit -m "add practice checklist and follow-up email template"
```

---

## Execution Notes

**Total tasks:** 7
**Estimated effort:** Tasks 1-5 are the core work. Task 6 depends on having the live demo accessible. Task 7 is quick.

**Dependencies:**
- Task 2 depends on Task 1
- Task 3 depends on Task 1
- Task 4 depends on Task 1
- Tasks 2, 3, 4 can run in parallel after Task 1
- Task 5 is independent
- Task 6 depends on Tasks 2, 3, and 5
- Task 7 depends on all other tasks

**What's already done (from design doc):**
- Investor questions (Section 5) -- ready to use as-is
- FAQ prep (Section 6) -- ready to use as-is
- User personas (Section 7) -- ready to use as-is
- Financial model (Section 8) -- ready to use as-is
- Competitive landscape (Section 9) -- ready to use as-is
- Market sizing (Section 10) -- ready to use as-is
- Founder story arc (Section 1) -- ready to practice from

These don't need separate deliverables -- they live in the design doc as reference material.
