# Pre-Seed Angel Investor Meeting Preparation

**Date:** 2026-02-16
**Meeting context:** Casual coffee/call with angel investor (warm intro), next week
**Goal:** Build relationship, get advice. Investment could follow naturally.

---

## Deliverables

### For the Meeting (A+B)

1. **Founder Story Script** -- 2-minute narrative arc
2. **Live Demo Runbook** -- what to show, in what order, with fallback
3. **Lightweight Deck (10 slides)** -- backup visual aid and leave-behind
4. **One-Pager** -- clean summary to forward
5. **Investor Questions** -- 7 questions to ask them
6. **FAQ Prep** -- answers to 12 common angel questions

### For Own Preparation (C Exercise)

1. **User Personas** -- 3 personas with user journeys
2. **Financial Model** -- unit economics, growth scenarios, runway
3. **Competitive Landscape** -- positioning map
4. **Market Sizing** -- TAM/SAM/SOM

---

## 1. Founder Story Arc (2 minutes)

**Hook (~20s):**
"I've been playing piano since I was 8. I went to Berklee for percussion, toured internationally with orchestras, and I still play in local orchestras and practice piano every day."

**The Problem You Lived (~30s):**
"When you're practicing alone -- which is most of the time -- you don't have a teacher's ear. You can hear something is off, but you can't always pinpoint what. Is it your pedaling? Your dynamics? Your phrasing? A great teacher can tell you in seconds. But access to that level of feedback is expensive, infrequent, and gatekept."

**The Insight (~30s):**
"I taught myself ML engineering and discovered something: new audio foundation models -- trained on millions of hours of music -- can actually hear the things that matter. Not just right notes and wrong notes like MIDI-based apps, but tone quality, pedal clarity, dynamic shaping. The stuff that separates a good performance from a great one. I proved it: 55% better than existing approaches, published as a first-author paper."

**Demo Transition (~20s):**
"Let me show you what it actually does."

**Vision (~20s):**
"Right now it evaluates curated performances across 19 dimensions and gives you feedback like a warm, knowledgeable teacher. Next step: any pianist uploads a recording and gets that feedback in under 15 seconds."

**Principles:**

- This is a conversation starter, not a monologue
- If they interrupt with questions, let it flow
- No jargon beyond "audio foundation models"
- The demo does the heavy lifting

---

## 2. Live Demo Runbook

**Flow (3-4 minutes):**

1. Open the gallery -- "Here are performances from some of the greatest pianists: Horowitz, Argerich, Zimerman..."
2. Select a performance -- Pick Horowitz for name recognition. Brief context.
3. Run the analysis -- Let them watch the 5-15s loading state.
4. Walk through radar chart -- Pick 2-3 interesting dimensions. "It rated his dynamics and timbre extremely high, which is exactly what Horowitz is known for. But pedal clarity is lower, consistent with his heavy pedaling style."
5. Show teacher feedback -- Read a sentence or two aloud. Point out it's specific and actionable.
6. Closer -- "This is what a $200/hour masterclass teacher gives you, available to anyone with a recording, in 15 seconds."

**Fallback plan:**

- Screen recording on phone in case of wifi issues
- Pre-cache results for instant display if GPU endpoint is cold

**What NOT to do:**

- Don't explain technical architecture during demo
- Don't apologize for things not built yet
- Don't show more than 1-2 performances

---

## 3. Lightweight Deck (10 Slides)

**Slide 1: Title**

- CrescendAI -- "Expert-level piano feedback, powered by AI"
- Name, crescend.ai, contact

**Slide 2: The Problem**

- Students practice alone most of the time
- Feedback is expensive ($50-200/hr), infrequent, gatekept
- Existing apps only check note accuracy (MIDI) -- miss what matters
- 30+ educator interviews confirmed this gap

**Slide 3: The Insight**

- Audio foundation models hear what MIDI can't
- Trained on millions of hours of music
- Timing moment: these models only became capable in last 1-2 years

**Slide 4: The Product**

- Demo screenshot or embed
- Upload/select performance -> 19-dimension analysis in <15 seconds
- Radar chart + teacher feedback + practice tips
- Runs on Cloudflare edge, costs <$20/month

**Slide 5: The Results**

- 55% improvement over symbolic approaches (R^2 = 0.537 vs 0.347)
- p < 10^-25
- Validated across soundfonts, difficulty levels, multiple performers
- arXiv paper, ISMIR 2026 submission

**Slide 6: Market**

- ~40M piano students globally
- Online music ed growing post-COVID
- B2C subscription ($10-30/month) -> institutions -> API licensing

**Slide 7: Traction & Validation**

- 30+ educator interviews
- Published research (arXiv, ISMIR 2026)
- Feedback from MIR researchers, ML engineers at OpenAI and Google
- 3x hackathon winner, founding engineer experience (Capture: 0->50 users)
- 890K+ lines shipped

**Slide 8: Founder**

- Berklee College of Music (percussion, 5x Dean's List)
- Pianist since age 8, active orchestral musician
- Self-taught ML engineer -> founding engineer -> founder
- Deep domain expertise + builds the whole stack alone

**Slide 9: Roadmap**

- Now: curated gallery with AI feedback
- 3 months: user uploads, accounts, progress tracking
- 6 months: mobile app, real-time analysis, instrument expansion
- Research: dual-encoder (audio + score), large-scale data

**Slide 10: The Ask**

- Pre-seed to accelerate: first hire, GPU credits, user research
- Keep vague -- goal is relationship-building

---

## 4. One-Pager

**Top third: Headline**

- CrescendAI: AI-powered piano performance analysis
- "We use audio foundation models to deliver expert-level feedback on piano performances across 19 musical dimensions."

**Middle third: Three proof points**

- Works: 55% better than existing approaches (published, peer-reviewed)
- Real product: Live demo at crescend.ai, <15 second analysis
- Real demand: 30+ educator interviews, ~40M piano students globally

**Bottom third: Founder + contact**

- Berklee-trained musician (piano since age 8) + full-stack ML engineer
- arXiv paper, ISMIR 2026 submission, 3x hackathon winner
- Contact info, website

---

## 5. Investor Questions (To Ask Them)

**Understanding their perspective:**

1. "What's your take on the music education space? Venture-scale or niche?"
2. "What separates AI products you're excited about from ones that feel like features?"

**Tactical advice:**
3. "When you've seen solo technical founders work well at pre-seed, what made the difference?"
4. "B2C subscription vs B2B institutional -- any instinct on which to prove first?"

**Understanding their world:**
5. "What are you seeing in pre-seed deal structures? SAFEs, priced rounds?"
6. "Anyone in your network I should be talking to -- investment, advice, or partnerships?"

**Natural close:**
7. "I'd love to keep you in the loop. What's the best way to do that?"

**Usage:** Weave naturally, don't fire as checklist. Lead with #1-2 early. Save #6-7 for last 5 minutes.

---

## 6. FAQ Prep

### Product

**"How does this actually work?"**
A recording goes in. An audio foundation model extracts deep understanding. A prediction head maps to 19 dimensions. A RAG pipeline generates teacher-like feedback. Under 15 seconds.

**"Why audio instead of MIDI?"**
MIDI captures notes and timing. Misses tone, pedal resonance, dynamic shading. Audio approach is 55% more accurate because it hears what a human hears.

**"What stops someone from copying this?"**
Proprietary masterclass dataset, published research advantage, non-trivial training pipeline.

### Market

**"How big is this really?"**
~40M piano students globally. Even a fraction at $15/month is meaningful. Tech generalizes to other instruments.

**"Who's your competition?"**
Piano apps teach what to play, check note accuracy via MIDI. Nobody evaluates how you play from audio.

**"Have you talked to users?"**
30+ structured educator interviews. Consistent feedback: want expression and musicality evaluation, not just note accuracy.

### Founder

**"Can you build this alone?"**
Have been: 250+ commits, published paper, working product. Need help scaling: ML engineer, eventually music educator for curriculum.

**"Why did you leave Capture?"**
Great experience (founding engineer, 0->50 users). CrescendAI is the intersection of everything I care about.

### Business

**"How do you make money?"**
B2C subscription ($10-30/month). Longer term: institutional licenses, API for manufacturers and apps.

**"What's your burn rate?"**
Nearly zero. Product runs <$20/month on Cloudflare. Advantage of solo technical founder on serverless.

### Hard Questions

**"Why now?"**
Audio foundation models only became capable in last 1-2 years. MuQ released in 2024.

**"What if a big company does this?"**
Niche they're unlikely to prioritize. Head start on research, data, and domain expertise.

---

## 7. User Personas

### Persona 1: The Dedicated Self-Learner (B2C Core)

- Age 25-45, serious hobby, 1-3 hours daily practice
- Had lessons as a kid, may not have teacher now
- Frustration: "I can tell something is off but I don't know what to fix"
- Wants: specific, actionable feedback on how they play
- WTP: $15-30/month
- Journey: Record on phone -> upload -> 19-dimension analysis -> adjust practice -> track improvement

### Persona 2: The Music Educator (B2B Entry)

- Piano teacher, 15-40 students
- Frustration: "I only hear each student 30-60 min/week"
- Wants: extend their ears between lessons, see student progress
- WTP: $50-100/month (or school pays)
- Journey: Assign piece -> student uploads practice -> teacher reviews AI analysis before lesson -> more focused lessons

### Persona 3: The Institution (B2B Scale)

- Conservatory, university, competition organizer
- Frustration: "Need consistent, objective evaluation standards"
- Wants: standardized assessment, progress analytics, audition screening
- WTP: $3,000-10,000/year site license
- Journey: Integrate into curriculum -> students submit recordings -> faculty use alongside own evaluation -> track cohorts

**Prioritization:** Start Persona 1 (lowest friction) -> Persona 2 (natural discovery) -> Persona 3 (requires sales, later stage)

---

## 8. Financial Model

### Unit Economics

| Item | Amount |
|------|--------|
| Subscription price | $15/month (starter), $30/month (pro) |
| Infrastructure per user | ~$0.50-2.00/month |
| Gross margin | ~87-97% |

### Growth Scenarios

| Milestone | Users | ARR | Monthly infra |
|-----------|-------|-----|---------------|
| Validation | 100 | $18K | ~$50 |
| Seed-ready | 1,000 | $180K | ~$500 |
| Series A | 10,000 | $1.8M | ~$5,000 |

### Runway on $250K Raise

| Expense | Monthly | Annual |
|---------|---------|--------|
| Founder salary | $5K | $60K |
| ML engineer (contract) | $5-8K | $60-96K |
| GPU credits | $1-2K | $12-24K |
| Infrastructure | $0.5K | $6K |
| Misc (legal, tools) | $0.5K | $6K |
| **Total** | **$12-16K** | **$144-192K** |

15-20 months runway. Enough to hit 1,000-user milestone and raise seed.

### Key Metrics to Track

- Analyses per user per week (engagement)
- Week 4 retention
- Free -> paid conversion rate
- NPS from educator users

---

## 9. Competitive Landscape

### Direct Competitors

| Product | What they do | CrescendAI advantage |
|---------|-------------|---------------------|
| Simply Piano | Note accuracy via microphone | No expression, dynamics, tone |
| Flowkey | Visual note matching | MIDI-level only |
| Piano Marvel | MIDI-connected scoring | Requires MIDI keyboard, misses acoustic qualities |
| Tonestro | Pitch/rhythm matching | Generic, no perceptual dimensions |

### Adjacent Players

| Product | What they do | Why not competing |
|---------|-------------|-------------------|
| Yousician | Gamified learning | Teaches what to play, not how |
| Masterclass | Video lessons | Passive, no personalized feedback |
| SmartMusic | Ensemble practice | Band/orchestra focus |

### Positioning

Everyone else: "Did you play the right notes at the right time?"
CrescendAI: "How did it sound? Was your tone warm? Were your dynamics compelling?"

The difference between a metronome and a teacher.

### Threats

- Big tech unlikely to prioritize this niche
- Simply Piano (Spotify) is MIDI-native, retrofitting audio is a major pivot
- New entrant would face 12-18 month head start from published research + proprietary data

---

## 10. Market Sizing

**TAM:** ~40M active piano students globally. At $15/month = ~$7.2B/year. Extends to all instruments.

**SAM:** English-speaking markets (~10M), digitally engaged (~60%) = 6M. At $15/month = ~$1.1B/year.

**SOM (3-year):** 0.1-0.5% of SAM = 6,000-30,000 users. At blended $20/month = $1.4M-$7.2M ARR.

**Conservative because:**

- Excludes non-English markets (China alone: ~80M piano students)
- Excludes B2B institutional revenue
- Excludes API licensing
- Assumes piano only

**One sentence:** "Even capturing a fraction of a percent of piano students in English-speaking markets gets us to seven-figure ARR, and the technology extends to every instrument."
