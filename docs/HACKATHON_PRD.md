# CrescendAI: Product Requirements Document

**xAI Hackathon | December 6-7, 2025 | Track: Grok Radio**

---

## One-Liner

An AI piano teacher that listens to your performance, catches every mistake, and coaches you with the wit of Douglas Adams and the expertise of a conservatory professor.

---

## The Problem

Learning piano alone means practicing mistakes for hours—or weeks—before anyone catches them. Traditional lessons are expensive ($60-150/hour), infrequent (once weekly), and unavailable when you're actually practicing. Apps like Yousician gamify note-hitting but don't teach *musicality*—the timing, dynamics, and expression that separate mechanical playing from artistry.

**The gap:** No tool combines real-time performance analysis with genuinely helpful, personality-driven coaching.

---

## The Solution

CrescendAI analyzes piano recordings and delivers instant, measure-specific feedback through Grok—not as a generic chatbot, but as an opinionated, encouraging teacher who orchestrates your entire practice session.

**Demo flow:**

1. User uploads a recording (or we use pre-loaded examples)
2. Grok introduces itself and the piece, setting context
3. System analyzes performance against the reference score
4. Grok delivers streaming feedback—specific measures, timing issues, dynamics
5. Grok calls functions to play the "correct" version, highlight problem areas, suggest exercises
6. User can ask follow-up questions; Grok maintains lesson context

**What makes it different:**

- **Grok as conductor, not commentator.** Function calling lets Grok orchestrate the lesson—triggering playback, adjusting difficulty, sequencing exercises—rather than just describing what went wrong.
- **Personality that sticks.** Feedback in the style of a Hitchhiker's Guide narrator: witty, warm, occasionally irreverent, but always pedagogically sound.
- **Symbolic depth under the hood.** Score-aligned analysis catches timing deviations, wrong notes, articulation issues, and dynamic inconsistencies—not just "you hit the wrong key."

---

## Demo Recordings

Two pre-loaded performances of **Für Elise** (universally recognized, demonstrates clear contrast):

| Recording | Description | Purpose |
|-----------|-------------|---------|
| **Student Version** | Intentionally flawed: rushed tempo in measures 3-4, missed notes in arpeggios, flat dynamics | Shows error detection + coaching |
| **Professional Version** | Clean performance with musical expression | Shows positive reinforcement + nuanced feedback |

The demo narrative: "Let's hear the difference between practicing alone and practicing with CrescendAI."

---

## Grok Integration (Why This Must Be Grok)

### Function Calling as Lesson Orchestration

Grok doesn't just respond—it *conducts* the lesson by calling tools:

| Function | What It Does | Example Trigger |
|----------|--------------|-----------------|
| `play_reference` | Plays the correct version of a specific passage | "Let me show you how measure 5 should sound..." |
| `highlight_measures` | Visually marks problem areas on piano roll | "Watch the red sections—that's where we need work" |
| `set_tempo` | Adjusts practice tempo | "Let's slow this down to 60% speed" |
| `play_comparison` | A/B comparison of student vs. reference | "Hear the difference in the left hand?" |
| `suggest_exercise` | Recommends targeted practice | "Try this five-finger pattern to build evenness" |

This makes Grok the **intelligent orchestrator**, not a text formatter.

### Personality as Differentiator

The teaching persona is distinctively Grok:

- Leads with encouragement, but doesn't sugarcoat
- Makes musical concepts accessible through unexpected analogies
- Occasionally breaks the fourth wall ("Yes, I'm an AI, but I've listened to more Beethoven than anyone you know")
- Culturally aware—can reference trending performances, famous recordings, pianist memes

**Example feedback:**
> "Right, so here's the thing about measure 3—you're playing it like you're late for a bus. Beethoven wrote those sixteenth notes to *breathe*, not sprint. Listen to how Barenboim handles it... [plays reference]. See? The rubato isn't random; it's gravitational. Try again at 70% tempo, and pretend you're being paid by the hour."

---

## Target User

**Primary:** Adult piano learners (intermediate level) practicing independently between lessons.

**Why them:**

- Motivated enough to record and analyze their practice
- Sophisticated enough to appreciate nuanced feedback
- Underserved by gamified apps designed for beginners

---

## Success Metrics (Demo Day)

| Metric | Target |
|--------|--------|
| End-to-end latency (upload → first token) | <4 seconds |
| Demo completion without failure | 100% (with fallbacks) |
| Judge "wow" reaction | Streaming feedback + function call in action |
| Memorable pitch | Personal story + personality demo |

---

## The Pitch (90 seconds)

> "I spent four years at Berklee College of Music and performed over a hundred classical piano concerts. I know what it's like to practice alone for hours, unsure if you're reinforcing mistakes.
>
> CrescendAI is the teacher I wished I had.
>
> *[Plays student recording—notes appear on piano roll]*
>
> Watch what happens. Grok analyzes the performance in real-time, catches the rushed tempo in measure 3, and doesn't just tell me—it *shows* me.
>
> *[Grok streams: "Ah, measure 3. You're treating those sixteenth notes like they owe you money..."]*
>
> *[Grok calls function: plays reference passage]*
>
> See that? Grok orchestrates the entire lesson—playing examples, highlighting problems, suggesting exercises. It's not a chatbot bolted onto a music app. It's an AI that teaches like a human would, with personality and patience.
>
> *[Shows professional recording analysis for contrast]*
>
> That's CrescendAI. Every student deserves a teacher this attentive."

---

## Builder Profile

**Jai** — ML Engineer & Classical Pianist

- **Music:** Bachelor of Music in Piano Performance from Berklee College of Music. 100+ concert performances including international tours. Performed with Sonoma County Philharmonic and Boston Philharmonic.

- **Engineering:** Founding engineer at Capture (privacy-focused social platform). Built recommendation systems achieving 0.68 Recall@5, metrics frameworks tracking 50+ KPIs, infrastructure handling 10K+ requests/minute.

- **AI/ML:** Building audio ML systems using MERT-330M architecture, multi-modal fusion approaches. Winner of Inception AI Buildathon (1st place), Meta Executorch competition (2nd place).

- **Why this project:** "I've sat on both sides—as a performer who needed better feedback, and as an engineer who can build it. CrescendAI is where those worlds collide."

---

## Out of Scope (For Hackathon)

- Real-time MIDI input (no hardware available)
- Multiple piece support (Für Elise only)
- User accounts / progress persistence
- Mobile app
- Voice output (stretch goal only)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Audio transcription fails | Pre-transcribed MIDI files as instant fallback |
| Grok API timeout | Cached example responses ready to display |
| Cold start delay | Pre-warming protocol, min_containers=2 |
| Demo nerves | Pre-recorded video backup at 720p |
| "Why Grok?" question | Function calling demo + personality contrast prepared |
