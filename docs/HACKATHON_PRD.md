# CrescendAI: Product Requirements Document

**xAI Hackathon | December 6-7, 2025 | Track: Grok Radio**

---

## One-Liner

Talk to Grok. Play piano. Get coached. The first AI piano teacher that lives inside your Grok conversation.

---

## The Problem

Learning piano alone means practicing mistakes for hours—or weeks—before anyone catches them. Traditional lessons are expensive ($60-150/hour), infrequent (once weekly), and unavailable when you're actually practicing. Apps like Yousician gamify note-hitting but don't teach *musicality*—the timing, dynamics, and expression that separate mechanical playing from artistry.

**The gap:** No tool combines real-time performance analysis with genuinely helpful, personality-driven coaching.

---

## The Solution

CrescendAI extends Grok's capabilities through xAI's Remote MCP Tools, enabling voice-driven piano lessons where Grok orchestrates everything—listening to your performance, analyzing it, and coaching you with personality.

**The key insight:** Grok can't process audio directly (yet), but it *can* call external tools. We build the audio analysis bridge; Grok provides the intelligence and personality.

**Demo flow:**

1. User opens Grok (voice mode) on phone + CrescendAI companion app on laptop
2. Companion displays session code: `A7X2`
3. User says: "Hey Grok, start my piano lesson. My session code is A7X2."
4. Grok calls MCP tool → Companion confirms connection, shows "Ready"
5. User selects their recording in companion app (amateur or professional)
6. User says: "Analyze my performance"
7. Grok calls MCP tool → Gets analysis → Streams feedback with personality
8. Grok calls tools to highlight problem areas, play reference, suggest exercises
9. User asks follow-up questions; Grok maintains full lesson context

**What makes it different:**

- **Grok IS the interface.** You talk to Grok, not our app. Grok orchestrates the entire lesson through MCP tool calls.
- **Uses xAI's newest feature.** Remote MCP Tools let Grok call our analysis backend autonomously.
- **Real-time X integration.** Grok can search X for piano tips, famous interpretations, and trending discussions mid-lesson—no other AI can do this.
- **Guided personality.** Our tool responses include teaching hints that shape Grok's feedback style—encouraging first, specific second, always musical.
- **Future-proof.** When Grok adds native audio input, we flip a switch and go fully native.

---

## Session Linking Design

The companion app and Grok need to communicate. Here's how:

```
┌─────────────────────────────────────────────────────────────────┐
│  COMPANION APP (Laptop Browser)                                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │     Your session code:  A 7 X 2                          │   │
│  │                                                          │   │
│  │     Tell Grok: "My session code is A7X2"                 │   │
│  │                                                          │   │
│  │     Status: ⏳ Waiting for Grok to connect...            │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

User says to Grok: "Start my piano lesson. My session code is A7X2."

Grok calls: start_lesson(session_code="A7X2", piece="pathetique")

┌─────────────────────────────────────────────────────────────────┐
│  COMPANION APP (Connected)                                       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │     ✅ Connected to Grok                                 │   │
│  │     Piece: Pathétique Sonata, 3rd Movement               │   │
│  │                                                          │   │
│  │     Select recording:                                    │   │
│  │     ○ My Practice Recording (amateur)                    │   │
│  │     ○ Professional Reference                             │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Why this works:**

- 4-character codes are easy to say aloud ("A7X2" not "session-id-7f3a9b2c")
- Natural in conversation: "My session code is..." feels like joining a video call
- No QR scanning, no account linking, no complexity
- Bulletproof for demo—code is visible on screen, user just reads it

**For the hackathon demo:** Pre-generate a memorable code like `PIANO` or `JAI1` so it flows naturally in the pitch.

---

## Demo Recordings

Two pre-recorded performances of **Pathétique Sonata, 3rd Movement** (Beethoven's most dramatic piano sonata—judges will recognize the opening immediately):

| Recording | Description | Purpose |
|-----------|-------------|---------|
| **Amateur (Jai's recording)** | Real practice session with authentic mistakes: rushed passages in development, uneven runs, dynamic inconsistencies | Shows error detection + coaching |
| **Professional Reference** | Clean performance with musical expression | Shows positive reinforcement + nuanced feedback on interpretation |

**Why pre-recorded (not live):**

- Controlled audio quality (no venue noise, no laptop mic issues)
- Authentic mistakes that demonstrate the analysis (not artificial errors)
- Faster demo flow (no waiting for someone to play)
- Can still show "live" analysis—user selects recording, Grok analyzes it

**Demo narrative:** "This is me practicing last week. Let's see what Grok catches."

---

## Why This Must Be Grok (Not ChatGPT, Not Claude)

### 1. Remote MCP Tools (Only Grok Has This)

Grok's Remote MCP Tools allow Grok to call external servers autonomously. This is what makes the architecture possible:

```python
tools=[
    mcp(server_url="https://crescendai.modal.run/mcp"),
    x_search(),      # Real-time X integration
]
```

### 2. Real-Time X Integration (Unique to Grok)

Mid-lesson, Grok can search X for:

- What pianists are saying about Pathétique right now
- Tips from piano teachers on specific techniques  
- Famous interpretations and discussions

> "I just checked X—there's a thread from @PianoPedagogy about this exact passage. They recommend practicing the left hand octaves separately before combining."

### 3. MCP Tool Calling as Lesson Orchestration

Grok doesn't just respond—it *conducts* the lesson by calling tools:

| MCP Tool | What It Does | Grok Trigger |
|----------|--------------|--------------|
| `start_lesson` | Links session, loads piece | "Start my piano lesson, session A7X2" |
| `analyze_performance` | Transcribes audio, analyzes, returns metrics + teaching hints | "Analyze my performance" |
| `play_reference` | Plays correct version on companion | "Let me show you how it should sound..." |
| `highlight_measures` | Marks problem areas on piano roll | "Watch the red sections..." |
| `play_comparison` | A/B student vs. reference | "Hear the difference?" |
| `set_practice_tempo` | Adjusts playback speed | "Try it at 60% tempo" |
| `get_measure_detail` | Deep dive on specific measure | "What's wrong with measure 47?" |

### 4. Guided Personality Through Tool Responses

Our tools don't just return data—they return **teaching hints** that shape Grok's response:

```json
{
  "analysis": {
    "overall_accuracy": 0.84,
    "tempo_stability": 0.71,
    "measures_with_issues": [...]
  },
  "teaching_context": {
    "lead_with": "The dramatic opening chords show real commitment to the character of the piece",
    "primary_focus": "Tempo rushing in the sixteenth-note passages (measures 43-51)",
    "analogy_suggestion": "The left hand octaves are like a heartbeat—steady, inevitable, grounding the drama above",
    "personality_note": "Be encouraging but direct. This student can handle specific criticism."
  }
}
```

Grok receives these hints and naturally incorporates them, maintaining its wit while following pedagogically sound guidance.

---

## Example Conversation

> **User:** "Hey Grok, start my piano lesson. My session code is A7X2. I'm working on the Pathétique."
>
> **Grok:** *[calls start_lesson]* "Got it—I'm connected to your companion app and I've loaded the Pathétique, third movement. Beethoven's angriest rondo. Select your recording and tell me when you're ready for feedback."
>
> *[User selects amateur recording in companion app]*
>
> **User:** "Analyze my performance."
>
> **Grok:** *[calls analyze_performance, receives analysis + teaching hints]*
>
> "Alright, let's dig in. First—those opening chords? You're committing to them. That's the right instinct; this piece demands drama, not politeness.
>
> Now, measures 43 through 51... *[calls highlight_measures]* ...see the red section on your screen? You're rushing. The sixteenth notes are tumbling over each other like they're late for something. But here's the thing—the left hand octaves are supposed to be the heartbeat underneath. Steady. Inevitable. When you rush, the heartbeat goes into arrhythmia.
>
> *[calls play_comparison]*
>
> Listen to this comparison. Your version, then the reference... Hear how the reference keeps that pulse locked? Try it at 60% tempo. *[calls set_practice_tempo]* I've slowed it down on your app. Play just measures 43-51, and pretend your left hand is a metronome that doesn't know your right hand exists."
>
> **User:** "Why do I always rush there?"
>
> **Grok:** "Probably because the right hand pattern is hard and your brain is trying to 'get through it.' Classic survival instinct. But Beethoven isn't trying to kill you—he's trying to build tension. The slower you play that passage, the more dramatic it becomes. Counterintuitive, right?"

---

## Target User

**Primary:** Adult piano learners (intermediate level) practicing independently between lessons.

**Why them:**

- Motivated enough to record and analyze their practice
- Sophisticated enough to appreciate nuanced feedback
- Underserved by gamified apps designed for beginners
- Comfortable with voice assistants

---

## Success Metrics (Demo Day)

| Metric | Target |
|--------|--------|
| Session linking (code → connected) | <3 seconds |
| Analysis completion (select → first token) | <5 seconds |
| Demo completion without failure | 100% (with fallbacks) |
| Judge "wow" moment | Grok calling tools + companion reacting live |
| "Why Grok?" answer | MCP demo + X integration + personality |

---

## The Pitch (90 seconds)

> "I spent four years at Berklee College of Music and performed over a hundred classical concerts. I know what it's like to practice alone for hours, unsure if you're reinforcing mistakes.
>
> What if you could just talk to your teacher?
>
> *[Shows phone with Grok, points to laptop with companion app showing session code]*
>
> 'Hey Grok, start my piano lesson. My session code is PIANO.'
>
> *[Companion app shows: ✅ Connected to Grok]*
>
> This is me practicing the Pathétique last week. Let's see what Grok finds.
>
> 'Analyze my performance.'
>
> *[Grok streams response, companion highlights measures 43-51 in red]*
>
> See that? Grok called our analysis tools, found where I was rushing, and is coaching me through it—with personality, with specifics, with actual playback.
>
> *[Grok calls play_comparison, companion plays A/B]*
>
> It's orchestrating the entire lesson. And because it's Grok, it can pull tips from X in real-time. No other AI can do that.
>
> That's CrescendAI. The first AI piano teacher that lives inside your conversation."

---

## Builder Profile

**Jai** — ML Engineer & Classical Pianist

- **Music:** Bachelor of Music in Piano Performance from Berklee College of Music. 100+ concert performances including international tours. Performed with Sonoma County Philharmonic and Boston Philharmonic.

- **Engineering:** Founding engineer at Capture (privacy-focused social platform). Built recommendation systems achieving 0.68 Recall@5, metrics frameworks tracking 50+ KPIs, infrastructure handling 10K+ requests/minute.

- **AI/ML:** Building audio ML systems using MERT-330M architecture, multi-modal fusion approaches. Winner of Inception AI Buildathon (1st place), Meta Executorch competition (2nd place).

- **Why this project:** "I've sat on both sides—as a performer who needed better feedback, and as an engineer who can build it. CrescendAI is where those worlds collide."

---

## Scope

### In Scope (Hackathon MVP)

- Voice-driven lesson flow via Grok + MCP
- Companion web app with session linking, recording selection, piano roll
- MCP server with core tools (start_lesson, analyze, playback, highlight)
- Pre-recorded audio files (amateur + professional)
- Pre-transcribed MIDI fallback (if live transcription fails)
- Teaching hints in tool responses for guided personality
- Single piece: Pathétique Sonata, 3rd Movement

### Out of Scope

- Live audio recording (using pre-recorded files)
- Native Grok audio input (waiting on xAI)
- Multiple piece support
- User accounts / progress persistence  
- Mobile companion app

### Stretch Goals

- X search integration demo (if Grok naturally uses it)
- Additional tool: `suggest_exercise` with specific drills
- Voice output from Grok

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| MCP tools don't work as expected | Day 1 proof-of-concept; have direct API fallback |
| Session linking confuses judges | Pre-generate memorable code (PIANO); practice the flow |
| Audio transcription fails | Pre-transcribed MIDI files load instantly |
| Grok ignores teaching hints | Hints are suggestions; Grok's default personality is good |
| Companion ↔ MCP latency | WebSocket connection; pre-warm before demo |
| Cold start delay | Modal min_containers=2, pre-warming protocol |
| Demo nerves | Pre-recorded video backup at 720p |
| WiFi issues | Phone hotspot ready |

---

## Audio & MIDI Assets Checklist

| Asset | Format | Purpose |
|-------|--------|---------|
| `pathetique_amateur.mp3` | MP3/WAV | Jai's practice recording with authentic mistakes |
| `pathetique_amateur.mid` | MIDI | Pre-transcribed fallback |
| `pathetique_professional.mp3` | MP3/WAV | Clean reference performance |
| `pathetique_professional.mid` | MIDI | Pre-transcribed fallback |
| `pathetique_reference.mid` | MIDI | Ground truth score for alignment |
| `pathetique_reference.xml` | MusicXML | Score data for measure mapping |
