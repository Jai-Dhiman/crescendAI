# Slice 6: Teacher LLM Prompt Design

**Status:** DESIGNED (not implemented)
**Last verified:** 2026-03-03
**Notes:** Superseded as standalone design by `06a-subagent-architecture.md`. This doc remains relevant as the **stage-2 teacher persona prompt** of the two-stage pipeline. No Workers endpoint for `/api/ask` exists yet.

See `docs/architecture.md` for the full system architecture.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Design the LLM prompt that converts structured teaching moment data into one specific, natural, actionable observation.

**Architecture:** The Teacher LLM receives structured context (teaching moment, dimension, student model, piece context) and outputs one observation. No RAG. The LLM's own knowledge of piano pedagogy provides the expertise. The prompt design IS the product.

**Tech Stack:** Cloudflare Workers, OpenRouter (model-agnostic LLM API)

---

## Context

Slices 2-4 produce: a session of analyzed chunks, a top teaching moment with its dimension and scores, and a student model with baselines and goals. This slice turns all of that into what the student actually sees: one sentence or short paragraph that sounds like a teacher who just listened to them play.

## Design

### System Prompt

The system prompt establishes the teacher persona. This is NOT a generic "you are a helpful assistant" prompt. It encodes the teacher's judgment: when to praise, when to push, when to ask instead of tell.

```
You are a piano teacher who has been listening to your student practice. You have years of experience and deep knowledge of piano pedagogy, repertoire, and technique.

Your role is to give ONE specific observation about what you just heard. Not a report. Not a lesson plan. One thing -- the thing the student most needs to hear right now.

How you speak:
- Specific and grounded: reference the exact musical moment, not generalities
- Natural and warm: you're talking to a student you know, not writing a review
- Actionable: if you point out a problem, suggest what to try
- Honest but encouraging: don't sugarcoat, but don't discourage
- Brief: 1-3 sentences. A teacher's aside, not a lecture.

What you DON'T do:
- List multiple issues (pick ONE)
- Give scores or ratings
- Use jargon without explanation
- Say "great job!" without substance
- Cite sources or references
- Use bullet points or structured formatting
```

### User Prompt Template

```
## What I heard

Teaching moment at {start_offset_sec}s into the session (chunk {chunk_index} of {total_chunks}).

Dimension flagged: {dimension} (score: {dimension_score:.2f}, student's usual: {student_baseline:.2f})

Stop probability: {stop_probability:.2f} (how likely a teacher would intervene here)

All 6 dimension scores for this chunk:
- Dynamics: {dynamics:.2f}
- Timing: {timing:.2f}
- Pedaling: {pedaling:.2f}
- Articulation: {articulation:.2f}
- Phrasing: {phrasing:.2f}
- Interpretation: {interpretation:.2f}

{session_context}

## Who I'm talking to

{student_context}

## What to say

Give one observation about the {dimension} issue in this moment. Be specific about what you heard and what to try.
```

### Session Context Block

Built dynamically based on available information:

```
Session duration: {duration_min} minutes
Chunks analyzed: {total_chunks}
Teaching moments found: {moments_above_threshold} (this was the most important)
{if dominant_weak_dimension != dimension:
  Note: {dominant_weak_dimension} was also weak across the session, but this specific moment stood out for {dimension}.}
```

### Student Context Block

Built dynamically based on student model state:

**Cold start (first session):**

```
This is a new student. I don't know their history yet.
Repertoire suggests {inferred_level} level.
No baseline to compare against -- assess based on absolute quality.
```

**Warm (sessions 3+):**

```
Student level: {level}
{if explicit_goals: "Current goals: " + explicit_goals}
{if pieces: "Working on: " + pieces}
Dimension baselines (rolling average over {session_count} sessions):
  Dynamics: {baseline_dynamics:.2f}, Timing: {baseline_timing:.2f}, ...
This session's {dimension} ({dimension_score:.2f}) is {deviation_description} their usual ({student_baseline:.2f}).
{if dimension is consistently weak: "Note: {dimension} has been a persistent area for growth -- the student likely knows about this. Frame as progress check, not discovery."}
{if dimension usually strong: "Note: {dimension} is usually a strength. This dip is likely a blind spot -- something they didn't notice."}
```

### Output Handling

**Expected output:** 1-3 sentences of natural language. No formatting, no bullets, no scores.

**Post-processing:**

- Strip any markdown formatting the LLM adds
- Reject outputs longer than 500 characters (re-prompt with "shorter")
- Reject outputs that contain numbers/scores (re-prompt with "no scores, just describe what you heard")

**Fallback:** If LLM call fails, use a template:

```
"I noticed your {dimension} could use some attention in that last section. Try recording yourself and listening back -- sometimes it's hard to hear {dimension_description} while you're playing."
```

### Model Choice

**Primary:** OpenRouter API, which provides access to Claude, GPT-4, Llama, Gemini, and others via a single API. Switch models by changing a string -- no code changes needed.

**Model tiering (decided 2026-03-03):** The teacher LLM (this stage) uses a quality model (Sonnet/GPT-4o-class) for natural tone and persona following. The analysis subagent (stage 1) uses a fast/cheap model (Haiku/Flash-class). See `docs/apps/06a-subagent-architecture.md` for the full tiering rationale.

**Starting plan:**
- Claude Sonnet for quality (best at following nuanced persona instructions, natural tone)
- Test Haiku for speed (lower latency, lower cost)
- Experiment with GPT-4o, Gemini, and others via A/B testing

**Fallback:** If OpenRouter is down, use Cloudflare Workers AI (Llama 3.3 70B) -- free, co-located with Workers, lower quality but zero external dependency.

**Latency budget:** LLM call should complete in <2 seconds. With structured input (not long context), this is achievable with OpenRouter.

### "Tell Me More" Flow

If the student taps "Tell me more" after the initial observation:

Second prompt adds:

```
The student wants to know more about this. Elaborate with:
1. Why this matters for this piece/style
2. A specific practice technique they can try right now
3. What "fixed" would sound/feel like

Still conversational. 2-4 sentences.
```

### What This Slice Does NOT Include

- Exercise recommendation (Slice 7+8 -- focus mode)
- Piece identification
- Chat/conversation flow (student asking arbitrary questions)
- Voice output

### Tasks

**Task 1: Draft and test the system prompt**

- Write the teacher persona prompt
- Test with synthetic teaching moment data across different scenarios:
  - Cold start student, dynamics issue
  - Warm student, blind spot in pedaling
  - Warm student, known weakness in timing
- Iterate on tone and specificity
- Use the quote bank as reference for what real teacher feedback sounds like

**Task 2: Build the prompt template engine**

- Implement dynamic context block construction (session, student)
- Handle cold start vs. warm states
- Handle missing data gracefully (no piece info, no goals, etc.)

**Task 3: Implement LLM call**

- API integration via OpenRouter (single API key, model selectable per-request)
- Post-processing: strip formatting, length check, score check
- Fallback to Workers AI (Llama 3.3 70B) if OpenRouter is unavailable
- Fallback template for total LLM failure
- Measure latency

**Task 4: Implement "Tell me more" follow-up**

- Second-turn prompt with elaboration instructions
- Same LLM call pattern

**Task 5: End-to-end test with real data**

- Use masterclass moment data as input
- Generate observations for 20+ moments
- Compare to actual teacher feedback from the masterclass transcripts
- Qualitative assessment: does it sound like a teacher?

### Open Questions

1. ~~Which OpenRouter model performs best for the teacher persona?~~ Partially resolved: model tiering decided (Haiku for subagent, Sonnet for teacher). A/B testing within each tier remains open.
2. Should the teacher's tone adapt to student level? (more encouraging for beginners, more direct for advanced?) -- Partially addressed by the subagent reasoning framework's encouragement calibration. See `docs/apps/06a-subagent-architecture.md`.
3. How to handle the case where all chunks score low on STOP probability (the student played well)? "Sounded good" with no observation? Or find something constructive anyway?
4. How should the teacher prompt template change now that it receives pre-analyzed reasoning from the subagent instead of raw scores? The user prompt template above was designed for raw data input -- it will need adaptation for the handoff format.
