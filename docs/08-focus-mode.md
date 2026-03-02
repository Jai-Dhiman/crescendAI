# Slice 8: Focus Mode

See `docs/architecture.md` for the full system architecture.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A guided practice mode where the system identifies a weak dimension, presents targeted exercises (from the student's own music), evaluates each attempt, and closes the feedback loop.

**Architecture:** Focus mode is triggered by the system (after identifying a persistent weakness) or by the student ("I want to work on dynamics"). It draws from the exercise database (Slice 7), presents exercises, and uses MuQ to evaluate attempts on the target dimension specifically. MuQ inference during focus mode exercises runs on-device via Core ML, same as regular practice.

**Tech Stack:** Existing MuQ inference, Exercise DB (Slice 7), Teacher LLM (Slice 6), iOS UI (Slice 9)

---

## Context

Focus mode is the teaching loop: observe -> identify -> diagnose -> prescribe -> evaluate. It transforms the app from a passive listener into an active teacher. This is where exercises from the student's preferred music get used.

## Design

### Entry Points

**System-initiated (after sessions 5+):**

- System detects a dimension that has been the top teaching moment in 3+ of the last 5 sessions
- At the end of a session: "I've noticed {dimension} keeps coming up. Want to do a focused session on it next time?"
- Student can accept, dismiss, or say "later"

**Student-initiated:**

- Student says or taps: "I want to work on my dynamics" / "Let's focus on pedaling"
- System enters focus mode for that dimension immediately

### Focus Session Flow

1. **Introduction** (Teacher LLM generates):
   > "Let's work on your pedaling. I've been hearing some harmonic bleed in your Chopin -- the sustain is carrying across chord changes. We'll do 3 exercises to sharpen that."

2. **Exercise 1: Curated exercise from DB**
   - Query exercise DB: target_dimension, student level, repertoire match
   - Present: title, instructions, notation (if available)
   - Student plays the exercise
   - MuQ evaluates the attempt, focused on target dimension
   - Teacher LLM gives brief feedback: "Better -- the chord change at the top was cleaner. Try it once more and really listen for the bass note ringing over."

3. **Exercise 2: Custom exercise from student's piece**
   - LLM generates an exercise using the student's actual passage where the issue was heard
   - "Now take bars 20-24 of your Nocturne. Play just the LH with pedal. Lift and re-engage on each new harmony. Then add the RH and keep the same pedal discipline."
   - Student plays
   - MuQ evaluates
   - Teacher feedback

4. **Exercise 3: Integration**
   - "Play through the full passage (bars 18-28) with everything we just worked on."
   - Student plays the actual music
   - MuQ evaluates
   - Compare dimension score to the original teaching moment that triggered focus mode
   - Teacher LLM: "That's markedly cleaner. The harmonic changes in bars 22-23 aren't bleeding anymore. Keep this awareness next time you run through the whole piece."

5. **Wrap-up**
   - Record dimension scores before/after in student_exercises
   - Update student model: "Focused on pedaling in session N"
   - Suggest: "Try running through the piece now and I'll listen for whether the pedaling holds in context."

### Evaluation During Focus Mode

MuQ inference runs on-device (Core ML) on each exercise attempt -- no cloud call needed for scoring. Feedback is weighted toward the TARGET DIMENSION only.

- If the student is working on pedaling, don't comment on timing even if it's worse than usual
- Only surface non-target observations if something is severely off (STOP probability > 0.95 on a different dimension)
- This keeps the session focused and avoids overwhelming the student

### Session State

```json
{
    "mode": "focus",
    "target_dimension": "pedaling",
    "trigger": "system",  // or "student"
    "trigger_context": "Pedaling was the top teaching moment in 3 of last 5 sessions",
    "exercises": [
        {
            "type": "curated",
            "exercise_id": "ex-ped-003",
            "status": "completed",
            "attempts": 2,
            "target_dim_before": 0.35,
            "target_dim_after": 0.52
        },
        {
            "type": "generated",
            "exercise_id": "ex-gen-001",
            "status": "in_progress",
            "attempts": 1
        },
        {
            "type": "integration",
            "status": "pending"
        }
    ],
    "overall_improvement": null  // computed at wrap-up
}
```

### What This Slice Does NOT Include

- Notation rendering (Slice 9)
- MIDI playback of exercises
- Multi-session focus plans ("work on pedaling for the next 3 sessions")
- Adaptive exercise difficulty (start easy, get harder)

### Tasks

**Task 1: Focus mode trigger logic**

- Detect persistent weak dimensions from student session history
- Generate focus suggestion at session end
- Handle student acceptance/dismissal

**Task 2: Focus session orchestration**

- State machine: introduction -> exercise 1 -> feedback -> exercise 2 -> feedback -> integration -> wrap-up
- Handle student wanting to skip an exercise or end early
- Track attempts and scores per exercise

**Task 3: Custom exercise generation for focus mode**

- LLM prompt that generates passage-specific exercises
- Grounded in the teaching moment that triggered focus (which chunk, what dimension, what piece)

**Task 4: Focused evaluation**

- MuQ inference on exercise attempts
- Weight feedback to target dimension
- Compare before/after scores

**Task 5: Wrap-up and student model update**

- Record improvement (or lack thereof)
- Update student_exercises table
- Update dimension baseline if improvement is significant

### Open Questions

1. How many exercises per focus session? 3 feels right but may be too many for a student who's already been practicing for 45 minutes.
2. Should focus mode interrupt a regular practice session, or always be a separate "mini-session"?
3. What if the student doesn't improve during focus mode? Encouraging message + suggest trying again tomorrow? Or adapt exercises?
