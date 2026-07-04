# FRONT 7a timing-claim extraction prompt (LLM extractor; truth label is NEVER an LLM)

Used by Sonnet 4.6 subagents to decompose generator prose into structured **timing**
claims for the FRONT-7 claim-supply probe (#101 / #67). The extractor is allowed to be
an LLM (Path #1 rule: the *claim* may be LLM-extracted; the *verdict/truth label* may
not). Only timing claims are emitted; other dimensions are ignored for this front.

## Task

You are given a JSON array of generator feedback documents, each:
`{"recording_id": str, "run_id": str, "skill_bucket": int, "text": str}`.

For EACH document, extract every atomic claim the text makes about the pianist's
**timing** -- WHEN notes happen: tempo, pulse, pacing, rushing/dragging, steadiness,
rubato, rhythmic figures. Output a JSON array of claim objects. A document may yield
zero, one, or several timing claims. Do not invent claims the text does not make.

**Do NOT emit** claims that are fundamentally about something other than when-notes-happen:
loudness/dynamics, tone/timbre, pedaling, note connection (legato/staccato as
articulation), voicing/balance, phrasing-as-shaping, or wrong-notes/accuracy. If a
sentence is ambiguous between timing and another dimension, only emit it if the timing
reading is clearly present.

## The subtype distinction (critical -- this is the whole probe)

The FRONT-7 statistic is **signed mean onset-deviation vs the aligned score**
(rushing = notes systematically EARLIER than the score; dragging = systematically
LATER). It is a signed MEAN. It measures a *direction*, not a spread, not an aesthetic,
not a note-duration. Tag each timing claim with exactly one `timing_subtype`:

- `rush_drag` -- DIRECTIONAL tempo/pacing vs the expected timing. The performance is
  ahead/behind, fast/slow, hurried/labored, pushing/holding-back, or (neutral) well-paced.
  Examples: "you rushed the runs", "it dragged in the middle", "the tempo was too fast",
  "pushing ahead of the beat", "steady, well-controlled pulse", "don't hurry the coda".
  THIS IS THE ONLY IN-SCOPE SUBTYPE.

- `evenness` -- STEADINESS/CONSISTENCY of the pulse; wobble, unevenness, fluctuation
  with no single direction. Examples: "your sixteenths were uneven", "the pulse wasn't
  steady", "inconsistent tempo throughout", "rhythmically wobbly". A signed MEAN
  deviation is ~0 for symmetric unevenness -> a variance/spread claim, NOT signed. OUT.

- `rubato` -- EXPRESSIVE timing flexibility as an aesthetic, not an error. Examples:
  "beautiful rubato", "needs more rubato", "give the melody more time", "the push-pull
  felt natural", "too metronomic, breathe more". Intended deviation vs a metronomic
  score reads as "error" under a signed-vs-score statistic -> not falsifiable this way. OUT.

- `note_value` -- accuracy of RHYTHMIC FIGURES / note durations, not onset-vs-score.
  Examples: "the dotted rhythm flattened out", "triplets became duplets", "hold the half
  note its full value", "the rhythm of the theme was wrong". A duration/articulation
  matter (FRONT 8), NOT signed onset deviation. OUT.

- `hesitation` -- a momentary STOP/PAUSE/STUMBLE at a spot, not a sustained pace.
  Examples: "you hesitated before the climax", "a little stumble in bar 20", "brief
  pause crept in". Borderline (a localized drag) -- tagged separately, reported, but not
  counted as headline in-scope. OUT of the headline.

- `ambiguous` -- generic timing/rhythm mention with no directional, evenness, rubato, or
  figure cue. Examples: "work on your timing", "nice sense of rhythm", "rhythmic control
  needs attention". OUT.

Tag honestly by what the sentence actually claims. Prefer the more specific subtype; use
`ambiguous` only when no cue is present.

## Polarity (only meaningful for `rush_drag` and `hesitation`)

- `"rush"` -- the performance is AHEAD / too fast / hurried / pushing forward.
- `"drag"` -- the performance is BEHIND / too slow / labored / holding back / (for
  hesitation) pausing/stalling.
- `"neutral"` -- the pacing is asserted well-judged / steady / good tempo control.
- `"n/a"` -- for `evenness`/`rubato`/`note_value`/`ambiguous` (polarity not applicable).

Convert prescriptive advice to the implied descriptive state: "don't rush the coda" ->
the performance IS rushing (`rush`); "take more time in the melody" is rubato (`n/a`);
"keep the tempo steadier" is `evenness` (`n/a`).

## Location

- `"whole_piece"` unless the text gives EXPLICIT bar numbers (e.g. "bars 9-12", "bar 20").
- Only emit a region `{"bar_start": int, "bar_end": int}` when literal bar numbers appear.
  Section names ("the development", "the coda", "the opening") are NOT localizable ->
  use `"whole_piece"`.

## Output schema (STRICT)

Output ONLY a JSON array (no prose, no markdown fence), each element:

```
{
  "recording_id": "<echo>",
  "run_id": "<echo>",
  "proposition": "<short verbatim-or-close span of the claim>",
  "dimension": "timing",
  "timing_subtype": "rush_drag" | "evenness" | "rubato" | "note_value" | "hesitation" | "ambiguous",
  "location": "whole_piece" | {"bar_start": int, "bar_end": int},
  "polarity": "rush" | "drag" | "neutral" | "n/a",
  "rationale": "<one clause: why this subtype+polarity>"
}
```

If a document makes no timing claim, emit nothing for it. Return `[]` if the whole batch
has none.
