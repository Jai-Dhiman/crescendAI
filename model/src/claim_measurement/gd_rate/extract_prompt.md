# G-D dynamics-claim extraction prompt (LLM extractor; truth label is NEVER an LLM)

Used by Sonnet 4.6 subagents to decompose generator prose into structured **dynamics**
claims for the G-D faithfulness rate (#101 / #67). The extractor is allowed to be an LLM
(Path #1 rule: the *claim* may be LLM-extracted; the *verdict/truth label* may not). Only
dynamics claims are emitted; other dimensions are ignored for this front.

## Task

You are given a JSON array of generator feedback documents, each:
`{"recording_id": str, "run_id": str, "skill_bucket": int, "text": str}`.

For EACH document, extract every atomic claim the text makes about the pianist's
**dynamics** (loudness). Output a JSON array of claim objects. A document may yield zero,
one, or several dynamics claims. Do not invent claims the text does not make.

## The level-vs-contrast distinction (critical)

The verifier measures **mean loudness LEVEL** (mean note velocity). It does NOT measure
dynamic contrast/range/shape. Tag each dynamics claim:

- `dynamics_subtype = "level"` — about how loud/soft/projected/powerful the playing is.
  Examples: "too timid, project more", "wonderful power", "the sound never filled the
  hall", "play the opening more softly".
- `dynamics_subtype = "contrast"` — about dynamic RANGE, shaping, swell/ebb, crescendo,
  evenness, terracing. Examples: "the melody felt evenly lit", "needs more ebb and flow",
  "shape the phrase with a swell", "your crescendos lacked drama".
- `dynamics_subtype = "ambiguous"` — generic dynamics praise/critique with no level or
  shaping cue ("nice dynamics", "work on your dynamics").

Only `level` claims are in scope for the current statistic; `contrast`/`ambiguous` are
recorded but will abstain (`out_of_scope_statistic`). Tag honestly.

## Polarity (the asserted state of THIS performance)

Convert prescriptive advice into the implied descriptive state of the performance.

For `level` claims:
- `"+"` — the performance IS loud / projected / powerful (incl. "too loud, rein it in").
- `"-"` — the performance IS soft / timid / under-projected (incl. "project more",
  "needs more sound" → it is currently too soft).
- `"neutral"` — the level is asserted balanced / appropriate / well-judged.

For `contrast`/`ambiguous` claims set polarity by the same +/-/neutral logic on range
(wide=+, flat=-, balanced=neutral); it will not be verified but keep it consistent.

## Location

- `"whole_piece"` unless the text gives EXPLICIT bar numbers (e.g. "bars 9-12").
- Only emit a region `{"bar_start": int, "bar_end": int}` when literal bar numbers appear.
  Section names ("the development", "the coda") are NOT localizable -> use `"whole_piece"`.

## Output schema (STRICT)

Output ONLY a JSON array (no prose, no markdown fence), each element:

```
{
  "recording_id": "<echo>",
  "run_id": "<echo>",
  "proposition": "<short verbatim-or-close span of the claim>",
  "dimension": "dynamics",
  "dynamics_subtype": "level" | "contrast" | "ambiguous",
  "location": "whole_piece" | {"bar_start": int, "bar_end": int},
  "polarity": "+" | "-" | "neutral",
  "rationale": "<one clause: why this subtype+polarity>"
}
```

If a document makes no dynamics claim, emit nothing for it. Return `[]` if the whole
batch has none.
