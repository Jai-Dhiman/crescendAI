# Score Library Catalog — Adding & Regenerating Scores

How the score-library catalog is sourced, stored, and extended. The catalog is the
set of per-piece score JSONs (`model/data/scores/*.json`) that the fingerprint
index and the #21 chroma-identification harness are built from.

## The catalog is generated, not committed

`model/data/scores/*.json` is gitignored (`model/data/.gitignore:10`). The score
JSONs are build artifacts. What IS committed and authoritative:

| Committed artifact | Role |
|---|---|
| `model/data/manifests/manual_scores.json` | Intent: each piece's metadata + ranked source list |
| `model/data/manifests/manual_scores.lock.json` | Resolved source URL/path + sha256 per piece (uv.lock semantics) |
| `model/data/manifests/manual_midis/*.mid` | Committed local-source MIDIs (the "local" source type) |
| `model/data/evals/piece_id/eval_piece_map.json` | The 16-piece #21 eval contract |
| `model/data/fingerprints/*.json` | N-gram index + rerank features built from the scores |

The 243-piece ASAP base catalog is parsed separately (`parse` subcommand from a
cloned ASAP dataset); the 11 manual pieces come from the manifest. To rebuild the
generated scores on any fresh checkout:

```bash
cd model && uv run python -m score_library.cli parse-manual \
  --manifest data/manifests/manual_scores.json
just fingerprint
```

Fingerprint trigram counts can drift +/-1 between checkouts because the gitignored
ASAP base scores vary slightly; the committed fingerprint is authoritative for
whatever generated it.

## Adding a score to the catalog

Append one entry to `manual_scores.json`, then run `just catalog-add` (parse-manual
-> fingerprint -> reports catalog size). The validation gate inside `parse-manual`
HALTS loudly if any source is bad or off-grid — it never writes a partial catalog.

An entry:

```json
{
  "slug": "human_readable_slug",
  "piece_id": "composer.collection.number",
  "composer": "Full Name",
  "title": "Work title, key, opus",
  "expected_key": "C# minor",
  "expected_bars": 129,
  "license": "PD (...)",
  "sources": ["<source 1>", "<source 2>"]
}
```

`sources` is a RANKED list. The first source that parses AND passes the validation
gate wins; its (url/path + sha256) is pinned to the lockfile. Two source types:

- **URL-sourced** (`http(s)://...`): a public-domain engraved MIDI, e.g. from
  Mutopia. Fetched over the network at ingest time, sha256-pinned thereafter.
- **Local-sourced** (a relative path like `manual_midis/foo.mid`): resolved
  relative to the manifest's directory, path-traversal-guarded. Use this when no
  PD engraved-MIDI URL exists. Commit the `.mid` under
  `model/data/manifests/manual_midis/` — the `.gitignore` `*.mid` rule is
  overridden there (`model/.gitignore`), so no `git add -f` is needed. A missing
  local file raises a clear `SourceResolutionError` naming the resolved path.

Three pieces are local-sourced because they have no PD engraved-MIDI URL (Mutopia
lacks them; piano-midi.de serves performance MIDIs the gate rejects):
`chopin.waltzes.64-2`, `liszt.liebestraume.3`, `beethoven.piano_sonatas.14-1`.

## The validation gate (do not weaken)

`validate_score` is deliberately chroma-FREE so it does not rig the #21 chroma
harness. Primary discriminators: Krumhansl-Schmuckler key agreement + bar-count
plausibility band `[0.7x, 2.2x]`. Secondary backstops: pitch range + a coarse
16th-grid quantization check (sixteenth-units, meter-independent, threshold 0.4) —
a low-power gross/fixed-offset guard that rejects performance-timed MIDIs, NOT a
rubato detector. It never trusts the MIDI `key_signature` meta (defaulted/unreliable).

## The lockfile (sha256 semantics)

`manual_scores.lock.json` pins each piece's resolved source + sha256. On every
ingest, if a piece is already pinned, the fetched/read bytes must hash to the
pinned sha256 or that candidate is rejected (`hash_mismatch`) and the next ranked
source is tried. Ingest is all-or-nothing: winning JSONs stage in a temp dir and
only move into `data/scores/` after EVERY piece resolves; a mid-run HALT leaves
`data/scores/` and the lockfile untouched. To intentionally re-pin (e.g. a better
source), delete the piece's lockfile entry and re-run `catalog-add`.

## Catalog vs. eval set — two different things

Do not conflate them:

- **The general catalog** grows freely via the manifest + `just catalog-add`. Add
  as many pieces as you like; no code change.
- **The fixed 16-piece eval set** is `CANONICAL_MAP` in
  `model/src/score_library/catalog_coverage.py` plus `eval_piece_map.json`. It is
  the #21 chroma-harness acceptance contract, asserted by `just catalog-verify`.
  It is code-hardcoded and deliberately high-friction: adding an eval piece
  redefines what the harness is benchmarked against, so it must change in lockstep
  with `eval_piece_map.json`. Adding a catalog score is NOT adding an eval piece.
