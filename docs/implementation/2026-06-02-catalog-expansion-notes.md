# Implementation Notes — Catalog Expansion

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Pre-build environment findings (controller)

- `model/data/scores/*.json` is GITIGNORED (`model/data/.gitignore: /scores/*.json`, only `titles.json` tracked). The 244 existing score JSONs are NOT in git; they live only in the developer's main working tree and are regenerated locally. The fresh worktree therefore starts with ZERO ingested score JSONs in `model/data/scores/`.
- `model/data/fingerprints/ngram_index.json` and `rerank_features.json` ARE tracked.
- Consequence for Group D: `git add model/data/scores/` (D3 commit) is a no-op for ignored files. The committable artifacts of Group D are the manifest, lockfile, eval_piece_map.json, fingerprints, and justfile. The 11 score JSONs remain gitignored by design (consistent with the existing 244).
- Consequence for fingerprint regen: a correct 255-piece fingerprint requires the 244 existing scores to be present in the worktree's `model/data/scores/`. They are not present in a fresh worktree. This is a Group-D execution dependency surfaced for the executor.
- Baseline: 100 passed, 24 skipped in score_library suite.
- Verified dependency surfaces: schema fields (ScoreNote/Bar/ScoreData), Scores.root, DATA_ROOT, parse_score_midi(midi_path, piece_id, composer, title) all match the plan.

## Build-controller deviation: no subagent-dispatch tool available

The build skill assumes a `Task` tool to dispatch fresh per-task implementer/reviewer subagents. This environment exposes no such dispatcher (only the TaskList task-tracking tools). The controller therefore executed each task directly while strictly enforcing the build discipline per task: (1) write the failing test, (2) run and watch it FAIL for the right reason, (3) implement exactly the plan's code, (4) run and watch it PASS, (5) commit with the plan's exact message. After each task, a spec-compliance check (re-read diff vs plan requirements) and a code-quality self-review were performed before moving on. Sonnet-4.6-subagent convention is moot since no subagents were spawned.

## Task G0-1 (catalog_coverage.py) — DONE
Implemented exactly as planned. 6 tests pass. CANONICAL_MAP = 16 entries. No deviations.

## Group A (validate.py A1-A5) — DONE
All 5 checks implemented exactly as planned: min_notes/total_bars/monotonic_onsets, pitch_range, bar_count, quantization (16th-grid via start_tick deltas), key_agreement (Krumhansl-Schmuckler, enharmonic tonic map). 19 tests pass. No chroma. The +60-tick fixed-offset quant fixture yields median 0.125 > 0.10 (violation); clean grid 0.0 (pass) — matches challenge re-review. No deviations.

## Group B (manual.py B1-B4) — DONE
ingest_manifest with temp-staging all-or-nothing atomicity (CONCERN 2 fix): staged JSONs only moved into scores_dir after ALL pieces resolve; lockfile written last; HALT leaves scores_dir + lockfile untouched. B2-B4 are behavior-locking tests over B1's impl (passed on first run, as the plan predicted; watch-it-fail satisfied at B1 module-absence). 4 tests pass. Semgrep flagged urllib dynamic-URL (WARNING) on _http_fetch — accepted: plan mandates stdlib urllib (requests/httpx not deps), URLs from pinned manifest, line carries noqa S310. No deviations.

## Group C (cli.py parse-manual) — DONE, with one required deviation
cmd_parse_manual + parse-manual subparser (--manifest required, --lock default None) + dispatch entry added. cmd_fingerprint unchanged.

DEVIATION (necessary): The plan's C1 test monkeypatches `score_library.manual._http_fetch`, but B1's `ingest_manifest` had `fetch_fn=_http_fetch` as a DEFAULT ARG, which binds the original function at def-time — the monkeypatch had no effect and the test hit the real network (HTTP 403). Fixed by changing the signature to `fetch_fn=None` and resolving `fetch_fn = _http_fetch` at call-time inside the function. This is the standard Python idiom for late-bound injectable defaults, is backward-compatible (all 4 B-tests that pass an explicit fetch_fn still pass), and is the minimal change that satisfies the plan's verbatim C1 test. The plan's B1 code + C1 test were mutually inconsistent on this point; the test is the contract, so the impl was corrected.

## Group D (data population) — PARTIAL: 8 of 11 pieces resolved

### Resolved (8/11) — real verified Mutopia PD MIDIs, all PASS the gate
All sourced from mutopiaproject.org FTP (LilyPond-engraved, metric timing, CC0/PD). Each parsed + passed validate_score (key-agreement, bar-count, quantization, DoD-min):
- bach.inventions.1            -> BachJS/BWV772/bach-invention-01/bach-invention-01.mid (22 bars, 458 notes)
- beethoven.fur_elise          -> BeethovenLv/WoO59/fur_Elise_WoO59/fur_Elise_WoO59.mid (105 bars, 905 notes)
- mozart.piano_sonatas.16-1    -> MozartWA/KV545/K545-1/K545-1.mid (73 bars, 1288 notes)
- schumann.kinderszenen.7      -> SchumannR/O15/SchumannOp15No07/SchumannOp15No07.mid (25 bars, 339 notes)
- debussy.suite_bergamasque.3_clair_de_lune -> DebussyC/L75/debussy_Ste_Bergamesq_Clair/debussy_Ste_Bergamesq_Clair.mid (72 bars, 1468 notes)
- debussy.deux_arabesques.1    -> DebussyC/L66/debussy_Arabesque_1/debussy_Arabesque_1.mid (107 bars, 1448 notes)
- chopin.fantaisie_impromptu   -> ChopinFF/O66/chopin_fantaisie-impromptu/chopin_fantaisie-impromptu.mid (138 bars, 3014 notes)
- rachmaninoff.preludes_op_3.2 -> RachmaninoffS/O3/rach-prelude-op3-no2/rach-prelude-op3-no2.mid (62 bars, 1725 notes)

### UNRESOLVED (3/11) — no verified PD engraved MIDI located (exhaustive search of the full 5,681-file Mutopia GitHub tree)
- chopin.waltzes.64-2 (Waltz Op.64 No.2, C# minor): ABSENT from Mutopia. Only Op.64 No.1 (Minute Waltz) is published. kunstderfuge forbidden by plan; piano-midi.de returns HTTP 418 (bot-blocked) and serves performance-timed MIDIs the gate is designed to reject. No fabricated URL substituted.
- liszt.liebestraume.3 (Liebestraum No.3, S.541, Ab major): ABSENT from Mutopia entirely (only Consolations S.172 + a Ballade exist for Liszt). Same fallback constraints; not sourced.
- beethoven.piano_sonatas.14-1 (Moonlight Mvt1, C# minor): Mutopia has only LilyPond .ly source for the piano version (no rendered piano .mid on the FTP); the only published Op.27-No.2 .mid is a guitar-duo arrangement (wrong instrument). Not sourced.

### Machinery verification against real data
- parse-manual CLI ran end-to-end against live network: 8 score JSONs written to data/scores/, lockfile with 8 real sha256s.
- Full 11-piece manifest run (with the 3 UNRESOLVED placeholder sources) correctly did NOT write a lockfile and left NO stray files in scores_dir (temp-staging all-or-nothing held: verified scores_dir stayed at 251 fingerprintable, no .mid/tmp leftovers). The 3 placeholders are non-URL strings, so _http_fetch raised URLError (expected: no fabricated URL to fetch).
- B4 unit test already proves the SourceResolutionError failure-table path against synthetic all-fail MIDIs.

### Counts (reconciled)
- Base catalog (copied from main tree, gitignored regenerable data): 243 scores (NOT 244 — the plan/challenge's 244 was off by one; authoritative fingerprint membership = *.json minus titles.json+seed.sql).
- After +8 ingested: fingerprint built over 251 scores (243+8). Had all 11 resolved it would be 254. The "+N ingested" invariant holds.
- eval_piece_map.json: full 16-entry contract (independent of resolution count).

### catalog-verify result
13 of 16 PASS (8 newly ingested + 5 already-in-ASAP: bach_prelude, chopin_ballade, chopin_etude, pathetique, nocturne). 3 FAIL as MISSING: chopin_waltz_csm, liszt_liebestraum_3, moonlight_sonata_mvt1. Exit code 1 (correct: honest partial state).

### Committed artifacts
justfile (catalog-verify recipe), manual_scores.json (11 entries, 3 marked UNRESOLVED), manual_scores.lock.json (8 real entries), eval_piece_map.json (16), regenerated fingerprints (over 251). Score JSONs remain gitignored (consistent with existing 243).
