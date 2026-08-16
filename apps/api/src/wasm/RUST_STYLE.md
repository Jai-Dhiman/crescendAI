# Rust Style Guide

## CrescendAI WASM crates -- piece-identify, score-analysis

Scope: every Rust file under `apps/api/src/wasm/`. Both crates are synchronous
compute over owned per-call data, compiled to `wasm32-unknown-unknown` and run
inside Cloudflare workerd, plus one native bin (`dtw_chunk_cli`).

Normative statements carry a stable id and an RFC 2119 level. The compact index
agents and `scripts/standards_check.py` load is `docs/standards/rules.json`;
cite the id when reporting a violation. Mechanical enforcement lives in each
crate's `[lints]` table and `clippy.toml`; run `just lint-rust`.

Governing philosophy: minimum code that solves the problem. Nothing
speculative: no abstractions for single-use code, no unrequested flexibility,
no error handling for impossible scenarios. Explicit failures over silent
fallbacks.

## 1. Errors

- **[RS-ERR-001]** (MUST) Non-test code does not call `.unwrap()` or
  `.expect()` on a `Result`/`Option` that can fail at runtime from external
  input (JS boundary args, file data). Return `Err` instead -- a panic kills
  the whole wasm instance. Clippy: `unwrap_used`, `expect_used`
  (`allow-unwrap-in-tests` covers test code).
- **[RS-ERR-002]** (SHOULD) An invariant-backed `.expect()` cites the exact
  check in its message and sits within a few lines of that check
  (see `score-analysis/src/lib.rs:92`).
- **[RS-ERR-003]** (MUST) Float comparator closures use `total_cmp`, never
  `partial_cmp(..).unwrap()`. NaN in audio/chroma data is a real input to this
  pipeline, not a theoretical one.
- **[RS-ERR-004]** (MUST) No `panic!`, `unreachable!`, `todo!`, or
  `unimplemented!` outside `#[cfg(test)]`. `todo!` in particular is stub code
  shipped as finished. Clippy: `panic`, `todo`, `unimplemented`, `unreachable`.
- **[RS-ERR-005]** (MUST) Every `#[wasm_bindgen]`-exported function returns
  `Result<T, JsValue>` or a JSON `String`; it never panics across the
  boundary.

## 2. The WASM boundary

- **[RS-WASM-001]** (MUST) A value crossing the boundary that contains a
  `bool` or `Option<T>` field alongside other args in the same call goes
  through `serde_json::to_string`/`from_str`, never
  `serde_wasm_bindgen::to_value`/`from_value`. This codifies the workerd
  externref-aliasing corruption we already shipped once (see the inline
  comments at `score-analysis/src/lib.rs:60-63,100-112`).
- **[RS-WASM-002]** (MUST) Every exported function carries a `///` doc comment
  stating its parameter and return shapes as the JS caller sees them. The
  TS-side bridge is checked against this contract (TS-TYPE-002's SAFETY
  comments cite it).
- **[RS-WASM-003]** (SHOULD) No convenience-overload exports: a new exported
  function that only re-shuffles another export's argument shape is
  unrequested flexibility at the most expensive boundary to change.

## 3. Dependencies and escape hatches

- **[RS-DEP-001]** (MUST) No `Rc`, `RefCell`, `Arc`, `Mutex`, or `unsafe`
  without a comment stating the specific ownership or aliasing problem being
  solved. These crates own all their data per call; shared mutable state has
  no current use. `unsafe_code = "forbid"` is set crate-wide.
- **[RS-DEP-002]** (MUST) A new `#[allow(clippy::...)]` carries a same-line or
  preceding-line comment naming why the lint does not apply (the existing
  allows in `note_align.rs` and `teaching_moments.rs` are the pattern).

## 4. Minimum code

- **[RS-STYLE-001]** (SHOULD) The repeated
  `.map_err(|e| JsValue::from_str(&e.to_string()))` boilerplate collapses into
  one shared `fn js_err(e: impl std::fmt::Display) -> JsValue` per crate --
  a free function, not a trait or macro.
- **[RS-STYLE-002]** (MUST) No speculative `derive`s, traits, or generic
  parameters: a derive a type's current usage does not need, or a trait with
  one implementor, is flexibility nobody requested. Current derive lists
  (`Debug, Clone, Serialize, Deserialize` subsets) are the ceiling, not the
  floor.

## 5. Reference implementation

`score-analysis/src/bin/dtw_chunk_cli.rs` is the in-repo model for error
handling: `fn main() -> ExitCode`, a `match` on every parse failure, zero
unwraps. Follow it rather than inventing a new pattern.
